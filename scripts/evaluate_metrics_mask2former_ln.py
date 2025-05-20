"""
evaluate_metrics_mask2former_ln.py

This script evaluates the performance of the Mask2Former model on whole slide images (WSIs) at the slide level. 
It does model inference, evaluation, and visualization of segmentation results. 
The script integrates with tools like QuPath for downstream analysis.

Key Features:
- **Model Inference**: Loads a Mask2Former model from a checkpoint and performs segmentation on WSIs.
- **Evaluation Metrics**: Computes IOU, MCC, and Dice metrics for each class and identifies 
  the worst-performing samples.
- **Visualization**: Generates side-by-side visualizations of ground truth and predicted masks, color-coded 
  by class, and saves them as PNG files.
- **QuPath Integration**: Prepares GeoJSON outputs compatible with QuPath for further analysis.
- **Batch Processing**: Supports distributed processing using SLURM array jobs for efficient handling of 
  large datasets.

Authors: Javier Garcia-Baroja
Contact: javier.garcia@unibe.ch
Affiliation: Institute of Tissue Medicine and Pathology (ITMP), University of Bern, Switzerland
License: <Insert license information here>
Date: <Insert date here>
"""


import openslide

import os
import sys
import glob
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.dirname(SCRIPT_DIR))

import argparse

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils.models import create_mask2former_from_checkpoint
from utils.utils import divide_list_slurm_array, get_ann_files, check_wsi_exists_all_formats, combine_results, save_geojson_annotation, COLORMAP
from utils.inference import evaluate_wsi_slide

def print_summary(results:pd.DataFrame, output_dir:str, eval_class:str):
    """
    Generate and save a summary of evaluation metrics for a given class.

    This function processes the evaluation results, identifies the worst-performing cases,
    and computes the mean and standard deviation of metrics for each class. The results
    are saved to the specified output directory.

    Args:
        results (pd.DataFrame): A DataFrame containing evaluation metrics for each sample.
                                Expected to include columns for IoU scores per class.
        output_dir (str): Path to the directory where the summary and worst-case results
                          will be saved.
        eval_class (str): The class label for which the evaluation will be tracked. This should

    Outputs:
        - A CSV file named `worst_<class_label>.csv` containing the 20 worst-performing
          samples for the specified class, based on IoU scores.
        - A text file named `results_summary.txt` containing the mean and standard deviation
          of metrics for all classes in the format: `<metric>: <mean> +/- <std>`.
    """
    # save the results for the worst performing cases
    worst = results.sort_values(by=f'iou_{eval_class}', ascending=True).head(20)
    worst.loc[:, ['filename', f'iou_{eval_class}']].to_csv(os.path.join(output_dir, f'worst_{eval_class}.csv'), index=False)
    # now print the mean and std of the results per class
    summary = results.describe().loc[['mean', 'std']]
    # print as metric +/- std
    with open(os.path.join(output_dir, 'results_summary.txt'), 'w') as f:
        for col in summary.columns:
            f.write(f'{col}: {summary[col]["mean"]*100:.2f} +/- {summary[col]["std"]*100:.2f}\n')



def prepare_mask_png(pred:np.ndarray, gt:np.ndarray, category_dict:dict, output_dir:str):
    """
    Generate and save an overlay visualization of ground truth and predicted masks.

    This function creates a side-by-side visualization of the ground truth and predicted 
    segmentation masks, with each class color-coded according to the provided category 
    dictionary. The visualization is saved as an image file in the specified output directory.

    Args:
        pred (np.ndarray): The predicted mask as a 2D array, where each pixel value 
                           corresponds to a class ID.
        gt (np.ndarray): The ground truth mask as a 2D array, where each pixel value 
                         corresponds to a class ID.
        category_dict (dict): A dictionary mapping class labels (str) to their corresponding 
                              numeric IDs (int). The colors for each class are defined in 
                              the global `COLORMAP` variable.
        output_dir (str): Path to the directory where the overlay visualization will be saved.

    Outputs:
        - A PNG file containing the side-by-side visualization of the ground truth and 
          predicted masks, with a legend indicating the class labels and their colors.

    Raises:
        AssertionError: If the shapes of the ground truth and predicted masks do not match.
    """
    pred = cv2.resize(pred, (pred.shape[1] // 2, pred.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
    gt = cv2.resize(gt, (gt.shape[1] // 2, gt.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
    assert pred.shape == gt.shape, f'Pred shape {pred.shape} does not match gt shape {gt.shape}'
    
    gt_rgb = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    
    for label, id in category_dict.items():
        if id in gt:
            gt_rgb[gt == id] = COLORMAP[label]
        if id in pred:
            pred_rgb[pred == id] = COLORMAP[label]
    
    del gt, pred
    rgb = 255 * np.ones((pred_rgb.shape[0], pred_rgb.shape[1] * 2 + 1, 3), dtype=np.uint8)
    rgb[:, :pred_rgb.shape[1]] = gt_rgb
    rgb[:, pred_rgb.shape[1] + 1:] = pred_rgb

    patches = [mpatches.Patch(color=np.array(COLORMAP[k]) / 255, label=k) for k in category_dict.keys()]
    plt.imshow(rgb)
    plt.title('Ground Truth | Prediction')
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('off')
    plt.savefig(output_dir, bbox_inches='tight', pad_inches=0)
    plt.close()

def main(args):

    ann_files = get_ann_files(args.path_to_ann)

    assert len(ann_files) > 0, 'No annotation files found'
    if isinstance(args.checkpoint_path, str):
        assert os.path.exists(args.checkpoint_path), f'Model checkpoint path {args.checkpoint_path} does not exist'
    else:
        assert hasattr(args.checkpoint_path, 'forward'), 'Model checkpoint should be a path to a model checkpoint or a pytorch model'
    
    ann_wsi_pairs = []
    for ann_file in ann_files:
        wsi_name = os.path.splitext(os.path.basename(ann_file))[0] 
        if args.wsi_list is not None and wsi_name not in args.wsi_list:
            print(f'WSI for annotation {wsi_name} was not found in provided list')
            continue
        wsi_file_exists, wsi = check_wsi_exists_all_formats(wsi_name, args.wsi_path)
        if wsi_file_exists:
            ann_wsi_pairs.append((ann_file, wsi))
        else:
            print(f'WSI for annotation {wsi_name} was not found')


    # if no annotation files found, exit
    if len(ann_wsi_pairs) == 0:
        sys.exit('No annotation files found. Please check the path to annotation files and the WSI list')

    ann_wsi_pairs = divide_list_slurm_array(ann_wsi_pairs)   
    # Load model
    model = create_mask2former_from_checkpoint(checkpoint_path=args.checkpoint_path, label2id=args.label2id, encoder_name=args.encoder_model, decoder_model=args.decoder_model, out_indices=args.feature_layers)
    os.makedirs(args.output_dir, exist_ok=True)
    results_all = None
    for ann_file, wsi_file in ann_wsi_pairs:
        wsi_name = os.path.splitext(os.path.basename(ann_file))[0]
        try:
            print(f'Processing {wsi_name}')
            results, pred, gt, level, level_downsampling = evaluate_wsi_slide(
                model=model,
                wsi_path=wsi_file,
                annotation_path=ann_file,
                batch_size=args.batch_size,
                tile_size=args.tile_size,
                step_size=args.step_size,
                resolution=args.resolution,
                label2id=args.label2id,
                crop_pred_edge=args.crop_pred_edge,
                apply_post_processing=args.apply_post_processing,
            )
            if results_all is None:
                results_all = pd.DataFrame(data=results, index=[0])
            else:
                results_all.loc[len(results_all)] = results

            if args.prepare_qupath:
                save_geojson_annotation(out_path=os.path.join(args.output_dir, f'{wsi_name}.geojson'),
                                        mask=pred,
                                        level=level,
                                        level_downsampling=level_downsampling,
                                        category_dict=args.label2id)
            
            if args.prepare_overlay:
                prepare_mask_png(pred=pred,
                                gt=gt,
                                category_dict=args.label2id,
                                output_dir=os.path.join(args.output_dir, f'{wsi_name}.png'))
        except Exception as e:
            # print line where the error occurred
            exc_type, exc_obj, exc_tb = sys.exc_info()
            line_number = exc_tb.tb_lineno
            file_name = exc_tb.tb_frame.f_code.co_filename
            print(f'Error processing {wsi_name}: {e}, \nerror occurred in {line_number} of {file_name}')

    print(f'Finished processing {len(ann_wsi_pairs)} WSIs')
    # save the results
    slurm_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    results_all.to_csv(os.path.join(args.output_dir, f"results_{slurm_id}_finished.csv"), index=False)

    combine_results(args.output_dir)
    print_summary(results_all, args.output_dir, args.eval_class)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Infer mask2former on WSI')
    parser.add_argument('--wsi_path', type=str, help='Path to input WSI folder')
    parser.add_argument('--wsi_list', type=str, help="Path to TXT file with a list of WSI names. Format of file should be 'wsi_name_1.mrxs\nwsi_name_2.mrxs...'", default=None)
    parser.add_argument('--path_to_ann', type=str, help="Path to annotation file(s). If single file, provide 'path/to/file.geojson'. If multiple files, provide 'path/to/folder/*.geojson")
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--checkpoint_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--encoder_model', type=str, help='Encoder model')
    parser.add_argument('--decoder_model', type=str, help='Decoder model')
    parser.add_argument('--label2id', type=str, help='Dictionary mapping label to id')
    parser.add_argument('--feature_layers', type=str, help='List of indices of output feature maps', default='16,20,24,31')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=1)
    parser.add_argument('--tile_size', type=int, help='Tile size', default=224)
    parser.add_argument('--step_size', type=int, help='Step size', default=180)
    parser.add_argument('--crop_pred_edge', type=int, help='Step size', default=50)
    parser.add_argument('--eval_class', type=str, help='Class for which evaluation will be tracked', default='Lymph node')
    parser.add_argument('--resolution', type=float, help='Resolution', default=0.5)
    parser.add_argument('--apply_post_processing', action='store_true', help='Postprocess the output prediction before computing metrics')
    parser.add_argument('--prepare_qupath', action='store_true', help='Prepare QuPath compatible output')
    parser.add_argument('--prepare_overlay', action='store_true', help='Prepare sparse output')
    args = parser.parse_args()

    # process arguments
    args.label2id = {label.split(':')[0]: int(label.split(':')[1]) for label in args.label2id.split(',')}
    args.feature_layers = [int(i) for i in args.feature_layers.split(',')]
    if args.wsi_list is not None:
        assert os.path.exists(args.wsi_list), f'Provided WSI list {args.wsi_list} does not exist'
        with open(args.wsi_list, 'r') as f:
            args.wsi_list = f.read().split('\n')
            args.wsi_list = [os.path.splitext(os.path.basename(i))[0] for i in args.wsi_list if i != '']
            print(f'Processing {len(args.wsi_list)} WSI files from list {args.wsi_list}')
    assert args.eval_class in args.label2id.keys(), f'Provided class {args.eval_class} is not in label2id dictionary. Available classes are {args.label2id.keys()}'

    main(args)
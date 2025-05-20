"""
evaluate_metrics_mask2former_metastasis.py

This script evaluates the performance of the Mask2Former model on whole slide images (WSIs) for segmentation at the tile level. 
It does model inference, evaluation, and visualization of segmentation results. 
The script is designed to handle large-scale datasets and supports integration with distributed 
computing environments like SLURM.

Key Features:
- **Tile-Based Model Inference**: Performs segmentation on WSI tiles using a Mask2Former model loaded from a checkpoint.
- **Evaluation Metrics**: Computes metrics and identifies the worst-performing tiles for a given class.
- **Visualization**: Generates visualizations of the worst-performing tiles, including RGB, ground truth, 
  and predicted masks, color-coded by class.
- **Batch Processing**: Supports distributed processing using SLURM array jobs for efficient handling of 
  large datasets.
- **Result Summarization**: Aggregates evaluation metrics and generates summary statistics, including mean 
  and standard deviation.

This script is intended for academic use and is part of ongoing research in computational pathology. 
For questions or collaborations, please contact the authors.

Authors: Javier Garcia-Baroja
Contact: javier.garcia@unibe.ch
Affiliation: Institute of Tissue Medicine and Pathology (ITMP), University of Bern, Switzerland
License: <Insert license information here>
Date: <Insert date here>
"""

import openslide

import os
import sys
from glob import glob
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.dirname(SCRIPT_DIR))

import argparse
from typing import List, Tuple, Dict

import numpy as np
import zarr
import pandas as pd
import matplotlib.pyplot as plt

from utils.models import create_mask2former_from_checkpoint
from utils.utils import divide_list_slurm_array, COLORMAP, get_ann_files, check_wsi_exists_all_formats, combine_results
from utils.inference import evaluate_wsi_tiles, infer_tiles


def load_annontations(file_paths:List[str]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    rgbs = []
    gts = []

    for filename in file_paths:
        ann_file = zarr.open(filename)
        rgb = ann_file[:, :, :3]
        gt = ann_file[:, :, 3]
        rgbs.append(rgb)
        gts.append(gt)

    return rgbs, gts

def get_ann_file_paths(filenames:List[str], coords:List[List[int]], dataset_save_path:str) -> List[str]:
    file_paths = []
    for filename, coord in zip(filenames, coords):
        folder = os.path.join(dataset_save_path, filename)
        ann_filename = os.path.join(folder, f'*_x_{coord[0]}_{coord[1]}_y_{coord[2]}_{coord[3]}.zarr')
        ann_filename = glob(ann_filename)[0]
        file_paths.append(ann_filename)
    return file_paths


def save_worst_tiles(model, filenames:List[str], coords:List[List[int]], dice_vals:List[float], label2id:Dict[str,int], dataset_save_path:str, output_dir:str):
    """Save the worst performing tiles as png images.

    Args:
        model (_type_): Model to use for inference
        filenames (List[str]): list of filenames
        coords (List[List[int]]): coordinates of the tiles to define path to tiles. Formatted as [[x1, x2, y1, y2], ...]
        dice_vals (List[float]): dice values of the tiles
        label2id (Dict[str,int]): mapping of label to id, used for color mapping
        dataset_save_path (str): path to where tile dataset is saved
        output_dir (str): path to save the images
    """
    # remove png files in output_dir
    for filename in os.listdir(output_dir):
        if filename.endswith('.png'):
            os.remove(os.path.join(output_dir, filename))
    file_paths = get_ann_file_paths(filenames, coords, dataset_save_path)
    preds = infer_tiles(model, file_paths)
    rgbs, gts = load_annontations(file_paths)
    tile_size = rgbs[0].shape[0]

    for i in range(len(filenames)):
        arr_to_save = np.zeros((tile_size, tile_size*3+2, 3), dtype=np.uint8) # Grid of 3 images
        arr_to_save[:, :tile_size] = rgbs[i]
        # add red column to separate images
        arr_to_save[:, tile_size, 0] = 255
        gt_rgb = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        pred_rgb = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        for label, id in label2id.items():
            if id in gts[i]: # gt is of shape (224, 224) with values in args.label2id.values()
                gt_rgb[gts[i] == id] = COLORMAP[label]
            if id in preds[i]:
                pred_rgb[preds[i] == id] = COLORMAP[label]
        arr_to_save[:, 1+tile_size:2*tile_size+1] = gt_rgb
        # add red column to separate images
        arr_to_save[:, 2*tile_size+1, 0] = 255
        arr_to_save[:, 2+2*tile_size:] = pred_rgb
        plt.imshow(arr_to_save)
        plt.axis('off')
        plt.title(f"Dice: {dice_vals[i]*100:.2f} | RGB | GT | Pred")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{filenames[i]}_{coords[i][0]}_{coords[i][1]}_{coords[i][2]}_{coords[i][3]}.png"))


def main(args):

    ann_files = get_ann_files(args.path_to_ann)

    assert len(ann_files) > 0, 'No annotation files found'
    if isinstance(args.checkpoint_path, str):
        assert os.path.exists(args.checkpoint_path), f'Model checkpoint path {args.checkpoint_path} does not exist'
    else:
        assert hasattr(args.checkpoint_path, 'forward'), 'Model checkpoint should be a path to a model checkpoint or a pytorch model'
    
    keep = []
    for ann_file in ann_files:
        wsi_name = os.path.splitext(os.path.basename(ann_file))[0]
        if args.wsi_list is not None and wsi_name not in args.wsi_list:
            print(f'WSI for annotation {wsi_name} was not found in provided list')
            continue
        wsi_file_exists, __ = check_wsi_exists_all_formats(wsi_name, args.wsi_path)
        if wsi_file_exists:
            keep.append(ann_file)
        else:
            print(f'WSI for annotation {wsi_name} was not found')
    ann_files = keep

    # if no annotation files found, exit
    if len(ann_files) == 0:
        sys.exit('No annotation files found. Please check the path to annotation files and the WSI list')

    ann_files = divide_list_slurm_array(ann_files)   
    # Load model
    model = create_mask2former_from_checkpoint(checkpoint_path=args.checkpoint_path, label2id=args.label2id, encoder_name=args.encoder_model, decoder_model=args.decoder_model, out_indices=args.feature_layers)
    os.makedirs(args.output_dir, exist_ok=True)

    results = evaluate_wsi_tiles(
        model=model,
        wsi_root=args.wsi_path,
        annotations_paths=ann_files,
        batch_size=args.batch_size,
        tile_size=args.tile_size,
        step_size=args.step_size,
        resolution=args.resolution,
        label2id=args.label2id,
        dataset_save_path=args.dataset_save_path,
        ignore_index=args.label2id['Background'],
    )

    # write csv with results to output_dir
    slurm_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.output_dir, f"results_{slurm_id}_finished.csv"), index=False)
    combine_results(args.output_dir)

    if slurm_id == int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1)) - 1:
        # aggregate results to get a mean and std level, then write to txt file
        summary = results_df.describe().loc[['mean', 'std']]
        # print as metric +/- std
        with open(os.path.join(args.output_dir, 'results_summary.txt'), 'w') as f:
            for col in summary.columns:
                f.write(f'{col}: {summary[col]["mean"]*100:.2f} +/- {summary[col]["std"]*100:.2f}\n')

        # Now, save png for the tiles with worst dice scores. 
        worst_tiles = results_df.sort_values(by=f'dice_{args.eval_class}', ascending=True).head(20)
        filenames = worst_tiles['filename'].values
        coords = [[x1, x2, y1, y2] for x1, x2, y1, y2 in zip(worst_tiles['x_start'].values, worst_tiles['x_end'].values, worst_tiles['y_start'].values, worst_tiles['y_end'].values)]
        metric = worst_tiles[f'dice_{args.eval_class}'].values

        save_worst_tiles(model, filenames, coords, metric, args.label2id, args.dataset_save_path, args.output_dir)     

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
    parser.add_argument('--resolution', type=float, help='Resolution', default=0.5)
    parser.add_argument('--dataset_save_path', type=str, help='Path to save the dataset', default='/rs_scratch/users/jg23p152/metassist_data/evaluate_tiles')
    parser.add_argument('--eval_class', type=str, help='Class for which metrics will be tracked', default='Metastasis')
    args = parser.parse_args()

    # process arguments
    args.label2id = {label.split(':')[0]: int(label.split(':')[1]) for label in args.label2id.split(',')}
    args.feature_layers = [int(i) for i in args.feature_layers.split(',')]
    if args.wsi_list is not None:
        assert os.path.exists(args.wsi_list), f'Provided WSI list {args.wsi_list} does not exist'
        with open(args.wsi_list, 'r') as f:
            args.wsi_list = f.read().split('\n')
            args.wsi_list = [os.path.splitext(os.path.basename(i))[0] for i in args.wsi_list if i != '']

    main(args)
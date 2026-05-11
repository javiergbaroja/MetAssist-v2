import openslide

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.dirname(SCRIPT_DIR))


from tqdm import tqdm
import argparse
from natsort import natsorted

import numpy as np
import cv2
import torch
from scipy.ndimage import binary_fill_holes 

from models.model_io import create_mask2former_from_checkpoint
from engine.inference import infer_wsi
from utils.wsi import ACCEPTED_WSI_TYPES
from utils.io import get_file_list
from utils.geometry import decode_geojson_to_mask, save_geojson_annotation, save_sparse_annotation
from utils.hpc import combine_results, divide_list_slurm_array
from utils.evaluation import get_slide_level_result


def close_metastasis(pred_mask:np.ndarray, metastasis_class:int) -> np.ndarray:
    """
    Close metastasis regions by dilating and then eroding the mask.
    """
    kernel = np.ones((5, 5), np.uint8)
    pred_mask_met = (pred_mask == metastasis_class).astype(np.uint8)
    # apply morphological closing with cv2.morphologyEx
    pred_mask_met = cv2.morphologyEx(pred_mask_met, cv2.MORPH_CLOSE, kernel)
    pred_mask[pred_mask_met == 1] = metastasis_class

    return pred_mask

def process_args(args:argparse.Namespace) -> argparse.Namespace:
    args.label2id = {label.split(':')[0]: int(label.split(':')[1]) for label in args.label2id.split(',')}
    args.feature_layers = [int(i) for i in args.feature_layers.split(',')]

    if not args.prepare_sparse and not args.prepare_qupath and not args.prepare_wsi_level_result_csv:
        raise ValueError('At least one of prepare_sparse or prepare_qupath or prepare_wsi_level_result_csv should be set to True, else no output will be generated')

    if args.wsi_list is not None:
        assert os.path.exists(args.wsi_list), f'Provided WSI list {args.wsi_list} does not exist'
        with open(args.wsi_list, 'r') as f:
            args.wsi_list = f.read().split('\n')

    if "," in args.wsi_path:
        args.wsi_path = [path.strip() for path in args.wsi_path.split(',')]
    else:
        args.wsi_path = [args.wsi_path.strip()]
    return args

def merge_mucin_and_ln(ln_seg_file_all:np.ndarray, ln_class:int, deposit_class:int, mucin_class:int) -> np.ndarray:

    ln_mask = (ln_seg_file_all == ln_class).astype(np.uint8)
    deposit_mask = (ln_seg_file_all == deposit_class).astype(np.uint8)
    ln_mask |= deposit_mask  # merge deposit into LN
    if mucin_class not in ln_seg_file_all:
        return ln_seg_file_all, binary_fill_holes(ln_mask).astype(np.uint8)
    ln_dilated = cv2.dilate(ln_mask, np.ones((5, 5), np.uint8), iterations=1)
    mucin_mask = (ln_seg_file_all == mucin_class).astype(np.uint8)
    num, mucin_labeled = cv2.connectedComponents(mucin_mask, connectivity=8)

    # Loop through each mucin object
    for mucin_label in range(1, num):
        mucin_component = (mucin_labeled == mucin_label)

        # Check if it touches LN
        if np.any(mucin_component & ln_dilated):
            ln_mask |= mucin_component  # merge whole object
        else:
            ln_seg_file_all[mucin_component] = 0
    
    return ln_seg_file_all, binary_fill_holes(ln_mask).astype(np.uint8)

def main(args):
    args = process_args(args)
    downsample_factor = 1
    # get lists of files (WSI and LN segmentation). It should check folder and subfolders using glob
    wsi_files = []
    for wsi_path in args.wsi_path:
        wsi_files.extend(get_file_list(wsi_path, ext) for ext in ACCEPTED_WSI_TYPES)
    wsi_files = natsorted([item for sublist in wsi_files for item in sublist])
    if args.wsi_list is not None:
        wsi_files = [wsi_file for wsi_file in wsi_files if os.path.basename(wsi_file) in args.wsi_list]
    ln_seg_files = get_file_list(args.ln_seg_path, 'geojson')
    assert len(wsi_files) > 0, 'No WSI files found'
    assert len(ln_seg_files) > 0, 'No LN segmentation files found'
    assert os.path.exists(args.checkpoint_path), f'Checkpoint path {args.checkpoint_path} does not exist'
    keep = []
    for wsi_file in wsi_files:
        wsi_name = os.path.splitext(os.path.basename(wsi_file))[0]
        ln_seg_file_path = [ln for ln in ln_seg_files if wsi_name in ln]
        if len(ln_seg_file_path) > 0:
            keep.append(wsi_file)
        else:
            print(f'LN segmentation file for WSI {wsi_name} does not exist')
    wsi_files = keep

    wsi_files = divide_list_slurm_array(wsi_files)   

    # Now, filter already processed files for this specific SLURM task
    if args.prepare_wsi_level_result_csv:
        slurm_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        partial_csv_path = os.path.join(args.output_dir, f"results_{slurm_id}.csv")
        finished_csv_path = os.path.join(args.output_dir, f"results_{slurm_id}_finished.csv")
        global_csv_path = os.path.join(args.output_dir, f"results.csv")
        if os.path.exists(global_csv_path):
            print(f"inference already finished for all subtasks.")
            sys.exit(0)
        elif (os.path.exists(finished_csv_path)):
            if slurm_id != (int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1)) - 1):
                print(f"Finished file {finished_csv_path} already exists, skipping this task. {slurm_id} out of {int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))}")

            else:
                # check all others all also finished
                n_finished = 0
                for i in range(int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))):
                    if os.path.exists(os.path.join(args.output_dir, f"results_{i}_finished.csv")):
                        n_finished += 1
                if n_finished == int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1)):
                    print(f"All tasks finished, combining results")
                    combine_results(args.output_dir)
                else:
                    print(f"{n_finished} out of {int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))} tasks finished, waiting for others to finish")
                    combine_results(args.output_dir)
            sys.exit(0)
            

        if os.path.exists(partial_csv_path):
            with open(partial_csv_path, 'r') as f:
                data = f.readlines()
            processed_slides = {line.split(',')[0] for line in data[1:]}  # skip header
            wsi_files = [wsi for wsi in wsi_files if os.path.basename(wsi) not in processed_slides]

    # Load model
    model = create_mask2former_from_checkpoint(checkpoint_path=args.checkpoint_path, label2id=args.label2id, encoder_name=args.encoder_model, decoder_model=args.decoder_model, out_indices=args.feature_layers)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.prepare_wsi_level_result_csv:
        slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
        partial_csv_path = os.path.join(args.output_dir, f"results_{slurm_id}.csv")
        
        if os.path.exists(partial_csv_path):
            with open(partial_csv_path, 'r') as f:
                data = f.readlines()
            processed_slides = {line.split(',')[0] for line in data[1:]}  # skip header
            wsi_files = [wsi for wsi in wsi_files if os.path.basename(wsi) not in processed_slides]
    times = []
    with tqdm(total=len(wsi_files)) as pbar:
        for wsi_file in wsi_files:
            if not args.overwrite:
                if args.prepare_sparse and not args.prepare_qupath and os.path.exists(os.path.join(args.output_dir, f'{os.path.splitext(os.path.basename(wsi_file))[0]}.npz')):
                    print(f'Sparse output for {os.path.basename(wsi_file)} already exists, skipping')
                    pbar.update(1)
                    continue
                elif args.prepare_qupath and not args.prepare_sparse and os.path.exists(os.path.join(args.output_dir, f'{os.path.splitext(os.path.basename(wsi_file))[0]}_metastasis.geojson')) and os.path.exists(os.path.join(args.output_dir, f'{os.path.splitext(os.path.basename(wsi_file))[0]}_ln.geojson')):
                    print(f'QuPath output for {os.path.basename(wsi_file)} already exists, skipping')
                    pbar.update(1)
                    continue
                elif args.prepare_sparse and args.prepare_qupath and os.path.exists(os.path.join(args.output_dir, f'{os.path.splitext(os.path.basename(wsi_file))[0]}.npz')) and os.path.exists(os.path.join(args.output_dir, f'{os.path.splitext(os.path.basename(wsi_file))[0]}_metastasis.geojson')) and os.path.exists(os.path.join(args.output_dir, f'{os.path.splitext(os.path.basename(wsi_file))[0]}_ln.geojson')):
                    print(f'Both outputs for {os.path.basename(wsi_file)} already exist, skipping')
                    pbar.update(1)
                    continue
            pbar.update(1)
            # find corresponding LN segmentation file
            wsi_name = os.path.splitext(os.path.basename(wsi_file))[0]
            ln_seg_file_path = natsorted([ln for ln in ln_seg_files if wsi_name in ln])
            ln_seg_file_path = ln_seg_file_path[0]

            # Check arguments and paths
            assert args.step_size <= args.tile_size, 'Step size should be less than tile size'
            assert args.crop_pred_edge/2 <= (args.tile_size-args.step_size), 'Crop pred edge should be less or equal to half of the overlap of two adjacent tiles'
            assert os.path.exists(wsi_file), f'WSI file {wsi_file} does not exist'
            assert os.path.exists(ln_seg_file_path), f'LN segmentation file {ln_seg_file_path} does not exist'
            try:
                ln_seg_file_all = decode_geojson_to_mask(ln_seg_file_path)
                ln_seg_file_all, ln_seg_file = merge_mucin_and_ln(ln_seg_file_all, args.ln_class, args.deposit_class, args.mucin_class)

                pred_mask, level, level_downsampling, exact_resolution, tiling_downsample_factor, read_origin, time, __ = infer_wsi(model, wsi_file, ln_seg_file, args.batch_size, args.tile_size, args.step_size, args.crop_pred_edge, args.resolution, downsample_factor)
            except Exception as e:
                print(f'Error processing {wsi_name}: {e}')
                continue
            pred_mask = close_metastasis(pred_mask, args.label2id['Metastasis'])
            if args.prepare_sparse:
                save_sparse_annotation(out_path=os.path.join(args.output_dir, f'{wsi_name}.npz'),
                                    mask=pred_mask,
                                    level=level,
                                    downsample_factor=downsample_factor,
                                    read_origin=read_origin)
            
            if args.prepare_qupath:
                save_geojson_annotation(out_path=os.path.join(args.output_dir, f'{wsi_name}_metastasis.geojson'),
                                        mask=(pred_mask == args.label2id['Metastasis']).astype(np.uint8),
                                        level=level,
                                        level_downsampling=level_downsampling*tiling_downsample_factor,
                                        category_dict={'Metastasis': 1})
                save_geojson_annotation(out_path=os.path.join(args.output_dir, f'{wsi_name}_ln.geojson'),
                                        mask=cv2.resize(ln_seg_file, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST),
                                        level=level,
                                        level_downsampling=level_downsampling*tiling_downsample_factor,
                                        category_dict={'Lymph node': 1})
                message = f"Processed {wsi_name}"
            if args.prepare_wsi_level_result_csv:
                slurm_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
                status, label, measurement = get_slide_level_result(mask=pred_mask,
                                                                    ln_seg_mask=ln_seg_file_all,
                                                                    metastasis_class=args.label2id['Metastasis'],
                                                                    ln_class=args.ln_class,
                                                                    deposit_class=args.deposit_class, 
                                                                    fat_class=args.fat_class,
                                                                    mucin_class=args.mucin_class,
                                                                    resolution=exact_resolution*tiling_downsample_factor)
                # create csv file with wsi level results
                if not os.path.exists(os.path.join(args.output_dir, f"results_{slurm_id}.csv")):
                    with open(os.path.join(args.output_dir, f"results_{slurm_id}.csv"), 'w') as f:
                        f.write('B-Number,folder,status,length,outcome\n')
            
                with open(os.path.join(args.output_dir, f"results_{slurm_id}.csv"), 'a') as f:
                    f.write(f'{os.path.basename(wsi_file)},{os.path.basename(os.path.dirname(wsi_file))},{status},{measurement},{label}\n')
                message += f': {measurement:.2f} um - {status} - {label}'
            times.append(time)
            print(message)
            torch.cuda.empty_cache()
        
    print(f'Inference for {len(times)} WSIs took {sum(times):.2f} s, average time per WSI: {np.mean(times):.2f} \u00B1 {np.std(times):.2f} s')
    if args.prepare_wsi_level_result_csv:
        os.rename(os.path.join(args.output_dir, f"results_{slurm_id}.csv"), os.path.join(args.output_dir, f"results_{slurm_id}_finished.csv"))
        combine_results(args.output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Infer mask2former on WSI')
    parser.add_argument('--wsi_path', type=str, help='Path to input WSI folder')
    parser.add_argument('--ln_seg_path', type=str, help='Path to LN segmentation folder')
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
    parser.add_argument('--ln_class', type=int, help='LN class', default=1)
    parser.add_argument('--deposit_class', type=int, help='Deposit class', default=2)
    parser.add_argument('--fat_class', type=int, help='Fat class', default=4)
    parser.add_argument('--mucin_class', type=int, help='Mucin class', default=6)
    parser.add_argument('--wsi_list', type=str, help='Path to WSI list, a subset of WSIs to process, given as slidea.mrxs\nslideb.mrxs\nslidec.mrxs', default=None)
    parser.add_argument('--resolution', type=float, help='Resolution', default=0.5)
    parser.add_argument('--prepare_qupath', action='store_true', help='Prepare QuPath compatible output')
    parser.add_argument('--prepare_sparse', action='store_true', help='Prepare sparse output')
    parser.add_argument('--prepare_wsi_level_result_csv', action='store_true', help='Prepare WSI level')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing outputs')
    args = parser.parse_args()

    main(args)
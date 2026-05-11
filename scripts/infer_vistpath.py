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
from utils.wsi import ACCEPTED_WSI_TYPES, detect_tissue_mask, prepare_read_from_slide
from utils.io import get_file_list 
from utils.geometry import decode_geojson_to_mask, save_geojson_annotation
from utils.hpc import divide_list_slurm_array

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

    if args.wsi_list is not None:
        assert os.path.exists(args.wsi_list), f'Provided WSI list {args.wsi_list} does not exist'
        with open(args.wsi_list, 'r') as f:
            args.wsi_list = f.read().split('\n')

    if "," in args.wsi_path:
        args.wsi_path = [path.strip() for path in args.wsi_path.split(',')]
    else:
        args.wsi_path = [args.wsi_path.strip()]
    return args

def merge_mucin_and_ln(ln_seg_file_all:np.ndarray, ln_class:int, mucin_class:int) -> np.ndarray:

    ln_mask = (ln_seg_file_all == ln_class).astype(np.uint8)
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
    tissue_mask_files = get_file_list(args.tissue_mask_path, 'geojson')
    assert len(wsi_files) > 0, 'No WSI files found'
    assert os.path.exists(args.checkpoint_path), f'Checkpoint path {args.checkpoint_path} does not exist'
    
    wsi_files = divide_list_slurm_array(wsi_files)   

    # Load model
    model = create_mask2former_from_checkpoint(checkpoint_path=args.checkpoint_path, label2id=args.label2id, encoder_name='vistapath', decoder_model='vistapath')
    os.makedirs(args.output_dir, exist_ok=True)

    times = []
    with tqdm(total=len(wsi_files)) as pbar:
        for wsi_file in wsi_files:
            pbar.update(1)
            # find corresponding LN segmentation file
            wsi_name = os.path.splitext(os.path.basename(wsi_file))[0]
            if len(tissue_mask_files) == 0:
                tissue_mask_file_path = ""
            else:
                tissue_mask_file_paths = [tm for tm in tissue_mask_files if wsi_name in tm]
                if len(tissue_mask_file_paths) == 0:
                    tissue_mask_file_path = ""
                else:
                    tissue_mask_file_path = tissue_mask_file_paths[0]
            
            if not args.overwrite and os.path.exists(os.path.join(args.output_dir, f'{wsi_name}.geojson')):
                print(f'Skipping {wsi_name} as output already exists')
                continue

            # Check arguments and paths
            assert args.step_size <= args.tile_size, 'Step size should be less than tile size'
            assert args.crop_pred_edge/2 <= (args.tile_size-args.step_size), 'Crop pred edge should be less or equal to half of the overlap of two adjacent tiles'
            assert os.path.exists(wsi_file), f'WSI file {wsi_file} does not exist'
            # assert os.path.exists(tissue_mask_file_path), f'Tissue mask file {tissue_mask_file_path} does not exist'
            if os.path.exists(tissue_mask_file_path):
                tissue_mask_file_all = decode_geojson_to_mask(tissue_mask_file_path)
                tissue_mask_file_all, tissue_mask = merge_mucin_and_ln(tissue_mask_file_all, args.ln_class, args.mucin_class)

                # pred_mask, level, level_downsampling, exact_resolution, tiling_downsample_factor, read_origin, time, __ = infer_wsi(model, wsi_file, tissue_mask, args.batch_size, args.tile_size, args.step_size, args.crop_pred_edge, args.resolution, downsample_factor, normalize_input=False)
            else:
                level, level_downsampling, exact_resolution, tiling_downsample_factor, original_dim, read_origin = prepare_read_from_slide(wsi_file, resolution=args.resolution, file_type=os.path.splitext(wsi_file)[1].lower())
                tissue_mask, __ = detect_tissue_mask(wsi_file)
            
            try:
                pred_mask, level, level_downsampling, exact_resolution, tiling_downsample_factor, read_origin, time, __ = infer_wsi(model, wsi_file, tissue_mask, args.batch_size, args.tile_size, args.step_size, args.crop_pred_edge, args.resolution, downsample_factor, normalize_input=False)
            except Exception as e:
                print(f'Error processing {wsi_name}: {e}')
                continue
            
            save_geojson_annotation(out_path=os.path.join(args.output_dir, f'{wsi_name}.geojson'),
                            mask=pred_mask,
                            level=level,
                            level_downsampling=level_downsampling*tiling_downsample_factor,
                            category_dict={k: v+1 for k, v in args.label2id.items()})
            message = f"Processed {wsi_name}"
            
            times.append(time)
            print(message)
            torch.cuda.empty_cache()
        
    print(f'Inference for {len(times)} WSIs took {sum(times):.2f} s, average time per WSI: {np.mean(times):.2f} \u00B1 {np.std(times):.2f} s')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Infer mask2former on WSI')
    parser.add_argument('--wsi_path', type=str, help='Path to input WSI folder')
    parser.add_argument('--tissue_mask_path', type=str, help='Path to LN segmentation folder')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--checkpoint_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--label2id', type=str, help='Dictionary mapping label to id')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=1)
    parser.add_argument('--tile_size', type=int, help='Tile size', default=224)
    parser.add_argument('--step_size', type=int, help='Step size', default=180)
    parser.add_argument('--crop_pred_edge', type=int, help='Step size', default=50)
    parser.add_argument('--ln_class', type=int, help='LN class', default=1)
    parser.add_argument('--mucin_class', type=int, help='Mucin class', default=6)
    parser.add_argument('--wsi_list', type=str, help='Path to WSI list, a subset of WSIs to process, given as slidea.mrxs\nslideb.mrxs\nslidec.mrxs', default=None)
    parser.add_argument('--resolution', type=float, help='Resolution', default=0.5)
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing outputs')
    args = parser.parse_args()

    main(args)
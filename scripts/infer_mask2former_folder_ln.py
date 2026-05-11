import openslide

import os
import sys
from tqdm import tqdm
import argparse
from natsort import natsorted

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.dirname(SCRIPT_DIR))

import numpy as np
import cv2
import torch

from models.model_io import create_mask2former_from_checkpoint
from engine.inference import infer_wsi
from utils.hpc import divide_list_slurm_array
from utils.postprocessing import post_process
from utils.geometry import save_geojson_annotation, save_npy_mask
from utils.wsi import detect_colors, ACCEPTED_WSI_TYPES
from utils.visualization import save_overlay
from utils.io import get_file_list

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
    assert len(wsi_files) > 0, 'No WSI files found'
    assert os.path.exists(args.checkpoint_path), f'Checkpoint path {args.checkpoint_path} does not exist'

    if not args.overwrite:
        # check if output files already exist, if yes, remove them from the list of files to process
        existing_files = []
        for wsi_file in wsi_files:
            wsi_name = os.path.splitext(os.path.basename(wsi_file))[0]
            if args.prepare_overlay:
                overlay_path = os.path.join(args.output_dir, f'{wsi_name}_overlay.png')
                if os.path.exists(overlay_path):
                    existing_files.append(wsi_file)
            if args.prepare_qupath:
                qupath_path = os.path.join(args.output_dir, f'{wsi_name}.geojson')
                if os.path.exists(qupath_path):
                    existing_files.append(wsi_file)
            if args.prepare_pred_mask:
                pred_mask_path = os.path.join(args.output_dir, f'{wsi_name}.npy')
                if os.path.exists(pred_mask_path):
                    existing_files.append(wsi_file)
        existing_files = set(existing_files)
        wsi_files = [wsi_file for wsi_file in wsi_files if wsi_file not in existing_files]
        print(f'{len(existing_files)} files already exist and will be skipped. Use --overwrite flag to overwrite existing files.')

    # divide list of files to process in parallel
    wsi_files = divide_list_slurm_array(wsi_files)

    # Load model
    model = create_mask2former_from_checkpoint(checkpoint_path=args.checkpoint_path, label2id=args.label2id, encoder_name=args.encoder_model, decoder_model=args.decoder_model, out_indices=args.feature_layers)
    os.makedirs(args.output_dir, exist_ok=True)
    times = []
    with tqdm(total=len(wsi_files)) as pbar:
        for wsi_file in wsi_files:
            pbar.update(1)
            # find corresponding LN segmentation file
            wsi_name = os.path.splitext(os.path.basename(wsi_file))[0]

            # Check arguments and paths
            assert args.step_size <= args.tile_size, 'Step size should be less than tile size'
            assert args.crop_pred_edge <= (args.tile_size-args.step_size), 'Crop pred edge should be less or equal to half of the overlap of two adjacent tiles'
            assert os.path.exists(wsi_file), f'WSI file {wsi_file} does not exist'
            # save as numpy
            filter_mask = np.ones((5,5), dtype=np.uint8)
            
            try:
                pred_mask, level, level_downsampling, exact_resolution, tiling_downsample_factor, read_origin, time, __ = infer_wsi(model, wsi_file, filter_mask, args.batch_size, args.tile_size, args.step_size, args.crop_pred_edge, args.resolution, downsample_factor)
            except Exception as e:
                print(f'Error processing {wsi_name}: {e}')
                continue
            mucin_mask = (pred_mask == args.label2id['Mucin']).astype(np.uint8)
            if args.apply_post_processing:
                min_ln_area = (300 / (exact_resolution*tiling_downsample_factor)) ** 2 * np.pi
                pred_mask = post_process(segmentation_mask=pred_mask, 
                                 lymph_node_class=args.label2id['Lymph node'], 
                                 classes_to_merge=[args.label2id['Primary tumor'], args.label2id['Mucin']], 
                                 merge_thresholds=[0.95, 0.05], 
                                 erase_thresholds=[0.01, 0.01], 
                                 apply_opening=[True, False],
                                 min_ln_area=int(min_ln_area),
                                 complexity_threshold=args.complexity_threshold) 
                
                # # remove LNs detected in noise
                ln_mask = (pred_mask == args.label2id['Lymph node']).astype(np.uint8)
                num_labels, labeled_lns = cv2.connectedComponents(ln_mask)
                for i in range(1, num_labels):
                    bbox = cv2.boundingRect((labeled_lns == i).astype(np.uint8))
                    # read region of interest from the original WSI
                    crop = np.array(openslide.open_slide(wsi_file).read_region((read_origin[0]+bbox[0]*level_downsampling, read_origin[1]+bbox[1]*level_downsampling), level, (bbox[2]*tiling_downsample_factor, bbox[3]*tiling_downsample_factor)))
                    crop[crop[:, :, 3] == 0] = 255
                    crop = cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)
                    if tiling_downsample_factor > 1:
                        crop = cv2.resize(crop, (crop.shape[1] // tiling_downsample_factor, crop.shape[0] // tiling_downsample_factor), interpolation=cv2.INTER_LINEAR)
                    has_colors = detect_colors(crop[ln_mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] > 0], 0.025)
                    if not has_colors:
                        pred_mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = args.label2id['Background']


            if args.prepare_overlay:
                save_overlay(out_path=os.path.join(args.output_dir, f'{wsi_name}_overlay.png'),
                             wsi_path=wsi_file,
                             mask=pred_mask,
                             level=level,
                             read_origin=read_origin,
                             level_downsampling=level_downsampling,
                             downsizing_factor=tiling_downsample_factor,
                             label2id=args.label2id)
            
            if args.prepare_qupath:
                mucin_mask *= (pred_mask == args.label2id['Lymph node']).astype(np.uint8)
                save_geojson_annotation(out_path=os.path.join(args.output_dir, f'{wsi_name}.geojson'),
                                        mask=pred_mask,
                                        overlapping_class_mask=mucin_mask,
                                        overlapping_class_name='Mucin',
                                        level=level,
                                        level_downsampling=level_downsampling*tiling_downsample_factor,
                                        category_dict=args.label2id)
            
            if args.prepare_pred_mask:
                save_npy_mask(os.path.join(args.output_dir, f'{wsi_name}.npy'), pred_mask)
    
            print(f'Processed {wsi_name} in {time:.2f} seconds')
            torch.cuda.empty_cache()
            times.append(time)
    print(f'Inference for {len(times)} WSIs took {sum(times):.2f} s, average time per WSI: {np.mean(times):.2f} \u00B1 {np.std(times):.2f} s')



def process_args(args):
    # process arguments
    args.label2id = {label.split(':')[0]: int(label.split(':')[1]) for label in args.label2id.split(',')}
    args.feature_layers = [int(i) for i in args.feature_layers.split(',')]

    if not args.prepare_overlay and not args.prepare_qupath and not args.prepare_pred_mask:
        raise ValueError('At least one of prepare_overlay or prepare_qupath or prepare_pred_mask should be set to True, else no output will be generated')
    
    if "," in args.wsi_path:
        args.wsi_path = [path.strip() for path in args.wsi_path.split(',')]
    else:
        args.wsi_path = [args.wsi_path.strip()]

    if args.wsi_list is not None:
        assert os.path.exists(args.wsi_list), f'Provided WSI list {args.wsi_list} does not exist'
        with open(args.wsi_list, 'r') as f:
            args.wsi_list = f.read().split('\n')
        print(f'Processing {len(args.wsi_list)} WSI files from list {args.wsi_list}')
    return args
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Infer mask2former on WSI')
    parser.add_argument('--wsi_path', type=str, help='Path to input WSI folder')
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
    parser.add_argument('--wsi_list', type=str, help='Path to WSI list, a subset of WSIs to process, given as slidea.mrxs\nslideb.mrxs\nslidec.mrxs', default=None)
    parser.add_argument('--apply_post_processing', action='store_true', help='Prepare QuPath compatible output')
    parser.add_argument('--complexity_threshold', type=float, help='Complexity threshold for post-processing', default=2.9)
    parser.add_argument('--prepare_qupath', action='store_true', help='Prepare QuPath compatible output')
    parser.add_argument('--prepare_overlay', action='store_true', help='Prepare sparse output')
    parser.add_argument('--prepare_pred_mask', action='store_true', help='Prepare predicted mask')
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing output files')
    args = parser.parse_args()

    

    main(args)
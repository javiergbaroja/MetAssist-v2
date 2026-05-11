import openslide

import os
import sys
import argparse

import numpy as np
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.dirname(SCRIPT_DIR))

from models.model_io import create_mask2former_from_checkpoint
from engine.inference import infer_wsi
from utils.geometry import decode_geojson_to_mask, save_geojson_annotation, save_sparse_annotation
from utils.evaluation import get_slide_level_result

def main(args):
    downsample_factor = 1

    # Check arguments and paths
    assert args.step_size <= args.tile_size, 'Step size should be less than tile size'
    assert args.crop_pred_edge/2 <= (args.tile_size-args.step_size), 'Crop pred edge should be less or equal to half of the overlap of two adjacent tiles'
    assert os.path.exists(args.wsi_path), f'WSI path {args.wsi_path} does not exist'
    if isinstance(args.checkpoint_path, str):
        assert os.path.exists(args.checkpoint_path), f'Model checkpoint path {args.checkpoint_path} does not exist'
    else:
        assert hasattr(args.checkpoint_path, 'forward'), 'Model checkpoint should be a path to a model checkpoint or a pytorch model'

    assert os.path.exists(args.ln_seg_path), f'LN segmentation path {args.ln_seg_path} does not exist'
    wsi_name = os.path.splitext(os.path.basename(args.wsi_path))[0]
    ln_seg_file = os.path.join(args.ln_seg_path, f'{wsi_name}.geojson')
    assert os.path.exists(ln_seg_file), f'LN segmentation file {ln_seg_file} does not exist'
    # ln_seg_file = (np.load(ln_seg_file) == args.ln_class).astype(np.uint8)  
    ln_seg_file = (decode_geojson_to_mask(ln_seg_file) == args.ln_class).astype(np.uint8)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = create_mask2former_from_checkpoint(checkpoint_path=args.checkpoint_path, label2id=args.label2id, encoder_name=args.encoder_model, decoder_model=args.decoder_model, out_indices=args.feature_layers)
    pred_mask, level, level_downsampling, read_origin,__,__ = infer_wsi(model, args.wsi_path, ln_seg_file, args.batch_size, args.tile_size, args.step_size, args.crop_pred_edge, args.resolution, downsample_factor)

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
                                level_downsampling=level_downsampling,
                                category_dict={'Metastasis': 1})
        save_geojson_annotation(out_path=os.path.join(args.output_dir, f'{wsi_name}_ln.geojson'),
                                mask=cv2.resize(ln_seg_file, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST),
                                level=level,
                                level_downsampling=level_downsampling,
                                category_dict={'Lymph node': 1})
        
    if args.prepare_wsi_level_result_csv:
        status, label = get_slide_level_result((pred_mask == args.label2id['Metastasis']).astype(np.uint8), args.resolution)
        if not os.path.exists(os.path.join(args.output_dir, f"results.csv")):
            with open(os.path.join(args.output_dir, f"results.csv"), 'w') as f:
                f.write('B-Number,folder,status,outcome\n')
        with open(os.path.join(args.output_dir, f"results.csv"), 'a') as f:
                    f.write(f'{os.path.basename(args.wsi_path)},{os.path.basename(os.path.dirname(args.wsi_path))},{status},{label}\n')

        print(f'WSI {wsi_name}: {status} - {label}')
    print(f'Processed {wsi_name}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Infer mask2former on WSI')
    parser.add_argument('--wsi_path', type=str, help='Path to input WSI')
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
    parser.add_argument('--resolution', type=float, help='Resolution', default=0.5)
    parser.add_argument('--prepare_qupath', action='store_true', help='Prepare QuPath compatible output')
    parser.add_argument('--prepare_sparse', action='store_true', help='Prepare sparse output')
    parser.add_argument('--prepare_wsi_level_result_csv', action='store_true', help='Prepare WSI level')
    args = parser.parse_args()

    # process arguments
    args.label2id = {label.split(':')[0]: int(label.split(':')[1]) for label in args.label2id.split(',')}
    args.feature_layers = [int(i) for i in args.feature_layers.split(',')]

    if not args.prepare_sparse and not args.prepare_qupath and not args.prepare_wsi_level_result_csv:
        raise ValueError('At least one of prepare_sparse or prepare_qupath or prepare_wsi_level_result_csv should be set to True, else no output will be generated')

    main(args)
import openslide

import os
import sys
import random
import argparse

import numpy as np
import torch
from sklearn.utils import check_random_state

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.dirname(SCRIPT_DIR))

from engine.trainer_mask2former import TrainerMask2Former

# set all seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
check_random_state(42)
random.seed(42)
torch.backends.cudnn.deterministic = True

def preprocess_args(args):
    args.wsi_path = [wsi_path.strip() for wsi_path in args.wsi_path.split(',')]
    args.path_to_ann = [path_to_ann.strip() for path_to_ann in args.path_to_ann.split(',')]
    args.label2id = {label.split(':')[0]: int(label.split(':')[1]) for label in args.label2id.split(',')}
    args.id2label = {v: k for k, v in args.label2id.items()}
    args.num_classes = len(args.label2id)
    args.feature_layers = [int(i) for i in args.feature_layers.split(',')]
    args.data_augs = {aug.split(':')[0]:aug.split(':')[1] for aug in args.data_augs.split(',')}
    # convert values to appropriate types, go key by key
    for key in args.data_augs.keys():
        if key in ['rotation', 'flip']:
            # convert to list of integers for rotate and of strings for flip
            v = [eval(v) for v in args.data_augs[key].split('-')] if key == 'rotation' else args.data_augs[key].split('-')
            args.data_augs[key] = v
        elif key in ['brightness', 'contrast', 'blur', 'noise']:
            args.data_augs[key] = eval(args.data_augs[key])
        elif key == 'color':
            args.data_augs[key] = args.data_augs[key]

    if args.early_stopping is None:
        args.early_stopping = args.num_epochs

    with open(args.valid_set, 'r') as f:
        assert os.path.isfile(args.valid_set), f'File {args.valid_set} does not exist'
        args.valid_set_file = args.valid_set
        args.valid_set = f.read().split('\n')

    if args.train_set is not None:
        with open(args.train_set, 'r') as f:
            assert os.path.isfile(args.train_set), f'File {args.train_set} does not exist'
            args.train_set_file = args.train_set
            args.train_set = f.read().split('\n')

    assert args.eval_class in args.label2id.keys(), f'Class {args.eval_class} not found in label2id. Available classes: {args.label2id.keys()}'

    return args

def main(args):
    args = preprocess_args(args)

    for fold in range(args.start_fold, args.n_folds):
        args.fold = fold+1

        trainer = TrainerMask2Former(args)
        if not trainer.resume_training:
            continue
        trainer.train()
        trainer.accelerator.print('Training completed. Running predictions on validation WSIs...')
        
        trainer.infer_on_validation_set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument('--path_to_ann', 
                          type=str, 
                          required=True, 
                          help='Path to geojson annotations'
                          )
    
    required.add_argument('--wsi_path',
                            type=str,
                            required=True,
                            help='Path to WSIs. Can be a list separated by commas or a folder with WSIs'
                            )
    required.add_argument('--dataset',
                            type=str,
                            required=True,
                            help="Dataset name (from package/datasets), such as 'dataset_seg_metastasis'"
                            )
    
    required.add_argument('--segmentation_task',
                            type=str,
                            required=True,
                            help='Segmentation task: "ln" or "metastasis"'
                            )
    
    
    required.add_argument('--output_path',
                            type=str,
                            required=True,
                            help='Path to save results'
                            )
    
    required.add_argument('--dataset_save_path',
                            type=str,
                            required=True,
                            help='Path to save dataset'
                            )
      
    required.add_argument('--eval_class',
                            type=str,
                            required=True,
                            help='Class for which metrics will be tracked')  
    
    
    required.add_argument('--data_augs',
                            type=str,
                            required=True,
                            help="Data augmentations. Example: 'rotation:0-90,flip:v-h,contrast:True,blur:True,noise:True,color:hsv'"
                            )
    
    required.add_argument('--valid_set',
                            type=str,
                            required=True,
                            help="Path to TXT file with a list of WSI names. Format of file should be 'wsi_name_1.mrxs\nwsi_name_2.mrxs...'",
                            )
    
    optional.add_argument('--train_set',
                            type=str,
                            default=None,
                            help="Path to TXT file with a list of WSI names. Format of file should be 'wsi_name_1.mrxs\nwsi_name_2.mrxs...'",
                            )
    
    optional.add_argument('--ignore_class',
                            type=int,
                            help='Index of class to ignore in loss calculation',
                            default=255,
                            )

    optional.add_argument('--batch_size',
                            type=int,
                            default=128,
                            help='Batch size'
                            )   
    
    optional.add_argument('--num_epochs',
                            type=int,
                            default=10,
                            help='Number of epochs'
                            )
    
    optional.add_argument('--train_by_fixed_batches',
                            type=bool,
                            default=False,
                            help='Train by fixed number of batches'
                            )

    optional.add_argument('--valid_by_fixed_batches',
                            type=bool,
                            default=False,
                            help='Validate by fixed number of batches'
                            )
    
    optional.add_argument('--num_batches_per_epoch',
                            type=int,
                            default=70,
                            help='Number of batches per epoch'
                            )
    
    optional.add_argument('--resolution',
                            type=float,
                            default=4.0,
                            help='Resolution in microns per pixel (mpp). If not available, the resolution corresponding to the closest WSI level will be used'
                            )
    
    optional.add_argument('--tile_size',
                            type=int,
                            default=384,
                            help='Tile size'
                            )
    
    optional.add_argument('--tile_size_inference',
                            type=int,
                            default=384,
                            help='Tile size at inference'
                            )
    
    optional.add_argument('--step_size',
                            type=int,
                            default=380,
                            help='Step size'
                            )

    optional.add_argument('--step_size_inference',
                            type=int,
                            default=380,
                            help='Step size at inference'
                            )
    optional.add_argument('--crop_pred_edge',
                            type=int,
                            default=50,
                            help='Crop prediction tile edge'
                            )
    
    optional.add_argument('--learning_rate',
                            type=float,
                            default=5e-5,
                            help='Learning rate'
                            )
    
    optional.add_argument('--n_folds',
                            type=int,
                            default=5,
                            help='Number of folds'
                            )
    
    optional.add_argument('--label2id',
                            type=str,
                            default='Background:0,Lymph node:1,Tumor deposits:2,Primary tumor:3,Primary tissue:4,Ink:5,Vessels:6,Metastasis:7,Necrosis:8,Connective tissue:9,Folds:10',
                            help='Label2id'
                            )   
    
    optional.add_argument('--encoder_model',
                            type=str,
                            default='dinov2_l',
                            help='Encoder name'
                            )
    
    optional.add_argument('--feature_layers',
                            type=str,
                            default='6,8,10,12',
                            help='extract features from the speficied layer indies in the encoder'
                            )
    
    optional.add_argument('--decoder_model',
                            type=str,
                            default='swin-large-cityscapes-semantic',
                            help='Decoder name'
                            )
    
    optional.add_argument('--finetuning_path',
                            type=str,
                            default=None,
                            help='Path to pretrained model, from which to continue training'
                            )
    
    optional.add_argument('--start_fold',
                            type=int,
                            default=0,
                            help='Fold to start training from. Starts from 0'
                            )
    
    optional.add_argument('--early_stopping',
                            type=int,
                            default=None,
                            help='Number of epochs to wait for early stopping'
                            )
    
    optional.add_argument('--save_thumbnail',
                            action='store_true',
                            help='Save thumbnail of the WSI at inference'
                            )
    
    optional.add_argument('--filter_mask_path',
                            type=str,
                            help='Folder with filter masks for each WSI',
                            default=None
                            )
    
    required.add_argument('--filter_mask_class',
                            type=int,
                            help='Index of class to keep from filter mask',
                            default=1
                            )
    
    args = parser.parse_args()
    main(args)
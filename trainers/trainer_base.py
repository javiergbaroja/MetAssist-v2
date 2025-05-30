import torch
from accelerate import Accelerator
import os
import gc
import json
import importlib
import argparse
from utils.utils import function_wo_output

class TrainerBase:
    def __init__(self, args):
        self.accelerator = Accelerator(gradient_accumulation_steps=1, device_placement=True)
        self.accelerator.print(f"Launching training with {self.accelerator.num_processes} GPUs")
        if torch.cuda.is_available():
            self.accelerator.print("CUDA is available. Name of GPU: ", torch.cuda.get_device_name(0))

        # print Namespace
        self.accelerator.print(args)

        # model related
        self.model = None
        self.optimizer = None
        self.scheduler = None

        # paths
        self.checkpoint_path = ''
        self.model_save_path = ''
        self.dataset_save_path = ''
        self.result_save_path = ''
        self.filter_mask_path = ''

        # data
        self.label2id = dict()
        self.tile_size = 0
        self.tile_size_inference = 0
        self.step_size = 0
        self.step_size_inference = 0
        self.resolution = 0.0
        self.dataset_train = None
        self.dataset_valid = None
        self.preprocessor = None

        # training and results
        self.start_epoch, self.current_epoch, self.waiting_epochs = 0, 0, 0
        self.train_results, self.valid_results = [], []
        self.train_by_fixed_batches = False
        self.valid_by_fixed_batches = False
        self.num_batches_per_epoch = 0
        self.valid_min_loss = float('inf')
        self.num_epochs = 0
        self.lr_start = 0.0
        self.early_stopping = 0
        self.batch_size = 0

        self.TileDataset, self.SlideDataset = self.get_dataset_classes(args.dataset)
        

    def initialize_paths(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.dataset_save_path, exist_ok=True)
        os.makedirs(self.result_save_path, exist_ok=True)
        self.accelerator.print(f"Results will be saved in {self.result_save_path}")
        self.accelerator.print(f"Checkpoints will be saved in {self.checkpoint_path}")

    def train(self):
        raise NotImplementedError
    
    def save_loss_as_dict(self):
        loss_dict = {
            'train_loss': self.train_results,
            'valid_loss': self.valid_results
        }
        # save as json        
        with open(os.path.join(self.result_save_path, 'loss.json'), 'w') as f:
            json.dump(loss_dict, f)


    def save_model(self):
        torch.save({
                    'model_state_dict': self.accelerator.get_state_dict(self.model, unwrap=True),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': self.train_results, 
                    'valid_loss': self.valid_results,
                    'epoch': self.current_epoch + 1
                    }, self.model_save_path)
    
    def create_model(self):
        raise NotImplementedError
    
    def create_dataloaders(self):
        raise NotImplementedError

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

    def reset_training(self):
        self.accelerator.wait_for_everyone()
        self.model = self.accelerator.free_memory(self.model)
        self.accelerator = Accelerator(gradient_accumulation_steps=1, device_placement=True)
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        return self.accelerator
    
    def get_dataset_classes(self, dataset):
        module = importlib.import_module(f'datasets.{dataset}')
        TileDataset = getattr(module, 'TileDataset')
        SlideDataset = getattr(module, 'SlideDataset')
        return TileDataset, SlideDataset
    
    def set_inference_fnc(self, args):
        module = importlib.import_module(f'scripts.infer_mask2former_wsi_{args.segmentation_task}')
        self.inference_fnc = getattr(module, 'main')
        module = importlib.import_module(f'scripts.evaluate_metrics_mask2former_{args.segmentation_task}')
        self.evaluate_fnc = getattr(module, 'main')
        self.args_inference = argparse.Namespace(
                wsi_path=args.wsi_path,
                path_to_ann = args.path_to_ann,
                wsi_list=args.valid_set,
                checkpoint_path=self.model_save_path,
                encoder_model=args.encoder_model,
                decoder_model=args.decoder_model,
                label2id=self.label2id,
                feature_layers=args.feature_layers,
                batch_size=args.batch_size,
                tile_size=args.tile_size_inference,
                step_size=args.step_size_inference,
                crop_pred_edge=args.crop_pred_edge,
                resolution=self.resolution,
                output_dir=self.result_save_path,
                prepare_qupath=True,
                eval_class=args.eval_class,
            )

        # adjust arguments for the inference script
        if args.segmentation_task == 'metastasis':
            self.args_inference.ln_seg_path = args.filter_mask_path
            self.args_inference.ln_class = args.filter_mask_class
            self.args_inference.prepare_sparse = False
            self.args_inference.prepare_wsi_level_result_csv = True
            self.args_inference.prepare_pred_mask = False
            self.args_inference.dataset_save_path = self.dataset_save_path

        elif args.segmentation_task == 'ln':
            self.args_inference.prepare_overlay = True
            self.args_inference.prepare_pred_mask = False
            self.args_inference.apply_post_processing = True
            self.inference_fnc = function_wo_output

        else:
            raise ValueError(f"Unsupported segmentation task: {args.segmentation_task}")
        
    def save_experiment_config(self, args:argparse.Namespace):
        config_path = os.path.join(self.result_save_path, 'config')
        code_path = os.path.join(config_path, 'code')
        os.makedirs(code_path, exist_ok=True)

        with open(os.path.join(config_path, 'args.txt'), 'w') as f:
            for key, value in vars(args).items():
                f.write(f"{key}: {value}\n")
        
        source_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for root, dirs, files in os.walk(source_path):
            for file in files:
                if file.endswith('.py'):
                    src_file = os.path.join(root, file)
                    rel_path = os.path.relpath(src_file, source_path)
                    dst_file = os.path.join(code_path, rel_path)
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    with open(src_file, 'r') as src:
                        with open(dst_file, 'w') as dst:
                            dst.write(src.read())

        self.accelerator.print(f"Experiment configuration saved to {config_path}")


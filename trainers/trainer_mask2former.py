import openslide

import os
import sys
import numpy as np
from glob import glob
from typing import Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.mask2former import TrainCollator, create_model, custom_post_process_semantic_segmentation
from utils.metrics import get_iou_multiclass
from utils.models import create_mask2former_from_checkpoint
from trainers.trainer_base import TrainerBase



class TrainerMask2Former(TrainerBase):
    def __init__(self, args):
        super().__init__(args)

        self.model = create_model(encoder_model=args.encoder_model, 
                                  decoder_model=args.decoder_model, 
                                  label2id=args.label2id, 
                                  id2label=args.id2label, 
                                  out_indices=args.feature_layers,
                                  freeze_encoder=True)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=round(args.num_epochs*0.2), T_mult=1, eta_min=1e-7, last_epoch=-1)
        
        model_name = args.encoder_model+'_'+args.decoder_model+'_res_'+str(args.resolution)+'_tile_size_'+str(args.tile_size)+'_step_size_'+str(args.step_size)
        result_path_parent = os.path.join(args.output_path, model_name)
        self.checkpoint_path = os.path.join(result_path_parent, 'checkpoints')
        self.model_save_path = os.path.join(self.checkpoint_path, f'{model_name}_fold_{args.fold}.pt')
        self.dataset_save_path = os.path.join(args.dataset_save_path, "/".join(os.path.normpath(result_path_parent).split(os.path.sep)[-2:]))
        self.result_save_path = os.path.join(result_path_parent, f'fold_{args.fold}')
        self.initialize_paths()

        # training params
        self.num_epochs = args.num_epochs
        self.early_stopping = args.early_stopping
        self.train_by_fixed_batches = args.train_by_fixed_batches
        self.valid_by_fixed_batches = args.valid_by_fixed_batches
        self.num_batches_per_epoch = args.num_batches_per_epoch
        self.fold = args.fold
        self.ignored_index = args.ignore_class
        self.filter_mask_path = args.filter_mask_path
        
        # load model if exists
        self.resume_training = self.load_model(args.finetuning_path)
        if not self.resume_training:
            return
        
        # data. order of categories in label2id is not sorted. 
        # The order is equal to the order in which classes are added to the GT masks. For example:
        self.label2id:dict = args.label2id
        self.id2label = {v:k for k,v in self.label2id.items()}
        self.categories_to_eval = sorted([i for i in self.label2id.values() if i != self.ignored_index])
        self.tile_size = args.tile_size
        self.step_size = args.step_size
        self.resolution = args.resolution
        self.train_wsi = args.train_set
        self.valid_wsi = args.valid_set
        self.dataloader_train, self.dataloader_valid = self.create_dataloaders(args.path_to_ann, args.wsi_path, args.data_augs, args.batch_size)

        # inference
        self.set_inference_fnc(args)

        # save experiment config
        self.save_experiment_config(args)

    def load_model(self, finetuning_path:str) -> bool:
        resume_training = True
        if os.path.exists(self.model_save_path):
            checkpoint = torch.load(self.model_save_path, map_location='cpu')
            self.start_epoch = len(checkpoint['train_loss']) 
            # continue to next fold if start_epoch is equal to number of epochs
            if self.start_epoch >= self.num_epochs:
                # check if inference was performed on this fold (if result_vist_path exists is empty)
                if len(os.listdir(self.result_save_path)) <=1:
                    self.accelerator.print(f"Model in fold {self.fold} already trained for {self.num_epochs} epochs. Skipping training of fold {self.fold}, but performing inference")
                else:
                    self.accelerator.print(f"Model in fold {self.fold} already trained for {self.num_epochs} epochs and evaluated. Skipping fold {self.fold}")
                    resume_training = False
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.accelerator.print(f'Model device: {next(self.model.parameters()).device}')
                optimizer = optim.AdamW(self.model.parameters(), lr=self.optimizer.defaults['lr'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6)
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.train_results, self.valid_results = checkpoint['train_loss'], checkpoint['valid_loss']
                self.valid_min_loss = checkpoint['valid_loss'][-1]
                self.accelerator.print(f"Model loaded from {self.model_save_path} at epoch {self.start_epoch}")
                del checkpoint

        elif finetuning_path is not None:
            checkpoint = torch.load(finetuning_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.accelerator.print(f"Model loaded from {finetuning_path} for finetuning")

        return resume_training
        

    def create_dataloaders(self, mask_source:list, wsi_root:str, data_augs:str, batch_size:int) -> Tuple[DataLoader, DataLoader]:
        
        all_masks = [glob(path) for path in mask_source]
        all_masks = sorted([mask for sublist in all_masks for mask in sublist])
        valid_names = [os.path.splitext(wsi)[0] for wsi in self.valid_wsi]
        valid_ind = [i for i, mask in enumerate(all_masks) if os.path.splitext(os.path.basename(mask))[0] in valid_names]
        
        if self.train_wsi is not None:
            train_names = [os.path.splitext(wsi)[0] for wsi in self.train_wsi]
            train_ind = [i for i, mask in enumerate(all_masks) if os.path.splitext(os.path.basename(mask))[0] in train_names and os.path.splitext(os.path.basename(mask))[0] not in valid_names]
        else:
            train_ind = [i for i in range(len(all_masks)) if i not in valid_ind]
        
        # base name of the model checkpoint

        dataset_train = self.TileDataset(
            list_of_masks=[all_masks[i] for i in train_ind],
            wsi_root=wsi_root,
            resolution=self.resolution,
            tile_size=self.tile_size,
            step_size=self.step_size,
            label2id=self.label2id,
            num_classes=len(self.label2id),
            dataset_save_path=self.dataset_save_path,
            data_augs=data_augs,
            print_function=self.accelerator.print,
        )
        self.accelerator.wait_for_everyone()
        dataset_valid = self.TileDataset(
            list_of_masks=[all_masks[i] for i in valid_ind],
            wsi_root=wsi_root,
            resolution=self.resolution,
            tile_size=self.tile_size,
            step_size=self.step_size,
            label2id=self.label2id,
            num_classes=len(self.label2id),
            dataset_save_path=self.dataset_save_path,
            data_augs=None,
            print_function=self.accelerator.print,
        )
        self.accelerator.wait_for_everyone()

        sampling_weights = dataset_train.sampling_weights
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.from_numpy(sampling_weights),
            num_samples=len(sampling_weights),
            replacement=True,
        )
        val_names = [os.path.splitext(os.path.basename(i))[0] for i in dataset_valid.list_of_masks]
        self.accelerator.print(f"Validation set B-numbers: {val_names}")
        train_collator = TrainCollator(ignore_index=self.ignored_index)
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            prefetch_factor=2,
            sampler=sampler,
            num_workers=6,
            pin_memory=True,
            collate_fn=train_collator,
        )      
        valid_loader = torch.utils.data.DataLoader(
            dataset_valid,
            batch_size=int(batch_size*1.25),
            num_workers=6,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=train_collator,
        )
        return train_loader, valid_loader
        

    def train(self):
        
        self.model, self.optimizer, self.scheduler, self.dataloader_train, self.dataloader_valid = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, self.dataloader_train, self.dataloader_valid
        )
        self.accelerator.wait_for_everyone()
        self.accelerator.print(f"\n###### FOLD {self.fold} ######")

        for epoch in range(self.start_epoch, self.num_epochs):
            self.current_epoch = epoch
            self.accelerator.print(f"\nEpoch {epoch+1}")
            
            train_loss = self.train_epoch()
            valid_loss = self.valid_epoch()
            
            self.train_results.append(train_loss)
            self.valid_results.append(valid_loss)
            self.save_loss_as_dict()
            if valid_loss < self.valid_min_loss:
                self.waiting_epochs = 0
                self.valid_min_loss = valid_loss
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    self.save_model()
            else:
                self.waiting_epochs += 1
                if self.waiting_epochs >= self.early_stopping:
                    self.accelerator.print(f'Early stopping fold {self.fold} at epoch {epoch+1}')
                    break
            
            torch.cuda.empty_cache()

        # del all variables that may be taking up gpu memory & load best model for inference
        self.model, self.optimizer, self.scheduler = self.accelerator.free_memory(self.model, self.optimizer, self.scheduler)


    def train_epoch(self) -> float:

        if self.train_by_fixed_batches:
            iterations=self.num_batches_per_epoch
        else:
            iterations=len(self.dataloader_train)
        running_loss = 0
        num_samples = 0
        ious_score = []

        loop = tqdm(self.dataloader_train, total=iterations, disable=not self.accelerator.is_local_main_process)

        for idx, batch in enumerate(loop):
            # break loop if we have trained on the specified number of batches
            if self.train_by_fixed_batches and idx == self.num_batches_per_epoch:
                break
            
            running_loss, num_samples = self.train_batch(batch, running_loss, num_samples)
            self.scheduler.step(self.current_epoch + idx / iterations)
            
            ious_score = self.valid_batch(batch, ious_score)
            loop.set_description(f"Epoch {self.current_epoch+1}")
            loop.set_postfix(loss = running_loss/num_samples)
            
        self.accelerator.print(f'Train set: Average epoch loss: {running_loss/num_samples:.4f}')
        ious_score = np.array([iou for sublist in ious_score for iou in sublist])
        ious_score = np.nanmean(ious_score, axis=0)
        # accelerator.print(ious_score)
        self.accelerator.print('Mean IoU per class: ', {self.id2label[idx]: round(ious_score[j], 5) for j, idx in enumerate(self.categories_to_eval)})
        
        return running_loss/num_samples   


    def train_batch(self, batch, running_loss, num_samples) -> Tuple[float, int]:
        # set model to train mode
        self.model.train()
        # reset the gradients back to zero
        self.optimizer.zero_grad()
        with self.accelerator.accumulate(self.model):
            outputs = self.model(
                pixel_values=batch["pixel_values"],
                mask_labels=[labels for labels in batch["mask_labels"]],
                class_labels=[labels for labels in batch["class_labels"]],
            )
            loss = outputs.loss
            self.accelerator.backward(loss)
        
        running_loss += self.accelerator.gather_for_metrics(loss).sum().item()
        num_samples += self.accelerator.gather_for_metrics(batch["pixel_values"]).size(0)

        # Update optimizer parameters
        self.optimizer.step()
        
        return running_loss, num_samples
    

    @torch.no_grad()
    def valid_batch(self, batch, ious_score:list, within_train_loop:bool=True) -> float:
        # evaluate model on training set 
        self.model.eval()
    
        # for train set evaluation, select a random subset of the training tiles
        if within_train_loop:
            indices = np.random.choice(len(batch['pixel_values']), len(batch['pixel_values'])//self.accelerator.num_processes, replace=False)
        else:
            indices = range(len(batch['pixel_values']))

        batch["pixel_values"] = batch["pixel_values"][indices]
        target_sizes = [(target.size(1), target.size(2)) for target in [batch["mask_labels"][i] for i in indices]]
        outputs = self.model(pixel_values=batch["pixel_values"]) if within_train_loop else self.model(pixel_values=batch["pixel_values"], mask_labels=[labels for labels in batch["mask_labels"]], class_labels=[labels for labels in batch["class_labels"]])
        batch = self.accelerator.gather_for_metrics(batch["original_segmentation_maps"][indices]).cpu()

        predicted_segmentation_maps = custom_post_process_semantic_segmentation(outputs, target_sizes=target_sizes, return_logits=False)
        predicted_segmentation_maps = self.accelerator.gather_for_metrics(torch.stack(predicted_segmentation_maps))    
        unsqueeze_first = True if predicted_segmentation_maps.shape[0] == 1 else False
        predicted_segmentation_maps = predicted_segmentation_maps.squeeze().unsqueeze(0).cpu().numpy() if unsqueeze_first else predicted_segmentation_maps.squeeze().cpu().numpy()
        unsqueeze_first = True if batch.shape[0] == 1 else False
        batch = batch.squeeze().unsqueeze(0).numpy() if unsqueeze_first else batch.squeeze().numpy()
        ious_score.append(
            list(
                get_iou_multiclass(y_true, 
                                   y_pred, 
                                   categories=self.categories_to_eval
                                   ) for y_true, y_pred in zip(batch, predicted_segmentation_maps)
                )
            )
        
        if within_train_loop:
            return ious_score
        else:
            loss = self.accelerator.gather_for_metrics(outputs.loss)
            return ious_score, loss.sum().item(), batch.shape[0]
    

    @torch.no_grad()
    def valid_epoch(self) -> float:
        if self.valid_by_fixed_batches:
        # take 30% of the validation set for validation
            iterations = int(0.3 * self.num_batches_per_epoch)
        else:
            iterations = len(self.dataloader_valid)

        self.model.eval()
        running_loss = 0
        num_samples = 0
        ious_score = []
        
        loop = tqdm(self.dataloader_valid, total=iterations, disable=not self.accelerator.is_local_main_process)


        for idx, batch in enumerate(loop):
            # break loop if we have trained on the specified number of batches
            if self.valid_by_fixed_batches and idx == iterations:
                break
            
            # delete entries in batch dictionary that can cause mismatch of dimensions (mask_labels, class_labels)
            ious_score, batch_loss, batch_size = self.valid_batch(batch, ious_score, within_train_loop=False)
            running_loss += batch_loss 
            num_samples += batch_size

            loop.set_description(f"Validation")
            loop.set_postfix(loss = running_loss/num_samples)
    
        self.accelerator.print(f'Validation set: Average loss: {running_loss/num_samples:.4f}')
        ious_score = np.array([iou for sublist in ious_score for iou in sublist])
        ious_score = np.nanmean(ious_score, axis=0)
        self.accelerator.print('Mean IoU per class: ', {self.id2label[idx]: round(ious_score[j],5) for j,idx in enumerate(self.categories_to_eval)})

        return running_loss/num_samples
    
    
    def load_best_epoch(self, encoder_name, decoder_model, out_indices):

        model = create_mask2former_from_checkpoint(self.model_save_path, self.label2id, encoder_name, decoder_model, out_indices)

        return model
    
    @torch.no_grad()
    def infer_on_validation_set(self):
        del self.model
        self.args_inference.checkpoint_path = self.load_best_epoch(self.args_inference.encoder_model, self.args_inference.decoder_model, self.args_inference.feature_layers)
        # evaluate on validation set (to get metrics)
        self.evaluate_fnc(self.args_inference)
        # perform inference on validation set (to get geojsons - WSI aggregation)
        for path_to_geojson in tqdm(self.dataloader_valid.dataset.list_of_masks, disable=not self.accelerator.is_local_main_process, desc='Inference', unit='WSI'):
            name = os.path.splitext(os.path.basename(path_to_geojson))[0]
            wsi_path = self.dataloader_valid.dataset._get_wsi_path(name)
            self.args_inference.wsi_path = wsi_path
            self.inference_fnc(self.args_inference)
    
        
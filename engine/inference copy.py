from tqdm import tqdm
import os
import shutil
import time
from typing import List, Tuple

import openslide
import numpy as np
import zarr
import cv2
import json
import gc

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

from data.dataset_seg import SlideDataset, TileDataset
from models.architectures.mask2former import TrainCollator
from utils.metrics import get_multi_class_metrics
from models.model_io import post_process_output, infer_collate_fn, get_model_funcs, get_model_class_from_model, InferCollator
from utils.geometry import create_mask_from_contours
from utils.postprocessing import post_process
from utils.wsi import detect_colors


@torch.no_grad()
def infer_tiles(model, file_paths:List[str]) -> List[np.ndarray]:
    model.eval()
    # _, _, post_process_output, _, _ = get_model_funcs(model)
    img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    preds = []

    for file_path in file_paths:
        ann = zarr.open(file_path)
        tile = ann[:, :, :3]
        tile = img_transform(tile).unsqueeze(0)

        pred = model(pixel_values=tile.to(model.device))
        target_sizes = [(t.shape[1], t.shape[2]) for t in tile]

        # gather for prediction mask creation
        pred = post_process_output(pred, target_sizes).cpu().squeeze().numpy()
        preds.append(pred)

    return preds    


def evaluate_wsi_slide(
        model,
        wsi_path:str, 
        annotation_path:str, 
        batch_size:int, 
        tile_size:int, 
        step_size:int, 
        resolution:float, 
        label2id:dict,
        crop_pred_edge:int,
        apply_post_processing:bool,
        retain_mucin:bool = True
) -> Tuple[dict, np.ndarray, np.ndarray, int, int]:
    
    model.eval()
    id2label = {v: k for k, v in label2id.items()}
    pred, level, downsampling_level, exact_resolution, tiling_downsample_factor, read_origin, time, time_tile = infer_wsi(model, wsi_path, np.ones((5,5), dtype=np.uint8), batch_size, tile_size, step_size, crop_pred_edge, resolution)
    with open(annotation_path) as f:
        geojson = json.load(f)

    gt = create_mask_from_contours(geojson, label2id, pred.shape, downsampling_level, order=list(label2id.values()))

    if apply_post_processing:
        min_area = int(((600/2) / (exact_resolution)*tiling_downsample_factor) ** 2 * np.pi) # Min diameter of 600 um, converted to pixels square
        mucin = pred == label2id['Mucin']
        pred = post_process(segmentation_mask=pred,
                            lymph_node_class=label2id['Lymph node'],
                            classes_to_merge=[label2id['Primary tumor'], label2id['Mucin']],
                            merge_thresholds=[0.95, 0.05],
                            erase_thresholds=[0.075, 0.01],
                            apply_opening=[True, False],
                            min_ln_area=min_area)
        if retain_mucin:
            pred[mucin] = label2id['Mucin']
        else:
            gt = post_process(segmentation_mask=gt,
                            lymph_node_class=label2id['Lymph node'],
                            classes_to_merge=[label2id['Mucin']],
                            merge_thresholds=[0.05],
                            erase_thresholds=[0.01],
                            apply_opening=[False],
                            min_ln_area=min_area)

        # # remove LNs detected in noise
        ln_mask = (pred == label2id['Lymph node']).astype(np.uint8)
        wsi = openslide.open_slide(wsi_path)
        num_labels, labeled_lns = cv2.connectedComponents(ln_mask)
        for i in range(1, num_labels):
            bbox = cv2.boundingRect((labeled_lns == i).astype(np.uint8))
            # read region of interest from the original WSI
            crop = wsi.read_region((read_origin[0]+bbox[0]*downsampling_level, read_origin[1]+bbox[1]*downsampling_level), level, (bbox[2]*tiling_downsample_factor, bbox[3]*tiling_downsample_factor))
            crop = np.array(crop)
            crop[crop[:, :, 3] == 0] = 255
            crop = cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)
            if tiling_downsample_factor > 1:
                crop = cv2.resize(crop, (crop.shape[1]//tiling_downsample_factor, crop.shape[0]//tiling_downsample_factor), interpolation=cv2.INTER_NEAREST)
            crop_ln = crop[ln_mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] > 0] 
            has_colors = detect_colors(crop_ln)
            if not has_colors:
                pred[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = label2id['Background']
                                 
        ln_mask = (gt == label2id['Lymph node']).astype(np.uint8)
        num_labels, labeled_lns = cv2.connectedComponents(ln_mask)
        for i in range(1, num_labels):
            aux = (labeled_lns == i).astype(np.uint8)
            area = cv2.countNonZero(aux)
            if area < min_area:
                gt[aux > 0] = label2id['Background']

        

    categories_to_eval = sorted(list(label2id.values()))
    iou, dice, mcc = get_multi_class_metrics(gt, pred, categories_to_eval)

    results = {}
    results["filename"] = os.path.splitext(os.path.basename(wsi_path))[0]
    results['time_inference_wsi'] = time
    results['time_inference_tile'] = time_tile

    for i, k in enumerate(categories_to_eval):
        results[f'dice_{id2label[k]}'] = dice[i]
        results[f'iou_{id2label[k]}'] = iou[i]
        results[f'mcc_{id2label[k]}'] = mcc[i]
    
    return results, pred, gt, level, downsampling_level


@torch.no_grad()
def evaluate_wsi_tiles(
        model, 
        wsi_root:str, 
        annotations_paths:List[str], 
        batch_size:int, 
        tile_size:int, 
        step_size:int, 
        resolution:float, 
        label2id:dict,
        dataset_save_path:str,
        ignore_index:int) -> dict:
    
    """
    Evaluate a list of whole slide images (WSIs) using a given model and compute various metrics.

    Args:
        model: The model used for evaluation.
        wsi_root (str): Root directory containing the WSI files.
        annotations_files (list): Paths to the annotation geojson files (GT).
        batch_size (int): Number of tiles to process in a batch.
        tile_size (int): Size of each tile extracted from the WSI.
        step_size (int): Step size for moving the tile extraction window.
        resolution (float): Resolution of the WSI.
        label2id (dict): Dictionary mapping class labels to IDs.
        dataset_save_path (str): Path to save the dataset.
        ignore_index (int): Index of class to ignore in the evaluation.

    Returns:
        dict: A dictionary containing filenames, coordinates, and computed metrics (dice, IoU, MCC).
    """

    # check that all entries in the annotation_paths list exist and are geojson
    for path in annotations_paths:
        assert os.path.exists(path), f"Path {path} does not exist"
        assert path.endswith('.geojson'), f"Path {path} is not a geojson file"
    
    # empty the dataset save path
    if os.path.exists(dataset_save_path):
        shutil.rmtree(dataset_save_path)
    os.makedirs(dataset_save_path, exist_ok=True)
    
    model.eval()
    model_class = get_model_class_from_model(model)
    # _, _, post_process_output, _, _ = get_model_funcs(model)
    dataset = TileDataset(
        list_of_masks=annotations_paths,
        wsi_root=wsi_root,
        resolution=resolution,
        tile_size=tile_size,
        step_size=step_size,
        label2id=label2id,
        num_classes=len(label2id),
        dataset_save_path=dataset_save_path,
        data_augs=None)
    
    id2label = {v: k for k, v in label2id.items()}
    collate_fn = TrainCollator(ignore_index) if model_class == 'Mask2FormerForUniversalSegmentation' else None
    tile_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    num_batches = len(tile_loader)
    dices, ious, mccs, coords, filenames = [], [], [], [], []
    categories_to_eval = [i for i in label2id.values() if i != ignore_index]
    with tqdm(total=num_batches, unit='Batch', desc="Batches", dynamic_ncols=True) as data_iterator:
        for batch in tile_loader:
            pixel_values = batch["pixel_values"].to(model.device)   
            mask_labels = [labels.to(model.device) for labels in batch["mask_labels"]]
            class_labels = [labels.to(model.device) for labels in batch["class_labels"]]
            outputs = model(
                pixel_values=pixel_values,
                mask_labels=mask_labels,
                class_labels=class_labels,
            )
            coord = batch["coords"]
            filename = batch["filename"]
            target_sizes = [(target.size(1), target.size(2)) for target in batch["mask_labels"]]
            batch = batch["original_segmentation_maps"]
            predicted_segmentation_maps = post_process_output(outputs, target_sizes).cpu()
            unsqueeze_first = True if batch.shape[0] == 1 else False
            batch = batch.squeeze().unsqueeze(0).numpy() if unsqueeze_first else batch.squeeze().numpy()
            unsqueeze_first = True if predicted_segmentation_maps.shape[0] == 1 else False
            predicted_segmentation_maps = predicted_segmentation_maps.squeeze().unsqueeze(0).numpy() if unsqueeze_first else predicted_segmentation_maps.squeeze().numpy()

            for pred, true in zip(predicted_segmentation_maps, batch):
                # account for ignored class, to avoid computing metrics on those pixels
                mask = true == ignore_index
                pred[mask] = ignore_index
                # erode metastasis class to avoid small false positives
                pred = cv2.erode(pred.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
                pred = cv2.dilate(pred.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)

                iou, dice, mcc = get_multi_class_metrics(true, pred, categories_to_eval)
                dices.append(dice)
                ious.append(iou)
                mccs.append(mcc)
            coords.extend(coord)
            filenames.extend(filename)
            
            data_iterator.update(1)

    # create aux dicts per metric, which now should be dict(str, list), where there is a key per class
    results = {}
    results["filename"] = filenames
    results["x_start"] = [coord[0] for coord in coords]
    results["x_end"] = [coord[1] for coord in coords]
    results["y_start"] = [coord[2] for coord in coords]
    results["y_end"] = [coord[3] for coord in coords]
    for i, k in enumerate(categories_to_eval):
        results[f'dice_{id2label[k]}'] = [d[i] for d in dices]
        results[f'iou_{id2label[k]}'] = [iou[i] for iou in ious]
        results[f'mcc_{id2label[k]}'] = [m[i] for m in mccs]

    return results


@torch.no_grad()
def infer_wsi(
        model, 
        wsi_path:str, 
        filter_mask:np.ndarray, 
        batch_size:int, 
        tile_size:int, 
        step_size:int, 
        crop_pred_edge:int, 
        resolution:float, 
        downsample_factor:int=1,
        normalize_input:bool=True) -> Tuple[np.ndarray, int, int, float, Tuple[int, int], float, float]:
    """
    Perform inference on a whole slide image (WSI) using a Mask2Former model.

    Args:
        model: The Mask2Former model used for inference.
        wsi_path (str): Path to the whole slide image file.
        filter_mask (np.ndarray): Binary mask on which to limit tile extraction.
        batch_size (int): Number of tiles to process in a batch.
        tile_size (int): Size of each tile extracted from the WSI.
        step_size (int): Step size for moving the tile extraction window.
        crop_pred_edge (int): Number of pixels to crop from the prediction edges.
        ln_class (int): Class label for lymph nodes.
        resolution (float): Resolution of the WSI.
        downsample_factor (int, optional): Factor by which to downsample the WSI. Defaults to 1.

    Returns:
        np.ndarray: The predicted mask for the WSI.
        int: The level of the WSI used for prediction.
        int: The downsampling level of the WSI.
        float: The exact resolution of the WSI in mpp.
        tuple: The origin coordinates of the WSI read.

    """
    
    model.eval()
    
    dataset = SlideDataset(
        wsi_path=wsi_path,
        filter_mask=filter_mask,
        downsample_factor=downsample_factor,
        resolution=resolution,
        tile_size=tile_size,
        step_size=step_size,
        crop_size=crop_pred_edge,
        apply_tta=False,
        rotations=[0],
        flips=[],
        color_jitter=None,
        noise=False,
        blur=False,
        gamma=False)
    start_time = time.time()
    times_tile = []
    num_tiles = len(dataset)
    cpus_per_task = int(os.getenv("SLURM_CPUS_PER_TASK", 1))
    print(f"Using {cpus_per_task} CPU cores for tile loading")
    tile_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cpus_per_task if cpus_per_task > 1 else 0,
        pin_memory=True,
        prefetch_factor=2 if cpus_per_task > 1 else None,
        drop_last=False,
        collate_fn=InferCollator(normalize=normalize_input)
    )
    # tile_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False, collate_fn=InferCollator(normalize=normalize_input))
    num_batches = (num_tiles + batch_size - 1) // batch_size
    with tqdm(total=num_batches, unit='Batch', desc="Batches", dynamic_ncols=True) as data_iterator:
        for batch, coords, augmentations in tile_loader:
            start_time_batch = time.time()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                target_sizes = [(t.shape[1], t.shape[2]) for t in batch]
                batch = model(pixel_values=batch.to(model.device))
                
                # gather for prediction mask creation
                batch = post_process_output(batch, target_sizes, return_logits=True)
            # times_tile.append((time.time() - start_time_batch) / len(outputs))      
                dataset.stitch_predictions(batch, coords, augmentations)
                
            data_iterator.update(1)  # Update the progress bar 

            if data_iterator.n % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
    pred_mask,__ = dataset.create_final_predictions(return_probs=False)
    # time_tile = np.mean(times_tile)
    time_tile = 0
    pred_mask = pred_mask[:dataset.original_shape[0]//dataset.tiling_downsample_factor, :dataset.original_shape[1]//dataset.tiling_downsample_factor].astype(np.uint8) 
    pred_mask *= cv2.resize(filter_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    level = dataset.level
    downsampling_level = dataset.level_downsampling
    exact_resolution = dataset.exact_resolution
    read_origin = dataset.read_origin
    tiling_downsample_factor = dataset.tiling_downsample_factor
    
    del dataset
    torch.cuda.empty_cache()
    gc.collect()
    
    return pred_mask, level, downsampling_level, exact_resolution, tiling_downsample_factor, read_origin, time.time() - start_time, time_tile
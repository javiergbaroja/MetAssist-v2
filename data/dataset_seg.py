import openslide

import os
from typing import Dict, List, Tuple, Optional
from glob import glob
from tqdm import tqdm

# image processing and array manipulation
import numpy as np
import cv2
import zarr

# torch 
import torch

# utils
from data.dataset_seg_base_chunked import BaseTileDataset, BaseSlideDataset
from utils.wsi import prepare_read_from_slide
from utils.geometry import extract_contours_from_geojson

#TODO: refactor to use new TileDataset within the trainer. 

class SlideDataset(BaseSlideDataset):
    """
    A dataset class for performing inference on whole slide images (WSIs) using a Mask2Former model.
    
    This class handles the preparation of WSIs, extraction of tiles, and stitching of predictions. It also supports test-time augmentations (TTA) to improve model robustness.

    Attributes:
        downsample_factor (int): Factor by which to downsample the WSI.
        filter_mask (np.ndarray): lymph node segmentation binary mask.
        tile_size (int): Size of each tile extracted from the WSI.
        step_size (int): Step size for moving the tile extraction window.
        wsi (openslide.OpenSlide): The WSI object.
        coords (List[Tuple[int, int, int, int]]): List of coordinates for tile extraction.
        num_augs (int): Number of augmentations to apply.
        flips_values (List[str]): List of flip augmentations to apply.
        rotations (List[int]): List of rotation augmentations to apply.
        color_jitter (Optional[str]): Color jitter augmentation to apply.
        apply_tta (bool): Whether to apply test-time augmentations.
        noise (Optional[bool]): Whether to apply noise augmentation.
        blur (Optional[bool]): Whether to apply blur augmentation.
        gamma (Optional[bool]): Whether to apply gamma correction augmentation.
        angles (List[int]): List of angles for rotation augmentations.
        flips (List[str]): List of flips for flip augmentations.
        wsi_pred (Optional[torch.Tensor]): Tensor to store the WSI predictions.
        count_map (Optional[torch.Tensor]): Tensor to store the count of overlapping tiles.
        crop_size (int): Number of pixels to crop from the prediction edges.
        level (int): The level of the WSI used for prediction.
        read_origin (Tuple[int, int]): The origin coordinates of the WSI read.
        padded_shape (Tuple[int, int]): The padded shape of the WSI.
    """
    def __init__(self, 
                 wsi_path:str,
                 filter_mask:np.ndarray,
                 resolution:float,
                 tile_size:int,
                 step_size:int,
                 downsample_factor:int,
                 crop_size:int,
                 apply_tta:bool,
                 rotations:Optional[List[int]],
                 flips:Optional[Tuple[str, str]],
                 color_jitter:Optional[str],
                 noise:Optional[bool],
                 blur:Optional[bool],
                 gamma:Optional[bool],):
        
        super().__init__(
            wsi_path=wsi_path,
            filter_mask=filter_mask,
            resolution=resolution,
            tile_size=tile_size,
            step_size=step_size,
            downsample_factor=downsample_factor,    
            crop_size=crop_size,
            apply_tta=apply_tta,
            rotations=rotations,
            flips=flips,
            color_jitter=color_jitter,
            noise=noise,
            blur=blur,
            gamma=gamma,
        ) 
    

class TileDataset(BaseTileDataset):
    def __init__(
        self,
        roi_class_name: str = "Training region",
        bg_threshold: float = 0.25,
        **kwargs):
        # list_of_masks:List[str],
        # wsi_root:List[str],
        # resolution:float,  # mpp
        # tile_size:int,
        # step_size:int,
        # label2id:dict,
        # num_classes:int,
        # dataset_save_path:str=None,
        # data_augs:Dict[str, Union[str, List[str]]]=None,
        # print_function=print,
        # infer_mode:bool=False):

        super().__init__(**kwargs)
        self.roi_class_name = roi_class_name
        self.bg_threshold = bg_threshold
        self.roi_id = self.label2id.get(self.roi_class_name)
        self.get_weights_and_coords()                 


    def get_geojson_multiclass_dicts(self, 
                                    path_to_geojson:str, 
                                    wsi:openslide.OpenSlide) -> Tuple[Dict[str, List[np.ndarray]],Dict[str, List[np.ndarray]], int, int, int, Tuple[int, int], Tuple[int, int]]:     
        """
        Parse a GeoJSON annotation file and return per-class contours, holes, and slide geometry.

        This takes the WSI to obtain level information and downsampling factors, reads the
        GeoJSON annotations, and converts polygons into per-class contour and hole dictionaries at
        the chosen resolution.

        Args:
            path_to_geojson (str): Path to the GeoJSON file containing region annotations.
            wsi (openslide.OpenSlide): OpenSlide object for the corresponding whole-slide image.

        Returns:
            Tuple containing:
                - polygon_dict_contours (Dict[str, List[np.ndarray]]): Per-class outer contours.
                - polygon_dict_holes (Dict[str, List[np.ndarray]]): Per-class inner holes to exclude.
                - level (int): OpenSlide level used for subsequent reads.
                - level_downsampling (int): Downsampling factor for the selected level.
                - tiling_downsample_factor (int): Additional tiling downsample factor applied.
                - original_shape (Tuple[int, int]): Full-resolution WSI height and width. Including offsets if any.
                - read_origin (Tuple[int, int]): Top-left origin (x, y) used when reading the slide to account for offsets.
        """
    
        level, level_downsampling, exact_resolution, tiling_downsample_factor, original_shape, read_origin = prepare_read_from_slide(wsi, self.resolution, file_type=os.path.splitext(wsi._filename)[1])
        
        data = self._load_geojson(path_to_geojson)

        polygon_dict_contours, polygon_dict_holes = extract_contours_from_geojson(data, self.label2id, level_downsampling)

        return polygon_dict_contours, polygon_dict_holes, level, level_downsampling, tiling_downsample_factor, original_shape, read_origin
   


    def get_weights_and_coords(self):
        count_list = []
        coordinates = []
        with tqdm(self.list_of_masks, unit='WSI', desc="Dataset Configuration", dynamic_ncols=True) as data_iterator:
            for mask in data_iterator:
                
                # tile_counter = 0
                count_list_mask = []
                tile_id = os.path.splitext(os.path.basename(mask))[0]
                path_to_class_count = os.path.join(self.dataset_save_path, f"{tile_id}_class_count.zarr")
                needs_extraction = True
                # check if all tiles have already been created and saved
                if os.path.exists(path_to_class_count) and os.path.isdir(os.path.join(self.dataset_save_path, tile_id)):
                    count_list_mask = zarr.open(path_to_class_count, mode='r')[:]
                    if len(glob(os.path.join(self.dataset_save_path, tile_id, f"*.zarr"))) == count_list_mask.shape[0]:
                        needs_extraction = False
                        coordinates.extend([[tile_id, path] for path in glob(os.path.join(self.dataset_save_path, tile_id, f"*.zarr"))])
                        count_list_mask = count_list_mask.tolist() 
                        # tile_counter = len(count_list_mask)               

                if needs_extraction:
                    wsi = self._get_wsi(tile_id, return_openslide=True)
                    ann_contours, ann_holes, level, level_ds, tiling_ds, orig_dim, read_origin = self.get_geojson_multiclass_dicts(mask, wsi)
                    
                    # Check if valid ROI polygons exist
                    if self.roi_id is not None and len(ann_contours.get(self.roi_id, [])) > 0:
                        # ROI Strategy
                        region_polys = ann_contours[self.roi_id]
                        region_holes = ann_holes.get(self.roi_id, [])
                    else:
                        # Global Strategy (Fallback)
                        # Define one giant rectangle covering the whole slide dimensions at this level
                        # Dimensions from prepare_read_from_slide are usually (w, h)
                        w, h = orig_dim 
                        global_rect = np.array([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=np.int32)
                        region_polys = [global_rect]
                        region_holes = []

                    
                    # We open the slide now for the heavy reading
                    slide_tiles, slide_counts = self._process_regions(
                        tile_id, wsi, 
                        region_polys, region_holes, 
                        ann_contours, ann_holes,
                        level, level_ds, tiling_ds, read_origin)
                    
                    # --- 4. Save Cache ---
                    if slide_counts:
                        # Save counts for weighting
                        zarr.save(path_to_class_count, np.stack(slide_counts))
                        count_list.extend(slide_counts)
                        coordinates.extend(slide_tiles)
                    elif not slide_tiles:
                        print(f"Warning: No valid tiles found for {tile_id}")

        self.indices = coordinates
        self._calculate_sampling_weights(count_list)


    def _calculate_sampling_weights(self, all_counts: List[np.ndarray]):
        """
        Calculates sampling weights.
        """
        if not all_counts:
            print("Warning: No class counts available. Using uniform weights.")
            self.sampling_weights = np.ones(len(self.indices))
            return

        # Stack into one big matrix (Total_Tiles, Num_Classes)
        # Each element in all_counts is (N_slide_tiles, N_classes)
        try:
            counts = np.vstack(all_counts)
        except ValueError as e:
            print(f"Error stacking counts: {e}. Checking dimensions...")
            # Fallback debug print if dimensions mismatch (e.g. different num classes)
            return

        # Double check alignment
        if len(counts) != len(self.indices):
             print(f"Warning: Count mismatch! {len(counts)} counts vs {len(self.indices)} indices.")
        
        # --- Weight Calculation Logic ---
        
        # 1. Total pixels per class across the entire dataset
        total_counts_per_class = counts.sum(axis=0, keepdims=True)
        
        # Avoid division by zero
        total_counts_per_class[total_counts_per_class == 0] = 1.0
        
        # 2. Normalize tile counts by global class frequency
        # Tiles with many pixels of a rare class get a high value
        normalized_counts = counts / total_counts_per_class
        
        # 3. Sum contributions of all classes in the tile
        self.sampling_weights = normalized_counts.sum(axis=1)
        
        # Handle NaNs
        self.sampling_weights = np.nan_to_num(self.sampling_weights)
                            
                    
    def _process_regions(self, 
                         slide_id:str, 
                         slide:openslide.OpenSlide, 
                         region_polys:List[np.ndarray], region_holes:List[np.ndarray], 
                         ann_contours:Dict[int, List[np.ndarray]], ann_holes:Dict[int, List[np.ndarray]],
                         level:int, level_ds:int, tiling_ds:int, read_origin:Tuple[int, int],
                         use_tissue_filter:bool=False) -> Tuple[List[List], List[List[int]]]:
        """
        For each annotated region, this method crops the
        corresponding WSI area at the target level, builds ROI and per-class ground-truth
        masks (respecting holes), and iterates a tiling grid.
        Tiles that exceed the background threshold or fail the tissue filter are skipped.
        Accepted tiles are saved to disk (image + mask stacked), and per-class pixel counts
        (excluding class 0) are accumulated for sampling weights.

        Args:
            slide_id (str): Identifier of the slide, used for naming outputs.
            slide (openslide.OpenSlide): OpenSlide handle for reading image regions.
            region_polys (List[np.ndarray]): List of ROI polygons at the target level.
            region_holes (List[np.ndarray]): Optional holes to exclude inside ROI polygons.
            ann_contours (Dict[int, List[np.ndarray]]): Per-class outer contours at the target level.
            ann_holes (Dict[int, List[np.ndarray]]): Per-class inner holes to remove from contours.
            level (int): OpenSlide level used for reading tiles.
            level_ds (int): Downsampling factor for the selected level.
            tiling_ds (int): Additional downsample factor applied during tiling.
            read_origin (Tuple[int, int]): (x, y) offset in level-0 coords for slide reads.
            use_tissue_filter (bool): If True, apply a color-based tissue presence filter. Defaults to False.

        Returns:
            Tuple[List[List[Any]], List[List[int]]]:
                - slide_coords: `[slide_id, tile_path]` entries for each saved tile.
                - slide_counts: Per-tile pixel counts for each class (excluding 0).
        """
        slide_coords = []
        slide_counts = []
        
        # Sort regions by area to handle potential nesting or just order deterministically
        region_polys = sorted(region_polys, key=lambda x: cv2.contourArea(x), reverse=True)
        
        os.makedirs(os.path.join(self.dataset_save_path, slide_id), exist_ok=True)
        
        tile_dim = self.tile_size * tiling_ds
        step_dim = self.step_size * tiling_ds

        for idx, region_poly in enumerate(region_polys):
            # --- A. Define Crop ---
            x_min, y_min = np.min(region_poly[:, :, 0]), np.min(region_poly[:, :, 1])
            x_max, y_max = np.max(region_poly[:, :, 0]), np.max(region_poly[:, :, 1])
            
            # Align crop to tile grid (padding)
            pad_x = tile_dim - (x_max - x_min) % tile_dim
            pad_y = tile_dim - (y_max - y_min) % tile_dim
            
            crop_w = (x_max + pad_x) - x_min
            crop_h = (y_max + pad_y) - y_min
            
            # Read Region
            # openslide.read_region takes (x,y) in level 0 coords
            loc_x = int(read_origin[0] + x_min * level_ds)
            loc_y = int(read_origin[1] + y_min * level_ds)
            
            try:
                img_crop = np.array(slide.read_region((loc_x, loc_y), level, (crop_w, crop_h)))
            except Exception as e:
                print(f"Read error at {loc_x},{loc_y}: {e}")
                continue

            # Handle alpha channel
            img_crop[img_crop[:, :, 3] != 255] = 255
            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGBA2RGB)

            # --- B. Create Masks ---
            
            # 1. ROI / Validity Mask
            # 1 = Valid Region, 0 = Invalid
            roi_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
            adj_poly = region_poly - np.array([x_min, y_min])
            cv2.fillPoly(roi_mask, [adj_poly], 1)
            
            # Handle ROI holes (invalid areas inside the region)
            # Only relevant if using explicit ROI annotations
            if region_holes and not use_tissue_filter:
                 adj_holes = [h - np.array([x_min, y_min]) for h in region_holes]
                 cv2.fillPoly(roi_mask, adj_holes, 0)
            
            # 2. Ground Truth Mask
            # Start with valid ROI being "Background/Normal" (assuming ID is 1 or similar)
            # We assume 0 is "Ignore/Void". If label2id has a 'Background' class = 0, 
            # we might need to verify what ID represents "Unlabeled Tissue".
            # For this logic, let's assume valid tissue starts as ID 1 (e.g. 'Normal' or implicit).
            # If your label2id['Background'] == 0, then we rely on the fact that 
            # the training loop ignores 0, or 0 is valid background. 
            # Let's assume we fill with a placeholder '1' for valid unannotated tissue.
            
            gt_mask = roi_mask.copy()             # 1 where valid, 0 where invalid
            gt_mask_aux = np.zeros_like(gt_mask)  # Auxiliary mask for each class
            # Paint Foreground Classes (Everything except ROI and Background)
            # We exclude 0 (Ignore) and the ROI ID itself
            fg_classes = [c for c in self.classes if c != self.roi_id and c != 0]
            
            for cls_id in fg_classes:
                polys = ann_contours.get(cls_id, [])
                holes = ann_holes.get(cls_id, [])
                if len(polys) == 0: continue
                
                # Adjust coords
                adj_polys = [p - np.array([x_min, y_min]) for p in polys]
                adj_holes = [h - np.array([x_min, y_min]) for h in holes]
                
                # Reset the layer mask for the current class
                gt_mask_aux.fill(0)
                
                # 1. Define the "Positive" shape (The Class Polygon)
                cv2.fillPoly(gt_mask_aux, adj_polys, 1)
                
                # 2. Remove Holes
                if adj_holes:
                    cv2.fillPoly(gt_mask_aux, adj_holes, 0)

                # 3. Update the main gt_mask where roi_mask is valid
                gt_mask[(gt_mask_aux == 1) & (roi_mask == 1)] = cls_id

            # --- C. Tiling Loop ---
            
            for x_adj in range(0, crop_h - tile_dim + 1, step_dim):
                for y_adj in range(0, crop_w - tile_dim + 1, step_dim):
                    
                    tile_img = img_crop[x_adj:x_adj+tile_dim, y_adj:y_adj+tile_dim]
                    tile_gt = gt_mask[x_adj:x_adj+tile_dim, y_adj:y_adj+tile_dim]
                    tile_roi = roi_mask[x_adj:x_adj+tile_dim, y_adj:y_adj+tile_dim]

                    # Check 1: ROI Validity
                    # If > 25% of the tile is outside the defined Region (Class 0 in roi_mask), skip.
                    invalid_pixel_ratio = np.sum(tile_roi == 0) / tile_roi.size
                    if invalid_pixel_ratio > self.bg_threshold:
                        continue
                            
                    # --- Save ---
                    x_global = int(x_min + x_adj)
                    y_global = int(y_min + y_adj)
                    
                    tile_count = len(slide_coords)
                    tile_name = f"{tile_count}_x_{x_global}_{x_global+tile_dim}_y_{y_global}_{y_global+tile_dim}.zarr"
                    tile_path = os.path.join(self.dataset_save_path, slide_id, tile_name)
                    
                    if not os.path.exists(tile_path):
                        # Resize if we used a tiling downsample > 1
                        if tiling_ds > 1:
                            tile_img = cv2.resize(tile_img, (self.tile_size, self.tile_size))
                            tile_gt = cv2.resize(tile_gt, (self.tile_size, self.tile_size), interpolation=cv2.INTER_NEAREST)
                        
                        # Stack
                        combined = np.dstack((tile_img, tile_gt))
                        zarr.save(tile_path, combined)
                    
                    # Compute counts (ignoring 0)
                    counts = [np.count_nonzero(tile_gt == c) for c in self.classes if c != 0]
                    slide_counts.append(counts)
                    slide_coords.append([slide_id, tile_path])
        
        return slide_coords, slide_counts
        

    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, str]:

        filename, path_to_tile = self.indices[idx] 
        
        # 1. Load Data from Zarr
        array = zarr.open(path_to_tile, mode='r')[:]
        original_image = array[..., :3]
        original_segmentation_map = array[..., 3] 

        if np.any(original_segmentation_map < 0):
            raise ValueError(f"Segmentation map contains values < 0: {np.unique(original_segmentation_map)}")
        
        # 2. Parse Coordinates from Filename
        # Format: {counter}_x_{x1}_{x2}_y_{y1}_{y2}.zarr
        path_to_tile = os.path.splitext(os.path.basename(path_to_tile))[0].split('_')
        coords = int(path_to_tile[2]), int(path_to_tile[3]), int(path_to_tile[5]), int(path_to_tile[6])
        
        # image transformations
        image, segmentation_map = self.apply_transforms(original_image, original_segmentation_map)
        return image, segmentation_map, original_image, original_segmentation_map, coords, filename
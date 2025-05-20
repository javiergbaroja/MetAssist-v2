import openslide

import os
from typing import Dict, List, Tuple, Union, Optional
from glob import glob
from tqdm import tqdm

# image processing and array manipulation
import numpy as np
import cv2
import zarr
import geojson

# torch 
import torch

# utils
from datasets.dataset_seg_base import BaseTileDataset, BaseSlideDataset
from utils.data import unpack_coords, prepare_read_from_slide
from utils.utils import extract_contours_from_geojson

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
        list_of_masks:list,
        wsi_root:list,
        resolution:float,  # mpp
        tile_size:int,
        step_size:int,
        label2id:dict,
        num_classes:int,
        dataset_save_path:str=None,
        data_augs:Dict[str, Union[str, List[str]]]=None,
        print_function=print,
        infer_mode:bool=False):

        super().__init__(list_of_masks, 
                         wsi_root, 
                         resolution, 
                         tile_size, 
                         step_size, 
                         label2id, 
                         num_classes,
                         dataset_save_path, 
                         data_augs, 
                         print_function, 
                         infer_mode)
    


    def get_geojson_multiclass_mask(self, 
                                    path_to_geojson:str, 
                                    slide:openslide.OpenSlide=None) -> Tuple[Dict[str, List[np.ndarray]],Dict[str, List[np.ndarray]], openslide.OpenSlide, int, int, Tuple[int, int]]:     
        """This function creates a binary mask from a geojson file.
        First, it loads the slide and extracts the dimensions of the H&E image.
        Then, it loads the geojson file and extracts the polygons for each category (training region and metastasis).


        Args:
            path_to_geojson (str): _description_
            slide (_type_, optional): _description_. Defaults to None.

        Returns:
            np.ndarray: 
        """

        if slide is None:
            filename = os.path.splitext(os.path.basename(path_to_geojson))[0]

            # openslide open H&E
            slide = self._get_wsi(filename)
        wsi_path = self._get_wsi(filename, return_openslide=False)
        level, level_downsampling, __, __, read_origin = prepare_read_from_slide(wsi_path, self.resolution, file_type=os.path.splitext(wsi_path)[1])
        
        with open(path_to_geojson) as f:
            data = geojson.load(f)

        polygon_dict_contours, polygon_dict_holes = extract_contours_from_geojson(data, self.label2id, level_downsampling)

        return polygon_dict_contours, polygon_dict_holes, slide, level, level_downsampling, read_origin
   


    def get_weights_and_coords(self):
        count_list = []
        coordinates = []
        with tqdm(self.list_of_masks, unit='WSI', desc="Dataset Configuration", dynamic_ncols=True) as data_iterator:
            for mask in data_iterator:
                
                tile_counter = 0
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
                        tile_counter = len(count_list_mask)               

                if needs_extraction:
                    ann_contours, ann_holes, slide, level, level_downsampling, read_origin = self.get_geojson_multiclass_mask(mask)
                    
                    # Extract polygons from the annotation dictionary
                    region_polygons = [np.array(poly, dtype=np.int32) for poly in ann_contours.get(self.label2id["Training region"], [])]
                    region_holes =[np.array(poly, dtype=np.int32) for poly in ann_holes.get(self.label2id["Training region"], [])]
                    metastasis_polygons = [np.array(poly, dtype=np.int32) for poly in ann_contours.get(self.label2id["Metastasis"], [])]
                    # sort by area, from largest to smallest
                    region_polygons = sorted(region_polygons, key=lambda x: cv2.contourArea(x), reverse=True)
                    metastasis_holes = [np.array(poly, dtype=np.int32) for poly in ann_holes.get(self.label2id["Metastasis"], [])]
                    # sort by area, from largest to smallest
                    metastasis_polygons = sorted(metastasis_polygons, key=lambda x: cv2.contourArea(x), reverse=True)

                    if len(region_polygons) == 0:
                        self.printf(f"No training regions found in {mask}")
                        continue

                    # Create a directory for the current WSI
                    os.makedirs(os.path.join(self.dataset_save_path, tile_id), exist_ok=True)
                    
                    # Process each region annotation separately
                    for region_poly in region_polygons:
                        # Compute bounding box for the current region polygon
                        x_min = np.min(region_poly[:, 0, 0])
                        y_min = np.min(region_poly[:, 0, 1])
                        x_max = np.max(region_poly[:, 0, 0])
                        y_max = np.max(region_poly[:, 0, 1])

                        # Before tiling, pad the cropped image and ground truth to the nearest multiple of the tile size with reflection padding
                        pad_x = self.tile_size - (x_max - x_min) % self.tile_size
                        pad_y = self.tile_size - (y_max - y_min) % self.tile_size
                        
                        # Crop the large image to this region's bounding box
                        cropped_image = np.array(slide.read_region((read_origin[0] + x_min*level_downsampling, read_origin[1] + y_min*level_downsampling), level, ((x_max+pad_x) - x_min, (y_max+pad_y) - y_min)))
                        cropped_image[cropped_image[:,:,3] != 255]=255
                        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGBA2RGB)
                        adjusted_region_poly = region_poly - np.array([x_min, y_min])
                        
                        # Create a region mask: fill the region polygon with label 1
                        region_mask = np.zeros(cropped_image.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(region_mask, [adjusted_region_poly], 1)
                        if len(region_holes) > 0:
                            adjusted_region_holes = [hole - np.array([x_min, y_min]) for hole in region_holes]
                            cv2.fillPoly(region_mask, adjusted_region_holes, 0)

                        # Create a metastasis mask: fill the metastasis polygons with label 2
                        metastasis_mask = np.zeros_like(region_mask, dtype=np.uint8)
                        adjusted_metastasis_polygons = [poly - np.array([x_min, y_min]) for poly in metastasis_polygons]
                        adjusted_metastasis_holes = [hole - np.array([x_min, y_min]) for hole in metastasis_holes]

                        # Get bounding box for the training region polygon
                        training_bbox = cv2.boundingRect(adjusted_region_poly)  # (x, y, w, h)

                        for metastasis_poly in adjusted_metastasis_polygons:
                            meta_bbox = cv2.boundingRect(metastasis_poly)

                            # Check for bounding box overlap between metastasis and training region
                            if not (
                                meta_bbox[0] > training_bbox[0] + training_bbox[2] or  # meta left > region right
                                meta_bbox[0] + meta_bbox[2] < training_bbox[0] or      # meta right < region left
                                meta_bbox[1] > training_bbox[1] + training_bbox[3] or  # meta top > region bottom
                                meta_bbox[1] + meta_bbox[3] < training_bbox[1]         # meta bottom < region top
                            ):
                                # Bounding boxes overlap ⇒ process metastasis object
                                cv2.fillPoly(metastasis_mask, [metastasis_poly], 1)

                                # Now erase holes that are fully within this metastasis object
                                meta_xmin, meta_ymin = meta_bbox[0], meta_bbox[1]
                                meta_xmax, meta_ymax = meta_bbox[0] + meta_bbox[2], meta_bbox[1] + meta_bbox[3]

                                for hole_poly in adjusted_metastasis_holes:
                                    hole_bbox = cv2.boundingRect(hole_poly)
                                    hole_xmin, hole_ymin = hole_bbox[0], hole_bbox[1]
                                    hole_xmax, hole_ymax = hole_bbox[0] + hole_bbox[2], hole_bbox[1] + hole_bbox[3]

                                    # Check if hole's bbox is fully inside this metastasis bbox
                                    if (meta_xmin <= hole_xmin <= meta_xmax and
                                        meta_ymin <= hole_ymin <= meta_ymax and
                                        meta_xmin <= hole_xmax <= meta_xmax and
                                        meta_ymin <= hole_ymax <= meta_ymax):
                                        cv2.fillPoly(metastasis_mask, [hole_poly], 0)
                        
                        metastasis_mask = metastasis_mask * region_mask  # remove metastasis outside the region
                        ground_truth = region_mask.copy()
                        ground_truth[metastasis_mask == 1] = 2
                        # make sure ground_truth only has 0s, 1s, and 2s
                        if np.any(ground_truth > 2):
                            raise ValueError(f"Ground truth contains values > 2: {np.unique(ground_truth)}")
                        if np.any(ground_truth < 0):
                            raise ValueError(f"Ground truth contains values < 0: {np.unique(ground_truth)}")
                        
                        # Tiling loop: extract non-overlapping tiles from the cropped image and ground truth. Stack and save
                        for x_adj in range(0, cropped_image.shape[0] - self.tile_size + 1, self.step_size):
                            for y_adj in range(0, cropped_image.shape[1] - self.tile_size + 1, self.step_size):
                                tile_img = cropped_image[x_adj:x_adj + self.tile_size, y_adj:y_adj + self.tile_size]
                                tile_gt = ground_truth[x_adj:x_adj + self.tile_size, y_adj:y_adj + self.tile_size]
                                if (tile_gt == 0).sum() / self.tile_size ** 2 > 0.25:
                                    continue
                                tile_counter += 1
                                
                                # find original coordinates
                                x = x_min + x_adj
                                y = y_min + y_adj
                                path_to_tile = os.path.join(self.dataset_save_path, tile_id, f"{tile_counter}_x_{x}_{x + self.tile_size}_y_{y}_{y + self.tile_size}.zarr")
                                if not os.path.exists(path_to_tile) and self.current_process == 'main':
                                    combined_view = np.concatenate((tile_img, tile_gt[..., np.newaxis]), axis=2)
                                    zarr_array = zarr.open(path_to_tile, mode='w', shape=combined_view.shape, dtype=combined_view.dtype, chunks=(self.tile_size, self.tile_size, combined_view.shape[2]), compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2))
                                    zarr_array[:] = combined_view

                                tmp_list = []
                                for c in self.classes[1:]: # exclude 0s (parts of the image that are not annotated)
                                    tmp_list.append(np.count_nonzero(tile_gt == c))
                                count_list_mask.append(np.stack(tmp_list))

                                coordinates.append([tile_id, path_to_tile])

                    if tile_counter == 0:
                        if self.current_process == 'main':
                            os.rmdir(os.path.join(self.dataset_save_path, tile_id))
                        continue
                    elif not os.path.exists(path_to_class_count) and self.current_process == 'main' :
                        zarr_array = zarr.open(path_to_class_count, mode='w', shape=(len(count_list_mask), self.num_classes-1), dtype=np.int32, chunks=(len(count_list_mask), self.num_classes), compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2))
                        zarr_array[:] = np.stack(count_list_mask) if len(count_list_mask) > 0 else np.zeros((0, self.num_classes))
                    else: 
                        raise ValueError(f"File {path_to_class_count} already exists. Please remove it and rerun the script.")
                count_list.extend(count_list_mask)

        counts = np.stack(count_list)  # shape = n_samples x classes
        sum_counts = counts.sum(0)[np.newaxis :, ...]
        sampling_weights = (
            counts / sum_counts
        )  
        # replace nan for 0
        sampling_weights[np.isnan(sampling_weights)] = 0
        self.sampling_weights = sampling_weights.sum(1)  # n_samples
        # save sampling weights in dataset folder
        self.indices = coordinates
        self.printf(f"{len(self.indices)} tiles in dataset from {len(self.list_of_masks)} WSIs")
        self.printf(f"Tile dimensions: {self.tile_size}x{self.tile_size} pixels, for a FOV of {self.tile_size*self.working_resolution}x{self.tile_size*self.working_resolution} \u00B5m\u00B2 ({self.resolution} mpp) ")
        self.printf(f"Percent of pixels per category: {[round(100*c, 2) for c in sum_counts / sum_counts.sum()]}")
        self.printf('Categories to use: ', self.categories_to_use)
        self.printf('Ignored categories: ', self.ignored_categories) 
        

    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, str]:

        filename, path_to_tile = self.indices[idx] 
        # load numpy file
        array = zarr.open(path_to_tile, mode='r')[:]
        path_to_tile = os.path.splitext(os.path.basename(path_to_tile))[0].split('_')
        
        coords = int(path_to_tile[2]), int(path_to_tile[3]), int(path_to_tile[5]), int(path_to_tile[6])
        original_image = array[..., :3]
        original_segmentation_map = array[..., 3] #- 1 # 0: background, 1: metastasis, it was 1: background, 2: metastasis
        if np.any(original_segmentation_map > 2):
            raise ValueError(f"Segmentation map contains values > 2: {np.unique(original_segmentation_map)}")
        if np.any(original_segmentation_map < 0):
            raise ValueError(f"Segmentation map contains values < 0: {np.unique(original_segmentation_map)}")
        # image transformations
        image, segmentation_map = self.apply_transforms(original_image, original_segmentation_map)
        return image, segmentation_map, original_image, original_segmentation_map, coords, filename

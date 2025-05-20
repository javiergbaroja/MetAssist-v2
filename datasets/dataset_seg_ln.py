import openslide

import os
from typing import Dict, List, Tuple, Union, Optional
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
from utils.utils import create_mask_from_contours


class SlideDataset(BaseSlideDataset):
    """
    A dataset class for performing inference on whole slide images (WSIs) using a Mask2Former model.
    
    This class handles the preparation of WSIs, extraction of tiles, and stitching of predictions. It also supports test-time augmentations (TTA) to improve model robustness.

    Attributes:
        downsample_factor (int): Factor by which to downsample the WSI.
        filter_mask (Union[np.ndarray, None]): lymph node segmentation binary mask.
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
                 filter_mask:Union[np.ndarray, None],
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
                 gamma:Optional[bool]):
        
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
            gamma=gamma
        )      


class TileDataset(BaseTileDataset):
    def __init__(
        self,
        list_of_masks:list,
        wsi_root:list,
        resolution:float=2.0,  
        tile_size:int=384,
        step_size:int=1,
        label2id:dict={'Background':0},
        num_classes:int=2,
        dataset_save_path:str=None,
        data_augs:Dict[str, Union[str, List[str]]]=None,
        print_function=print,
        infer_mode:bool=False,):

        super().__init__(
            list_of_masks=list_of_masks,
            wsi_root=wsi_root,
            resolution=resolution,
            tile_size=tile_size,
            step_size=step_size,
            label2id=label2id,
            num_classes=num_classes,
            dataset_save_path=dataset_save_path,
            data_augs=data_augs,
            print_function=print_function,
            infer_mode=infer_mode,
        )
    

    def get_polygon(self, annotation:dict, polygon_dict:Dict[str, List[np.ndarray]], level_downsampling:int) -> Dict[str, List[np.ndarray]]:         
        
        coord_list = annotation['geometry']['coordinates']
        if 'classification' not in annotation['properties'].keys():
            return polygon_dict
        category = annotation['properties']['classification']['name']

        if category not in polygon_dict.keys() and category not in self.ignored_categories:
            self.ignored_categories.append(category)
            return polygon_dict
        elif category in self.categories_to_use:
            return unpack_coords(coord_list, polygon_dict, category, level_downsampling)        
        else:
            return polygon_dict  
        
    def _load_geojson(self, path_to_geojson:str) -> dict:
        with open(path_to_geojson) as f:
            data = geojson.load(f)
        return data


    def load_annotations_from_geojson(self, path_to_geojson:str, categories_to_load:List[str], level_downsampling:int) -> Dict[str, List[np.ndarray]]:

        supported_ann_types = ['LineString', 'Polygon', 'MultiPolygon']
        data = self._load_geojson(path_to_geojson)

        # Extract and group polygons from geojson per class
        
        polygon_dict = {category: [] for category in categories_to_load}
        for annotation in data['features']:
            if annotation['geometry']['type'] not in supported_ann_types:
                self.printf(f"Unsupported annotation type: {annotation['geometry']['type']}")        
                continue
            polygon_dict = self.get_polygon(annotation, polygon_dict, level_downsampling)

        return polygon_dict
    
    def _get_gt_mask(self, path_to_geojson:str, mask_shape:tuple, level_downsampling:int) -> np.ndarray:
        
        annotation = self._load_geojson(path_to_geojson)
        try:
            gt_mask = create_mask_from_contours(annotation, {k:v for k,v in zip(self.categories_to_use, self.classes)}, mask_shape, level_downsampling, self.classes)
        except Exception as e:
            print(f"Error creating mask from geojson: {e}\nProbably happened because of multipolygon annotations, which are not supported yet. Using older method.")
            gt_mask = np.zeros(mask_shape, dtype=np.uint8)
            # get geojson

            # Load geojson data
            categories_to_fill = [category for i, category in enumerate(self.categories_to_use) if self.classes[i] != 0]
            values_to_fill = [i for i in self.classes if i != 0]
            polygon_dict = self.load_annotations_from_geojson(path_to_geojson, categories_to_fill, level_downsampling)

            # fill gt_mask
            for category, fill_value in zip(categories_to_fill, values_to_fill):
                if len(polygon_dict[category]) > 0:
                    polygons = polygon_dict[category]
                    cv2.fillPoly(gt_mask, pts=polygons, color=fill_value)
        return gt_mask


    def get_geojson_multiclass_mask(self, path_to_geojson:str, slide=None, binary:bool=False, return_slide:bool=False) -> np.ndarray:       

        if slide is None:
            filename = os.path.splitext(os.path.basename(path_to_geojson))[0]

            # openslide open H&E
            slide = self._get_wsi(filename)

        slide_path = self._get_wsi(filename, return_openslide=False)
        filetype = os.path.splitext(slide_path)[1]
        level, level_downsampling, __, original_dim, read_origin = prepare_read_from_slide(slide, self.resolution, filetype)
        newDim = (int(original_dim[1]), int(original_dim[0]))
        
        gt_mask_2d = self._get_gt_mask(path_to_geojson, (newDim[1], newDim[0]), level_downsampling)
        # one-hot encode the mask
        if binary:    
            if return_slide:
                slide = np.array(slide.read_region(read_origin, level, newDim))
                slide[slide[:,:,3] != 255]=255
                slide = cv2.cvtColor(slide, cv2.COLOR_RGBA2RGB)
                return np.eye(self.num_classes, dtype=np.uint8)[gt_mask_2d], slide
            else:   
                return np.eye(self.num_classes, dtype=np.uint8)[gt_mask_2d]
        else:
            if return_slide:
                slide = np.array(slide.read_region(read_origin, level, newDim))
                slide[slide[:,:,3] != 255]=255
                slide = cv2.cvtColor(slide, cv2.COLOR_RGBA2RGB)
                return gt_mask_2d, slide
            else: return gt_mask_2d   

    def adjust_to_tile_size(self, mask_array:np.ndarray, slide:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # pad mask if necessary. The padding is done by mirroring the edges of the mask and slide.
        # the padding should be applied to right and left, top and bottom.
        if mask_array.shape[0] % self.tile_size != 0:
            pad_top = (self.tile_size - mask_array.shape[0] % self.tile_size) // 2
            pad_bottom = self.tile_size - mask_array.shape[0] % self.tile_size - pad_top
            mask_array = np.pad(mask_array, ((pad_top, pad_bottom), (0, 0)), mode='reflect')
            slide = np.pad(slide, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='reflect')
        if mask_array.shape[1] % self.tile_size != 0:
            pad_left = (self.tile_size - mask_array.shape[1] % self.tile_size) // 2
            pad_right = self.tile_size - mask_array.shape[1] % self.tile_size - pad_left
            mask_array = np.pad(mask_array, ((0, 0), (pad_left, pad_right)), mode='reflect')
            slide = np.pad(slide, ((0, 0), (pad_left, pad_right), (0, 0)), mode='reflect')
        return mask_array, slide    
    
    def get_weights_and_coords(self):

        count_list = []
        coordinates = []
        with tqdm(self.list_of_masks, unit='WSI', desc="Dataset Configuration", dynamic_ncols=True) as data_iterator:
            for mask in data_iterator:
                tile_id = os.path.splitext(os.path.basename(mask))[0]
                # self.printf(f"Processing {tile_id}")
                mask_array, slide = self.get_geojson_multiclass_mask(mask, return_slide=True)
                mask_array, slide = self.adjust_to_tile_size(mask_array, slide)
                tile_counter = 0
                # create all potential tiles from mask_array
                for i in range(0, mask_array.shape[0] - self.tile_size, self.step_size):
                    for j in range(0, mask_array.shape[1] - self.tile_size, self.step_size):
                        view = mask_array[i : i + self.tile_size, j : j + self.tile_size]
                        view_rgb = slide[i : i + self.tile_size, j : j + self.tile_size, :]
                        tmp_list = []
                        for c in self.classes:
                            tmp_list.append(np.count_nonzero(view == c))
                        count_list.append(np.stack(tmp_list))
                        tile_counter += 1
                        
                        path_to_tile = os.path.join(self.dataset_save_path, f"{tile_id}_{tile_counter}_x_{i}_{i + self.tile_size}_y_{j}_{j + self.tile_size}.zarr")
                        
                        if not os.path.exists(path_to_tile) and self.current_process == 'main':
                            combined_view = np.concatenate((view_rgb, view[..., np.newaxis]), axis=2)
                            zarr_array = zarr.open(path_to_tile, mode='w', shape=combined_view.shape, dtype=combined_view.dtype, chunks=(self.tile_size, self.tile_size, combined_view.shape[2]), compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2))
                            zarr_array[:] = combined_view
                        coordinates.append([tile_id, path_to_tile])

        counts = np.stack(count_list)  # shape = n_samples x classes
        sum_counts = counts.sum(0)[np.newaxis :, ...]
        sampling_weights = (
            counts / sum_counts
        )  
        # replace nan for 0
        sampling_weights[np.isnan(sampling_weights)] = 0
        self.sampling_weights = sampling_weights.sum(1)  # n_samples
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
        original_image = array[..., :3]
        original_segmentation_map = array[..., 3]

        # image transformations
        image, segmentation_map = self.apply_transforms(original_image, original_segmentation_map)
        return image, segmentation_map, original_image, filename


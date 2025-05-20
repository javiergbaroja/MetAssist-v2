import openslide

import os
from typing import Dict, List, Tuple, Union, Optional
from glob import glob
import random

# image processing and array manipulation
import numpy as np
from PIL import Image
import cv2

# torch 
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

# utils
from utils.augmentations import identity_transformation, apply_pil_brightness_augmentation, apply_pil_hsv_augmentation, apply_pil_hed_augmentation, apply_pil_additive_noise, apply_pil_gaussian_blur, apply_pil_gamma_correction
from utils.augmentations import TestTimeAugmentation
from utils.data import extract_tile_coords, prepare_read_from_slide
from utils.utils import ACCEPTED_WSI_TYPES, check_wsi_exists_all_formats

class BaseSlideDataset():
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
        level_downsampling (int): The downsampling factor of the WSI level.
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
                 gamma:Optional[bool]):
        
        self.downsample_factor = downsample_factor
        self.filter_mask = filter_mask
        self.tile_size = tile_size
        self.step_size = step_size
        self._prepare_slide(wsi_path, resolution)
        self.coords = extract_tile_coords(self.wsi, filter_mask, self.original_shape, self.padded_shape, self.tile_size, self.step_size, self.read_origin, self.level, self.level_downsampling)
        self.num_augs = len(rotations) * (len(flips)+1) if apply_tta else 1
        # update coords according to num_augs. Each coord should be repeated num_augs times

        self.coords = [coord for coord in self.coords for _ in range(self.num_augs)]
        self.flips_values = []
        if 'h' in flips:
            self.flips_values.append('h')
        if 'v' in flips:
            self.flips_values.append('v')
        self.rotations = rotations
        self.color_jitter = color_jitter
        self.apply_tta = apply_tta
        self.noise = noise
        self.blur = blur
        self.gamma = gamma
        self.angles = []
        self.flips = []
        self.prepare_augmentation_series()
        self.wsi_pred = torch.empty(1)
        self.count_map = torch.empty(1)
        self.crop_size = crop_size if crop_size <= ((tile_size-step_size)//2) else ((tile_size-step_size)//2)


    def _prepare_slide(self, wsi_path:str, resolution:float):
        """This method will prepare the slide for reading and tile extraction.

        Args:
            wsi_path (str): Path to the WSI file. Only .mrxs and .svs files are supported.
            resolution (float): The resolution of the WSI in microns per pixel (mpp).
        """

        # get level from mpp
        self.wsi = openslide.open_slide(wsi_path)
        self.level, self.level_downsampling, self.exact_resolution, self.original_shape, self.read_origin = prepare_read_from_slide(self.wsi, resolution, file_type=os.path.splitext(wsi_path)[1])

        # if slide cannot contain an integer value of tile_size, pad it by mirror reflection at the end of the image
        pad_x, pad_y = 0, 0
        if self.original_shape[0] % self.tile_size != 0:
            pad_x = self.tile_size - self.original_shape[0] % self.tile_size
        if self.original_shape[1] % self.tile_size != 0:
            pad_y = self.tile_size - self.original_shape[1] % self.tile_size

        self.padded_shape = self.original_shape[0] + pad_x, self.original_shape[1] + pad_y

    

    def create_final_predictions(self, return_probs: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Finalize the prediction by normalizing aggregated probabilities and creating the segmentation map.
        """
        # Avoid division by zero in areas without any tile coverage
        self.count_map = torch.clamp(self.count_map, min=1.0)

        # Normalize the logits by the number of overlaps to get averaged logits
        averaged_probs = self.wsi_pred / self.count_map.unsqueeze(0).float()

        # Get the final segmentation map by taking the argmax over the class dimension
        # final_prediction = torch.argmax(averaged_probs, dim=0).cpu().numpy().astype(np.uint8)
        final_prediction = np.zeros((self.padded_shape[0], self.padded_shape[1]), dtype=np.uint8)
        # tile averaged_probs to fill final_prediction
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for row_start, row_end, col_start, col_end in self.coords:
            if row_end > final_prediction.shape[0]:
                row_end = final_prediction.shape[0]
            if col_end > final_prediction.shape[1]:
                col_end = final_prediction.shape[1]
            final_prediction[row_start:row_end, col_start:col_end] = averaged_probs[:, row_start:row_end, col_start:col_end].to(device).argmax(dim=0).cpu().numpy().astype(np.uint8)

        if return_probs:
            return final_prediction, averaged_probs.cpu().numpy()

        return final_prediction, None
    

    def prepare_augmentation_series(self):
        """This function will prepare a list of equal length to the coordinates list, where each entry specifies the series of augmentations to be applied
        """
        
        # create rotations and flips list, the operations that each tile should receive:
        # each rotation value should have 1 to 3 entries (original, flip left-right, flip top-bottom)
        existing_flips = [0] + self.flips_values
        angles = [[angle]*len(existing_flips) for angle in self.rotations]
        angles = [item for sublist in angles for item in sublist]
        flips = existing_flips * len(self.rotations)
        n_repeats = len(self.coords) / len(angles)
        self.angles = angles * int(n_repeats)
        self.flips = flips * int(n_repeats)


    def get_gt_mask_tiles(self, mask_gt:np.ndarray, coords:List[Tuple[int, int, int, int]]) -> np.ndarray:
        gt_mask_tiles = []
        for x, x_end, y, y_end in coords:
            gt_mask_tiles.append(mask_gt[x:x_end, y:y_end])
        return np.array(gt_mask_tiles)
    

    def stitch_predictions(self, tile_predictions:torch.Tensor, coords:List[Tuple[int, int, int, int]]) -> torch.Tensor:
        """
        Stitch predictions from tiles into the final whole-slide image prediction.
        Crops predictions by self.crop_size during aggregation to reduce boundary artifacts.
        """
        if len(self.wsi_pred ) == 1:
            # make a downsampled copy of the WSI for the final prediction
            downsampled_size = (self.padded_shape[0] // self.downsample_factor, self.padded_shape[1] // self.downsample_factor)
            self.wsi_pred = torch.zeros((tile_predictions.shape[1], *downsampled_size), dtype=torch.float32)
            self.count_map = torch.zeros(downsampled_size, dtype=torch.uint8)
        tile_predictions = torch.nn.functional.interpolate(tile_predictions, scale_factor=1/self.downsample_factor, mode='bilinear', align_corners=False)

        # Add tile logits to the final logits tensor, and count overlapping tiles per pixel
        crop_size = self.crop_size // self.downsample_factor
        for pred_logits, (row_start, __, col_start, __) in zip(tile_predictions, coords):
            # Crop edges of the tile predictions accounting for edges (cannot do center crop if tile touches the image edge)
            crop_start_row, crop_end_row, crop_start_col, crop_end_col = crop_size, -crop_size, crop_size, -crop_size
            row_start, col_start = row_start // self.downsample_factor, col_start // self.downsample_factor
            row_end, col_end = row_start + pred_logits.shape[1], col_start + pred_logits.shape[2]
            if row_start != 0: # if not at top edge, crop the top
                row_start += self.crop_size 
            else:
                crop_start_row = 0
            if row_end != self.padded_shape[0]:
                row_end -= self.crop_size
            else:
                crop_end_row = pred_logits.shape[1] // self.downsample_factor
            if col_start != 0:
                col_start += self.crop_size 
            else:
                crop_start_col = 0
            if col_end != self.padded_shape[1]:
                col_end -= self.crop_size 
            else:
                crop_end_col = pred_logits.shape[2] // self.downsample_factor
                
            cropped_pred_logits = pred_logits[:, crop_start_row:crop_end_row, crop_start_col:crop_end_col]

            # Add cropped logits to the final logits tensor
            self.wsi_pred[:, row_start:row_end, col_start:col_end] += cropped_pred_logits.softmax(dim=0)
            self.count_map[row_start:row_end, col_start:col_end] += 1
        
    def __len__(self) -> int:
        return len(self.coords)
    
    
    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, Tuple[int, int, int, int], Union[None, TestTimeAugmentation]]:
        coord = self.coords[idx]
        tile_size = (coord[3] - coord[2]), (coord[1] - coord[0])

        newLocation = (int(int(self.read_origin[0])+coord[2]*self.level_downsampling),int(int(self.read_origin[1])+coord[0]*self.level_downsampling))
        tile = np.array(self.wsi.read_region(newLocation, self.level, tile_size))
        if tile.shape[2] == 4:
            tile[:, :, 3] = 255
            tile = cv2.cvtColor(tile, cv2.COLOR_RGBA2RGB)
        
        # apply padding when necessary, it should be 255 padding (white)
        if tile.shape[0] != self.tile_size:
            pad = self.tile_size - tile.shape[0]
            tile = np.pad(tile, ((0, pad), (0, 0), (0, 0)), mode='constant', constant_values=255)
        if tile.shape[1] != self.tile_size:
            pad = self.tile_size - tile.shape[1]
            tile = np.pad(tile, ((0, 0), (0, pad), (0, 0)), mode='constant', constant_values=255)

        tile = Image.fromarray(tile)  
        if self.apply_tta:
            augmentation = TestTimeAugmentation(self.angles[idx], self.flips[idx], self.color_jitter, self.noise, self.blur, self.gamma)
            tile = augmentation(tile)
        else:
            augmentation = None
        tile = F.to_tensor(tile)
        
        return tile, coord, augmentation


class BaseTileDataset(Dataset):
    def __init__(
        self,
        list_of_masks:list,
        wsi_root:Union[str, List[str]],
        resolution:float=2.0,
        tile_size:int=384,
        step_size:int=1,
        label2id:dict={'Background':0},
        num_classes:int=2,
        dataset_save_path:str=None,
        data_augs:Dict[str, Union[str, List[str]]]=None,
        print_function=print,
        infer_mode:bool=False,):
        super().__init__()

        self.list_of_masks = list_of_masks
        self.wsi_root = self._get_wsi_roots(wsi_root)
        self.resolution = resolution
        self.tile_size = tile_size
        self.step_size = step_size
        self.classes = list(label2id.values())
        self.categories_to_use = list(label2id.keys())
        self.label2id = label2id
        self.ignored_categories = []
        self.num_classes = num_classes
        self.img_transform = transforms.Compose([
            transforms.Resize((self.tile_size, self.tile_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.tile_size, self.tile_size)),
        ])
        self.sampling_weights = None
        self.indices = None
        self.current_process = None
        self.level = 0
        self.level_downsampling = 0
        self.working_resolution = 0
        self.slide_cache = {}  # cache slide objects to reduce file opens overhead
        self.dataset_save_path = dataset_save_path
        self.printf = print_function
        os.makedirs(self.dataset_save_path, exist_ok=True)
        # initialization functions
        self.set_data_augs(data_augs)
        self.set_current_process()
        self.get_level_from_mpp()
        if not infer_mode:
            self.get_weights_and_coords()

    def _get_wsi_roots(self, wsi_root:Union[str, List[str]]) -> List[str]:
        if isinstance(wsi_root, str):
            wsi_root = glob(wsi_root)
        elif isinstance(wsi_root, list):
            wsi_root = [glob(wsi) for wsi in wsi_root]
            wsi_root = [item for sublist in wsi_root for item in sublist]

        have_wsi = [any([len(glob(os.path.join(wsi, f'*.{ext}'))) > 0 for ext in ACCEPTED_WSI_TYPES]) for wsi in wsi_root]
        wsi_root = [wsi for wsi, check in zip(wsi_root, have_wsi) if check]
        if len(wsi_root) == 0:
            raise ValueError("No WSI folders found. Please check the path.")
        return wsi_root

    def _set_color_augmentation(self, color_jitter: str):
        if color_jitter is None:
            self._apply_color_jitter = apply_pil_brightness_augmentation
        elif color_jitter == 'hsv':
            self._apply_color_jitter = apply_pil_hsv_augmentation
        elif color_jitter == 'hed':
            self._apply_color_jitter = apply_pil_hed_augmentation
        elif color_jitter == 'valid':
            self._apply_color_jitter = identity_transformation
    
    def set_data_augs(self, data_augs:Union[None, Dict[str, Union[str, bool, List[str]]]]) -> None:
        if data_augs is not None:
            for key, value in data_augs.items():
                if key == 'color':
                    if value not in ['hsv', 'hed', None, 'valid']:
                        raise ValueError(f"Invalid color augmentation value: {value}. Must be one of ['hsv', 'hed', None]")
                    self._set_color_augmentation(data_augs['color'])
                elif key == 'rotation':
                    if not isinstance(value, list):
                        raise ValueError(f"Invalid rotation augmentation value: {value}. Must be a list of integers")
                    if not all([i in [0, 90, 180, 270] for i in value]):
                        raise ValueError(f"Invalid rotation augmentation value: {value}. Must be a list of integers from [0, 90, 180, 270]")
                elif key == 'flip':
                    if not isinstance(value, list):
                        raise ValueError(f"Invalid flip augmentation value: {value}. Must be a tuple of strings")
                    if not all([i in ['h', 'v', 'None'] for i in value]):
                        raise ValueError(f"Invalid flip augmentation value: {value}. Must be a tuple of strings from ['h', 'v', 'None']")
                elif key == 'contrast':
                    if not isinstance(value, bool):
                        raise ValueError(f"Invalid contrast augmentation value: {value}. Must be a boolean")
                elif key == 'noise':
                    if not isinstance(value, bool):
                        raise ValueError(f"Invalid noise augmentation value: {value}. Must be a boolean")
                elif key == 'blur':
                    if not isinstance(value, bool):
                        raise ValueError(f"Invalid blur augmentation value: {value}. Must be a boolean")
                else:
                    raise ValueError(f"Invalid data augmentation key: {key}. Must be one of ['color', 'rotation', 'flip', 'contrast', 'noise', 'blur']")
        self.data_augs = data_augs

    def set_current_process(self):
        # main if not using gpu
        if not torch.cuda.is_available(): self.current_process='main'
        # or running on single gpu
        elif torch.cuda.is_available() and torch.cuda.device_count() == 1: self.current_process='main'
        # or running on multiple gpus and main process
        elif torch.cuda.is_available() and torch.cuda.device_count() > 1 and torch.distributed.get_rank() == 0: self.current_process='main'
        else: self.current_process='worker'


    def get_level_from_mpp(self):
        # get level from mpp
        filename = os.path.splitext(os.path.basename(self.list_of_masks[0]))[0]
        # define wsi root somewhere else, or another way of getting the wsi
        
        slide = self._get_wsi(filename)
        # get mpp from openslide properties
        mpp = slide.properties[openslide.PROPERTY_NAME_MPP_X]
        # convert to float numpy array
        mpp = np.array(mpp, dtype=np.float32)
        # get downsampling factor
        downsampling = self.resolution / mpp
        # get best level for downsampling
        self.level = slide.get_best_level_for_downsample(downsampling)
        self.level_downsampling = int(slide.level_downsamples[self.level])
        self.working_resolution = round(mpp * self.level_downsampling)
        if self.working_resolution != self.resolution:
            self.printf(f"Warning: Input resolution of {self.resolution} mpp is not available for selected WSI level ({self.level}). Using {self.working_resolution} mpp instead.")
    
    def _get_wsi(self, filename:str, return_openslide:bool=True) -> Union[openslide.OpenSlide, openslide.ImageSlide]:
        __, wsi_path = check_wsi_exists_all_formats(filename, self.wsi_root)
        return openslide.open_slide(wsi_path) if return_openslide else wsi_path

    def _get_wsi_path(self, filename:str) -> str:
        # define wsi root somewhere else, or another way of getting the wsi
        for wsi_root_i in self.wsi_root:
            wsi_path = os.path.join(wsi_root_i, filename + ".mrxs")
            if os.path.exists(wsi_path):
                return wsi_path
        raise FileNotFoundError(f"WSI file not found for {filename}")

    
    
    def apply_transforms(self, image:np.ndarray, mask:np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        # image transformations

        image = Image.fromarray(image.astype(np.uint8))
        mask = Image.fromarray(mask)
        if self.data_augs is not None:
            if self.data_augs['rotation'] is not None:
                degree = random.choice(self.data_augs['rotation'])
                image = image.rotate(degree)
                mask = mask.rotate(degree)
            if self.data_augs['flip'] is not None:
                flip = random.choice(self.data_augs['flip'])
                if flip == 'h':
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                elif flip == 'v':
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                    mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            if self.data_augs['color'] is not None:
                image = self._apply_color_jitter(image)
            if self.data_augs['noise']:
                image = apply_pil_additive_noise(image)
            if self.data_augs['blur']:
                image = apply_pil_gaussian_blur(image)
            if self.data_augs['contrast']:
                image = apply_pil_gamma_correction(image)
        
        if self.img_transform:
            image = self.img_transform(image)  
        if self.mask_transform:    
            mask = self.mask_transform(torch.from_numpy(np.array(mask)).unsqueeze(0)).squeeze().long()

        return image, mask     

    
    def __len__(self):
        return len(self.indices)  
    

    def get_weights_and_coords(self):
        raise NotImplementedError
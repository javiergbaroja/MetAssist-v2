import openslide
import os
from glob import glob
import numpy as np
import cv2
from typing import List, Tuple, Optional, Union, Literal
from warnings import warn
from skimage.morphology import remove_small_objects 
from scipy.ndimage import binary_fill_holes
ACCEPTED_WSI_TYPES = ['mrxs', 'svs', 'ndpi', 'tif', 'tiff']

def _get_level_for_resolution(slide:openslide.OpenSlide, target_res:int):
    available_mpp = [0.5*(float(slide.properties[openslide.PROPERTY_NAME_MPP_X]) + float(slide.properties[openslide.PROPERTY_NAME_MPP_X])) * i for i in slide.level_downsamples]
    if "HISTAI-endometrial" in slide._filename:
        available_mpp = [a/1000 * 0.5 for a in available_mpp]
    closest_level = np.argmin([np.abs(target_res - available_mpp[i]) for i in range(len(available_mpp))])
    level_downsampling = slide.level_downsamples[closest_level]
    # reading_resolution = available_mpp[closest_level]   
    downsample = round(np.log2(target_res / available_mpp[closest_level]))

    return closest_level, level_downsampling, downsample, available_mpp


def _get_thumbnail_dimensions_and_origin(slide: openslide.OpenSlide, lvl:int, lvl_ds: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    filetype = os.path.splitext(slide._filename)[1].lower()
    if filetype in ['.svs', '.tif', '.tiff', '.tif', '.ndpi']:
        thumb_dims = slide.level_dimensions[lvl]
        read_origin = (0, 0)
    elif filetype == '.mrxs':
        w, h = int(slide.properties[openslide.PROPERTY_NAME_BOUNDS_WIDTH]), int(slide.properties[openslide.PROPERTY_NAME_BOUNDS_HEIGHT])
        x, y = int(slide.properties[openslide.PROPERTY_NAME_BOUNDS_X]), int(slide.properties[openslide.PROPERTY_NAME_BOUNDS_Y])
        thumb_dims = (int(h//lvl_ds), int(w//lvl_ds))
        read_origin = (x, y)
    else:
        raise ValueError(f"Unsupported file type {filetype}.")
    
    return thumb_dims, read_origin

def prepare_read_from_slide(slide: Union[openslide.OpenSlide, str], 
                            resolution: float, 
                            file_type:Optional[str]=None) -> Tuple[int, int, int, float, Tuple[int, int], Tuple[int, int]]:
    """This function receives a slide object and the resolution from which to read the slide.
    It returns the level of the slide to read from, the original dimensions of the slide, and the origin coordinates of the slide read.

    Args:
        slide (Union[openslide.OpenSlide, str]): The slide object to read from.
        resolution (float): The resolution to read the slide from in microns per pixel (mpp).
        file_type (str): The file type of the slide. Supported types are '.svs' and '.mrxs'.

    Raises:
        ValueError: If the file type is not supported.
        ValueError: If the resolution is too high for the slide.

    Returns:
        level (int): The level of the slide to read from.
        level_downsampling (int): The number of downsamples to reach the chosen resolution.
        reading_resolution (float): The precise resolution of the slide at the chosen level in mpp.
        original_dim (Tuple[int, int]): The original dimensions of the slide.
        read_origin (Tuple[int, int]): The origin coordinates of the slide read.
    """

    if isinstance(slide, str):
        if os.path.isfile(slide):
            slide = openslide.open_slide(slide)
        else:
            raise ValueError(f"File {slide} not found.")
    elif isinstance(slide, openslide.OpenSlide):
        pass
    else:
        raise ValueError("slide must be a file path or an openslide.OpenSlide object.")

    closest_level, level_downsampling, downsample, available_mpp = _get_level_for_resolution(slide, resolution)

    if downsample < 0:
            raise ValueError(f"Resolution {resolution} is too high for the slide.")
    elif downsample > 0:
        warn(f"Chosen resolution not among available resolutions for the slide. Using closest available resolution {available_mpp[closest_level]} instead.")

    original_dim, read_origin = _get_thumbnail_dimensions_and_origin(slide, closest_level, level_downsampling)

    return int(closest_level), int(level_downsampling), available_mpp[closest_level], 2**downsample, original_dim, read_origin




def detect_tissue_mask(slide: Union[openslide.OpenSlide, str],
                       read_origin: Optional[Tuple[int, int]]=None, 
                       rm_background_method: Literal['saturation', 'fixed']='saturation',
                       ratio_object: float=1e-4,
                       resolution:float=8.0) -> Tuple[np.ndarray, float]:
    """
    Generates a tissue mask from a WSI thumbnail using saturation/Otsu thresholding.
    Returns the mask and the downsample factor of the thumbnail used.
    """

    # 0. If a path is provided, open the slide
    if isinstance(slide, str):
        if os.path.isfile(slide):
            slide = openslide.open_slide(slide)
        else:
            raise ValueError(f"File {slide} not found.")

    # 1. Determine optimal thumbnail level (~32x downsample)
    thumb_level, thumb_ds, __, __ = _get_level_for_resolution(slide, resolution)
    thumb_dims, read_origin = _get_thumbnail_dimensions_and_origin(slide, thumb_level, thumb_ds)
    # 2. Read and blur thumbnail
    thumbnail = np.array(slide.read_region(read_origin, thumb_level, (thumb_dims[1], thumb_dims[0])))
    # convert transparent pixels to white
    # transparency = thumbnail[:, :, 3] > 0
    thumbnail[thumbnail[:, :, 3] == 0] = 255
    thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_RGBA2RGB)
    # thumbnail = thumbnail[:, :, :3]  # Discard alpha channel if present
    thumbnail = cv2.blur(thumbnail, (5, 5))

    # 3. Apply Thresholding
    if rm_background_method == 'saturation':
        hsv = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)
        s_channel = hsv[:, :, 1]
        _, tissue_mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif rm_background_method == 'fixed':
        gray = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY)
        _, tissue_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        # tissue_mask *= transparency

    else:
        raise ValueError(f"Unknown background removal method: {rm_background_method}")
    
    # 4. Morphological Cleaning
    if ratio_object > 0:
        # Remove small objects (using a threshold relative to image size)
        min_size = int(ratio_object * tissue_mask.size)
        tissue_mask = remove_small_objects(tissue_mask.astype(bool), min_size=min_size).astype(np.uint8)
        print(f"Removing small objects smaller than {min_size} pixels.")

    disk_edge = np.ceil(np.max(tissue_mask.shape) * ratio_object).astype(int) if ratio_object > 0 else 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (disk_edge, disk_edge))
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
    tissue_mask = cv2.dilate(tissue_mask, kernel, iterations=1)
    # binary_fill_holes 

    
    return binary_fill_holes((tissue_mask > 0)).astype(np.uint8), thumb_ds



def detect_colors(image, threshold=0.0025):
    """This function detects the presence of purple, pink, or blue colors in an image

    Args:
        image (np.ndarray): Image to check for colors
        threshold (float): Threshold for the percentage of pixels that must be colored to be considered present

    Returns:
        bool: True if any of the specified colors are present, False otherwise
    """
    if image.size == 0:
        return False
    # check it is rgb, last channel should be 3
    if image.shape[-1] != 3:
        raise ValueError('Image should be RGB')
    # if shape is (n, 3), convert to (1, n, 3)
    if len(image.shape) == 2:
        image = image.reshape(1, image.shape[0], image.shape[1])

    # Downsample to half the size
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Convert the image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define HSV range for purples, pinks, and blues
    lower_bound = np.array([100, 35, 50])
    upper_bound = np.array([179, 255, 255])
    
    # Create mask for the color range
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    
    # Calculate the percentage of the image that is the specified colors
    color_pixels = np.sum(mask > 0)
    total_pixels = image.shape[0] * image.shape[1]
    color_percentage = color_pixels / total_pixels
    
    # Check if the percentage is greater than the threshold
    return color_percentage > threshold


def check_wsi_exists_all_formats(wsi_name:str, 
                                 wsi_path:Union[List, str], 
                                 wsi_format_only:bool=True) -> Tuple[bool, str]:
    exists, path = False, None
    if "*" not in wsi_name:
        wsi_name = wsi_name+'.*'

    if isinstance(wsi_path, str):
        wsi_path = [wsi_path]
    for p in wsi_path:
        paths = glob(os.path.join(p, '**', wsi_name), recursive=True)
        if wsi_format_only:
            paths = list(set([p for p in paths if os.path.splitext(p)[1][1:] in ACCEPTED_WSI_TYPES]))
        if len(paths) > 0:
            
            assert len(paths) == 1, f"Multiple files found for {wsi_name} in {p}"
            path = paths[0]
            assert os.path.isfile(path), f"File {path} does not exist"
            exists = True
            path = paths[0]
            return exists, path
        
    return exists, path



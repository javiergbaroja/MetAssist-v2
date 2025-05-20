import openslide
import os
import re
from glob import glob
import numpy as np
import cv2
import torch
import json
from typing import Mapping, Union, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict

ACCEPTED_WSI_TYPES = ['mrxs', 'svs', 'ndpi', 'tif', 'tiff']

COLORMAP = {
        'Background': (125, 125, 125),          # white
        'Lymph node': (229, 100, 84),           # orange
        'Tumor deposits': (212, 185, 60),       # dark yellow
        'Primary tumor': (54, 90, 113),         # blue
        'Primary tissue': (0, 124, 169),        # cyan
        'Ink': (11, 72, 205),                   # dark blue
        'Vessels': (106, 29, 125),              # purple
        'Metastasis': (117, 173, 81),           # light green
        'Necrosis': (50, 50, 50),               # grey
        'Connective tissue': (250, 71, 102),    # red
        'Folds': (73, 103, 40),                 # dark green
        'Fat tissue': (255, 255, 153),          # light yellow
        'Mucin': (220, 220, 220),               # light grey
        'Slide edge': (48, 213, 200),           # turquoise
        'Training region': (0, 0, 0)            # black
    }

def combine_results(output_dir:str):
    """This function combines the results of the SLURM array jobs into a single file. It checks if all jobs have finished successfully and then combines the results into a single file."""

    if int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)) == int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1)) - 1:
        # with while loop, check if all jobs have finished successfully
        still_running = True
        num_running_jobs = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
        pattern = re.compile(r"results_\d+_finished\.csv$")
        while still_running:
            # number of finished jobs
            num_finished_jobs = len([f for f in os.listdir(output_dir) if pattern.search(f)])
            if num_finished_jobs >= num_running_jobs:
                still_running = False
        
        # combine results
        for i in range(num_running_jobs):
            if i != 0:
                with open(os.path.join(output_dir, f'results_{i}_finished.csv'), 'r') as f:
                    data = f.readlines()
                with open(os.path.join(output_dir, 'results.csv'), 'a') as f:
                    f.writelines(data[1:])
                os.remove(os.path.join(output_dir, f'results_{i}_finished.csv'))
            else:
                os.rename(os.path.join(output_dir, f'results_{i}_finished.csv'), os.path.join(output_dir, 'results.csv'))
    print('Combined results')

def check_wsi_exists(wsi_name:str, wsi_path:str) -> Tuple[bool, str]:
    exists, path = False, None
    if '*' not in wsi_path:
        path = os.path.join(wsi_path, wsi_name)
        exists = os.path.exists(path)
    else:
        wsi_paths = glob(wsi_path)

        for wsi_path in wsi_paths:
            path = os.path.join(wsi_path, wsi_name)
            exists = os.path.exists(path)
            if exists:
                break
    return exists, path


def check_wsi_exists_all_formats(wsi_name:str, wsi_path:Union[List, str], wsi_format_only:bool=True) -> Tuple[bool, str]:
    exists, path = False, None
    if "*" not in wsi_name:
        wsi_name = wsi_name+'.*'

    # if '*' not in wsi_path:
    if isinstance(wsi_path, str):
        wsi_path = [wsi_path]
        # path = os.path.join(wsi_path, wsi_name)
        # paths = glob(path)
        # if wsi_format_only:
        #     paths = [p for p in paths if os.path.splitext(p)[1] in ACCEPTED_WSI_TYPES]

        # if len(paths) > 0:
        #     assert len(paths) == 1, f"Multiple files found for {wsi_name} in {wsi_path}"
        #     exists = True
        #     path = paths[0]
    # else:
    # elif isinstance(wsi_path, list):
    for p in wsi_path:
        path = os.path.join(p, wsi_name)
        paths = glob(path)
        if wsi_format_only:
            paths = [p for p in paths if os.path.splitext(p)[1][1:] in ACCEPTED_WSI_TYPES]
        if len(paths) > 0:
            assert len(paths) == 1, f"Multiple files found for {wsi_name} in {p}"
            exists = True
            path = paths[0]
            return exists, path
    return exists, path

def get_ann_files(path_to_ann:Union[str, List[str]], file_type:str='geojson')->List[str]:
    if isinstance(path_to_ann, list):
        path_to_ann = sorted([p for p in path_to_ann if p.endswith(f'.{file_type}')])
    elif path_to_ann.endswith(f'*.{file_type}'):
        return sorted(glob(path_to_ann))
    elif path_to_ann.endswith(f'.{file_type}'):
        return [path_to_ann] 
    elif os.path.isdir(path_to_ann):
        return sorted(glob(os.path.join(path_to_ann, f'*.{file_type}')))
    else:
        raise ValueError('Invalid path to annotation file(s). Please provide a directory or a list of files with the correct file type.')
    


def create_geojson(contours, hierarchy, level_downsampling:int, category:str, min_size:float=None) -> dict:
    output_geojson = {
    'type': 'FeatureCollection',
    'features': []
    }
    outer_contours = []
    holes = []
    for i, contour in enumerate(contours):
        contour = contour.squeeze()
        if len(contour.shape) == 1:
            continue

        if contour.shape[0] < 2:
            continue
        # ensure polygon is closed
        if not np.all(contour[0] == contour[-1]):
            contour = np.vstack([contour, contour[0]])

        contour *= level_downsampling
        # this is a cv2 contour, compute bbox
        if min_size is not None:
            x, y, w, h = cv2.boundingRect(contour)
            dim = max(w, h)
            if dim < min_size:
                continue

        if hierarchy[0][i][3] == -1:
            outer_contours.append(contour.tolist())
            holes.append([])
        else:
            holes[-1].append(contour.tolist())

        
    for contour, hole in zip(outer_contours, holes):
        feature = {
            'type': 'Feature',
            'properties': {
                'id': i,
                'type': category,
                'objectType': 'annotation',
                'classification': {
                    'name': category,
                    'color' : list(COLORMAP[category]),
                    }
            },
            'geometry': {
                'type': 'Polygon',
                'coordinates': [contour]
            }
        }

        if len(hole) > 0:
            feature['geometry']['coordinates'].extend(hole)
        output_geojson['features'].append(feature)

    return output_geojson

def sparse_encode(mask:np.ndarray)->tuple:
    """This function encodes a mask as a sparse representation, which is a tuple of coordinates, values, and the original shape of the mask

    Args:
        mask (np.ndarray): Mask to encode

    Returns:
        tuple: A tuple of coordinates, values, and the original shape of the mask
    """

    coords = np.column_stack(np.where(mask > 0))
    values = mask[mask > 0]
    original_shape = mask.shape
    return coords, values, original_shape

def contour_encode(mask:np.ndarray)->tuple:
    """This function encodes a mask as a contour representation, which is a tuple of outer contours and holes in each outer contour

    Args:
        mask (np.ndarray): Mask to encode

    Returns:
        tuple: A tuple of outer contours and holes in each outer contour
    """

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # the binary mask can have holes. we will save them as a json file, where we will have the following entries
    # contours: list of list of points [[[x1, y1], [x2, y2], ...], [ [x1, y1], [x2, y2], ...], ...]
    # holes: list of the same length as contours. Each item should have the wholes of the contour in that position

    # get outer contours and holes separately by using hierarchy
    outer_contours = []
    holes = []
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            outer_contours.append(contours[i])
            holes.append([])
        else:
            holes[-1].append(contours[i])
    return outer_contours, holes


def encode_contour_json(mask:np.ndarray, level_downsampling:int, category:str='', min_size:float=None) -> dict:

    # make sure it is uint8
    if not mask.dtype == np.uint8:
        mask = mask.astype(np.uint8)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return create_geojson(contours, hierarchy, level_downsampling, category, min_size)


def extract_contours_from_geojson(geojson:dict, 
                              category_dict:Dict[str, int], 
                              level_downsampling:int) -> Tuple[Dict[int, List[np.ndarray]], Dict[int, List[np.ndarray]]]:
    
    dict_of_outer_contour_lists = {k:[] for k in category_dict.values()}
    dict_of_hole_contour_lists = {k:[] for k in category_dict.values()}
        
    for feature in geojson['features']:
        if 'classification' in feature['properties']:
            category = feature['properties']['classification']['name']
        elif 'type' in feature['properties']:
            category = feature['properties']['type']
        else:
            continue
        if category not in category_dict.keys():
            continue
        category = category_dict[category]
        coordinates = feature['geometry']['coordinates']

        if len(coordinates) == 0:
            continue
        elif len(coordinates) >= 1 and isinstance(coordinates[0][0], (int, float)):
            coordinates = [coordinates]

        outer_contour = np.array(coordinates[0], dtype=np.int32) // level_downsampling
        outer_contour = remove_duplicate_points(outer_contour)
        if np.unique(outer_contour, axis=0).shape[0] > 3:
            # remove duplicates keeping the same order            
            dict_of_outer_contour_lists[category].append(outer_contour)      
        else:
            continue

        for hole in coordinates[1:]:
            hole = np.array(hole, dtype=np.int32) // level_downsampling
            hole = remove_duplicate_points(hole)
            if np.unique(hole, axis=0).shape[0] > 3:
                dict_of_hole_contour_lists[category].append(hole)

    return dict_of_outer_contour_lists, dict_of_hole_contour_lists


def create_mask_from_contours(geojson:Union[str, dict], 
                              category_dict:dict, 
                              mask_shape:Tuple[int,int], 
                              level_downsampling:int, 
                              order:List[int]=None) -> np.ndarray:
    """
    Creates a segmentation mask from GeoJSON contours.

    This function generates a 2D segmentation mask based on the contours provided in a GeoJSON file. 
    Each contour is assigned a category value based on the `category_dict`. The function also supports 
    handling overlapping regions by using a specified hierarchy (`order`).

    Args:
        geojson (dict): A dictionary containing GeoJSON data with features and their corresponding geometry.
        category_dict (dict): A dictionary mapping category names (from GeoJSON) to integer values for the mask.
        mask_shape (Tuple[int, int]): The shape of the output mask (height, width).
        level_downsampling (int): The downsampling factor to scale the coordinates from the GeoJSON file.
        order (List[int], optional): A list specifying the hierarchy of categories. If provided, overlapping 
                                    regions are resolved based on this order, where lower indices have higher priority.

    Returns:
        np.ndarray: A 2D segmentation mask of the specified shape, where each pixel is assigned a category value.

    Raises:
        AssertionError: If the `order` contains values not present in `category_dict`.
        Exception: If there is an error while drawing the outer contour.

    Notes:
        - The GeoJSON file must contain features with `classification` and `geometry` properties.
        - The `geometry` property should have `coordinates` in the format `List[List[List[float]]]`.
        - Holes within contours are supported and will be filled with a value of 0.
        - If `order` is not provided, overlapping regions are resolved by taking the maximum value.
    """

    if isinstance(geojson, str):
        with open(geojson, 'r') as f:
            geojson = json.load(f)
    elif not isinstance(geojson, dict):
        raise ValueError('geojson should be a string or a dictionary')
        
    mask = np.zeros(mask_shape, dtype=np.uint8)
    dict_of_outer_contour_lists, dict_of_hole_contour_lists = extract_contours_from_geojson(geojson, category_dict, level_downsampling)

    if order is None:
        for cat in dict_of_outer_contour_lists.keys():
            if len(dict_of_outer_contour_lists[cat]) == 0:
                continue
            cv2.fillPoly(img=mask, pts=dict_of_outer_contour_lists[cat], color=cat)
            
            if len(dict_of_hole_contour_lists[cat]) > 0:
                cv2.fillPoly(mask, dict_of_hole_contour_lists[cat], 0)

    else:
    # remove empty categories from order
        order = [cat for cat in order if len(dict_of_outer_contour_lists[cat]) > 0]
        for cat in order:
            aux_mask = np.zeros(mask_shape, dtype=np.uint8)
            cv2.drawContours(aux_mask, dict_of_outer_contour_lists[cat], -1, cat, thickness=cv2.FILLED)
            if len(dict_of_hole_contour_lists[cat]) > 0:
                cv2.drawContours(aux_mask, dict_of_hole_contour_lists[cat], -1, 0, thickness=cv2.FILLED)
            mask[(aux_mask > 0)] = cat
            
    return mask

def remove_duplicate_points(contour):
    contour = contour.reshape(-1, 2)  # (N, 1, 2) → (N, 2)
    seen = set()
    mask = []

    for pt in map(tuple, contour):
        if pt not in seen:
            seen.add(pt)
            mask.append(True)
        else:
            mask.append(False)

    unique_contour = contour[mask]
    # add original first point to the end
    if len(unique_contour) > 0 and not np.array_equal(unique_contour[0], unique_contour[-1]):
        unique_contour = np.vstack([unique_contour, unique_contour[0]])
    unique_contour =  unique_contour.reshape(-1, 1, 2).astype(np.int32)
    return unique_contour

def decode_geojson_to_mask(geojson_path: str) -> np.ndarray:
    with open(geojson_path, 'r') as f:
        geojson = json.load(f)
    
    mask = create_mask_from_contours(
        geojson,
        geojson['category_dict'],
        geojson['mask_shape'],
        level_downsampling=geojson['level_downsampling']
    )
    
    return mask

def get_contour_major_axis(contour:np.ndarray)->float:
    """This function calculates the major axis of a contour computed using cv2.findContours"""
    bounding =  cv2.minAreaRect(contour)
    return max(bounding[1])

def get_slide_level_result(mask:np.ndarray, ln_seg_mask:np.ndarray, metastasis_class:int, ln_class:int, deposit_class:int, fat_class:int, mucin_class:int, resolution:float, min_size:float=0.) -> Tuple[str, int]:
    """
    Determine the status of a whole slide image (WSI) based on the length of an object found in the mask.

    The function analyzes the contours in the binary mask to determine the major axis length of each contour. Based on the length of the major axis, the function classifies the WSI into one of the following categories:
    - 'negative': No significant findings.
    - 'itc' (isolated tumor cells): Maximum major axis length is between min_size and 200 micrometers.
    - 'micrometastasis': Maximum major axis length is between 200 and 2000 micrometers.
    - 'macrometastasis': Maximum major axis length is greater than or equal to 2000 micrometers.

    Args:
        mask (np.ndarray): Binary mask of the WSI where the regions of interest are marked.
        resolution (float): Resolution of the WSI in micrometers per pixel.
        min_size (float, optional): Minimum size of the contour to be considered. Default is 0.
    Returns:
        Tuple[str, int]: A tuple containing:
            - wsi_status (str): The status of the WSI, which can be 'negative', 'itc', 'micrometastasis', or 'macrometastasis'.
            - label (int): A binary label indicating the presence (1) or absence (0) of significant findings.
    """

    contours, _ = cv2.findContours((mask==metastasis_class).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    status = []
    label = 0
    wsi_status = 'negative'
    lengths = []
    for contour in contours:
        major_axis = get_contour_major_axis(contour) * resolution
        if major_axis > min_size and major_axis < 200:
            status.append('itc')
        elif major_axis >= 200 and major_axis < 2000:
            status.append('micrometastasis')
        elif major_axis >= 2000:
            status.append('macrometastasis')
        lengths.append(major_axis)
    major_axis = max(lengths) if len(lengths) > 0 else 0
    # check if there are any macrometastasis
    if 'macrometastasis' in status:
        label = 1
        wsi_status = 'macrometastasis'
    elif 'micrometastasis' in status:
        wsi_status = 'micrometastasis'
        label = 1
    elif 'itc' in status:
        wsi_status = 'itc'

    if label==0 and deposit_class in ln_seg_mask:
        deposit_mask = (ln_seg_mask == deposit_class).astype(np.uint8)
        fat_mask = (ln_seg_mask == fat_class).astype(np.uint8) + (ln_seg_mask == ln_class).astype(np.uint8)

        # Get deposit boundary using morphological gradient
        kernel = np.ones((3, 3), np.uint8)
        deposit_boundary = cv2.morphologyEx(deposit_mask, cv2.MORPH_GRADIENT, kernel)

        # Optionally dilate fat to allow slight gaps
        fat_mask_dilated = cv2.dilate(fat_mask, kernel, iterations=1)

        # Check how many deposit boundary pixels touch fat
        boundary_contact = fat_mask_dilated[deposit_boundary > 0]
        touching_pixels = np.count_nonzero(boundary_contact)
        total_boundary_pixels = np.count_nonzero(deposit_boundary)

        # Sanity check to avoid divide by zero
        if total_boundary_pixels == 0:
            fat_touch_fraction = 0.0
        else:
            fat_touch_fraction = touching_pixels / total_boundary_pixels

        # Pass if ≥50% of deposit boundary touches fat
        if fat_touch_fraction >= 0.5:
            # resize deposit mask to the size of the mask
            deposit_mask = cv2.resize(deposit_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours((deposit_mask==1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # compute area of each contour
            for contour in contours:
                major_axis_td = get_contour_major_axis(contour) * resolution
                if major_axis_td > 3000:
                    label = 1
                    wsi_status = 'tumor_deposit'
                    major_axis = major_axis_td
                    break

    if label==0 and mucin_class in ln_seg_mask:
        mucin_mask = (ln_seg_mask == mucin_class).astype(np.uint8)
        ln_mask = (ln_seg_mask == ln_class).astype(np.uint8) + mucin_mask
        num_labels, labeled_lns = cv2.connectedComponents(ln_mask)
        for i in range(1, num_labels):
            # get the mask of the lymph node
            lymph_node_mask = (labeled_lns == i).astype(np.uint8)
            # get the mask of the mucin
            mucin_mask_lymph_node = mucin_mask * lymph_node_mask
            # check if there is any mucin in the lymph node
            if cv2.countNonZero(mucin_mask_lymph_node)  / cv2.countNonZero(lymph_node_mask) > 0.1:
                major_axis = 0.
                wsi_status = 'acellular_mucin'
                label = 1
                break
    return wsi_status, label, major_axis
    


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


def nested_cpu(tensors:Union[torch.Tensor, list, tuple, Mapping])->Union[torch.Tensor, list, tuple, Mapping]:
    """This function moves all tensors in a nested structure to the CPU

    Args:
        tensors (Union[torch.Tensor, list, tuple, Mapping]): tensors to move to the CPU

    Returns:
        tensors (Union[torch.Tensor, list, tuple, Mapping]): tensors moved to the CPU
    """
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_cpu(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_cpu(t) for k, t in tensors.items()})
    elif isinstance(tensors, torch.Tensor):
        return tensors.cpu().detach()
    else:
        return tensors
    
def divide_list_slurm_array(lst:list) -> list:
    """Used to divide a list into sublists for use with SLURM array jobs. 
    It outputs the sublist that corresponds to the current job.

    Args:
        lst (list): List to divide

    Returns:
        list: A sublist of the input list
    """

    slurm_array_task_count = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
    slurm_array_job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    division_size = len(lst) // slurm_array_task_count
    divisions = [lst[i * division_size:(i + 1) * division_size] for i in range(slurm_array_task_count - 1)]
    divisions.append(lst[(slurm_array_task_count - 1) * division_size:])
    return divisions[slurm_array_job_id]


def get_file_list(path:str, file_type:str)->list:
    """This function returns a list of files with a specified file type in a directory. It searches in: path/**/*file_type

    Args:
        path (str): directory
        file_type (str): file type

    Returns:
        list: list of files with the specified file type
    """
    return glob(os.path.join(path, '**', f'*.{file_type}'), recursive=True)
    
def save_geojson_annotation(out_path:str, mask:np.ndarray, level:int, level_downsampling:int, category_dict:dict, overlapping_class_mask:np.ndarray=None, min_overlapping_object_size:float=None, overlapping_class_name:str=None):
    geojson = {
    'type': 'FeatureCollection',
    'features': []
    }

    # check category dict is valid, values should be unique. If a value is duplicate for two or more categories, keep the first one
    # remove duplicates
    category_dict_unique = {}
    for key, value in category_dict.items():
        if value not in category_dict_unique.values():
            category_dict_unique[key] = value

    for category, id in category_dict_unique.items():
        if id in mask:
            if id == 0:
                continue
            category_mask = (mask == id).astype(np.uint8)
            geojson['features'].extend(encode_contour_json(category_mask, level_downsampling, category)['features'])

    if overlapping_class_mask is not None:
        assert overlapping_class_name is not None, 'overlapping_class_name should be provided if overlapping_class_mask is provided'
        geojson['features'].extend(encode_contour_json(overlapping_class_mask, level_downsampling, overlapping_class_name, min_overlapping_object_size)['features'])

    geojson['mask_shape'] = [mask.shape[0], mask.shape[1]]
    geojson['level'] = level
    geojson['level_downsampling'] = level_downsampling
    geojson['category_dict'] = category_dict
    with open(out_path, 'w') as f:
        json.dump(geojson, f)

def save_sparse_annotation(out_path:str, mask:np.ndarray, level:int, downsample_factor:int, read_origin:tuple):
    non_zero_coords, non_zero_values, shape = sparse_encode(mask)
        
    np.savez_compressed(
        out_path, 
        coords=non_zero_coords, 
        values=non_zero_values, 
        shape=np.array(shape),
        downsample_factor_level = np.array((downsample_factor, level)), 
        read_origin=np.array(read_origin)
        )
    
def save_npy_mask(out_path, mask):
    np.save(out_path, mask)



def save_overlay(out_path:str, wsi_path:str, mask:np.ndarray, level:int, level_downsampling:int, read_origin:tuple, label2id:dict,downsizing_factor:int=1, colormap:dict=COLORMAP, ):
    # get level from mpp
    slide = openslide.open_slide(wsi_path)
    # get dimensions of H&E
    w, h = slide.properties[openslide.PROPERTY_NAME_BOUNDS_WIDTH], slide.properties[openslide.PROPERTY_NAME_BOUNDS_HEIGHT]
    newDim = (int(int(w)/level_downsampling),int(int(h)/level_downsampling))

    slide = np.array(slide.read_region(read_origin, level, newDim))  
    slide[slide[:,:,3] == 0] = 255
    slide = cv2.cvtColor(slide, cv2.COLOR_RGBA2RGB)
    if downsizing_factor > 1:
        slide = cv2.resize(slide, (slide.shape[1]//downsizing_factor, slide.shape[0]//downsizing_factor), interpolation=cv2.INTER_NEAREST)
    # if different shape, resize mask
    if mask.shape != slide.shape[:2]:
        mask = cv2.resize(mask, (slide.shape[1], slide.shape[0]), interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((5,5), np.uint8)  # Define structuring element
    # Closing (fills small holes and smooths edges)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # create contours (with holes)
    # for label, id in label2id.items():
    #     if id not in mask:
    #         continue
    #     aux_mask = (mask == id).astype(np.uint8)
    #     # erode before finding contours to avoid overlapping contours
    #     aux_mask = cv2.erode(aux_mask, kernel, iterations=2)
    #     contours, __ = cv2.findContours(aux_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #     # draw contours
    #     for contour in contours:
    #         cv2.drawContours(slide, contour, -1, colormap[label], thickness=4)
    
    # slide = cv2.cvtColor(slide, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(out_path, slide)

    # now create png for the mask
    id2label = {v: k for k, v in label2id.items()}
    out_path = out_path.replace('.png', '_mask.png')
    png = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    present = []
    for i in id2label.keys():
        if i in mask:
            png[mask == i] = colormap[id2label[i]]
            present.append(i)
    # create figure
    patches = [mpatches.Patch(color=np.array(colormap[id2label[i]])/255, label=id2label[i]) for i in id2label.keys() if i in present]
    plt.imshow(png)
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0) 
    plt.close()     

def function_wo_output(*args):
    pass
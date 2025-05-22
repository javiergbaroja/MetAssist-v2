import openslide
import os
import cv2
from warnings import warn
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Union
from scipy.ndimage import binary_fill_holes

from utils.utils import detect_colors


def prepare_read_from_slide(slide: Union[openslide.OpenSlide, str], resolution: float, file_type:str) -> Tuple[int, int,float, Tuple[int, int], Tuple[int, int]]:
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

    available_mpp = [0.5*(float(slide.properties[openslide.PROPERTY_NAME_MPP_X]) + float(slide.properties[openslide.PROPERTY_NAME_MPP_X])) * i for i in slide.level_downsamples]
    closest_level = np.argmin([np.abs(resolution - available_mpp[i]) for i in range(len(available_mpp))])
    level_downsampling = slide.level_downsamples[closest_level]
    reading_resolution = available_mpp[closest_level]   
    downsample = round(np.log2(resolution / reading_resolution))  

    if downsample < 0:
            raise ValueError(f"Resolution {resolution} is too high for the slide.")
    elif downsample > 0:
        warn(f"Chosen resolution not among available resolutions for the slide. Using closest available resolution {available_mpp[closest_level]} instead.")

    if file_type in ['.svs', '.tif', '.tiff', '.tif', '.ndpi']:
        original_dim = slide.level_dimensions[closest_level][1], slide.level_dimensions[closest_level][0]
        read_origin = (0, 0) 

    elif file_type == '.mrxs':
        # get dimensions of H&E
        x, y = int(slide.properties[openslide.PROPERTY_NAME_BOUNDS_X]), int(slide.properties[openslide.PROPERTY_NAME_BOUNDS_Y])
        w, h = int(slide.properties[openslide.PROPERTY_NAME_BOUNDS_WIDTH]), int(slide.properties[openslide.PROPERTY_NAME_BOUNDS_HEIGHT])
        original_dim = int(h//level_downsampling), int(w//level_downsampling)
        read_origin = (x, y)
    else:
        raise ValueError(f"Unsupported file type {file_type}.")

    return int(closest_level), int(level_downsampling), reading_resolution, original_dim, read_origin

def extract_tile_coords(slide:openslide.OpenSlide, filter_mask: Optional[np.ndarray], img_dim: Tuple[int, int], img_dim_padded: Tuple[int, int],  tile_size: int, step_size: int, read_origin: Tuple[int, int], level: int, level_downsampling:int, rm_background:bool=True) -> List[Tuple[int, int, int, int]]:
    """
    Extract coordinates for tiles from a whole slide image (WSI).

    This function extracts coordinates for tiles from a WSI, optionally using a filter mask to include only specific regions. It also discards tiles that are completely white or black, or do not HE colors.

    Args:
        slide (openslide.OpenSlide): The WSI object.
        filter_mask (Optional[np.ndarray]): A binary mask to filter regions of interest. If None, all regions are included.
        img_dim (Tuple[int, int]): The original shape of the WSI.
        tile_size (int): The size of each tile to extract.
        step_size (int): The step size for moving the tile extraction window.
        read_origin (Tuple[int, int]): The origin coordinates of the WSI read.
        level (int): The level of the WSI to use for extraction.
        level_downsampling (int): The downsampling to reach the chosen level.

    Returns:
        coords (List[Tuple[int, int, int, int]]): A list of coordinates for the extracted tiles.
    """

    if filter_mask is not None:
        filter_mask = filter_mask.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        filter_mask = cv2.morphologyEx(filter_mask, cv2.MORPH_CLOSE, kernel)
        filter_mask = cv2.morphologyEx(filter_mask, cv2.MORPH_OPEN, kernel)
        filter_mask = cv2.dilate(filter_mask, kernel, iterations=1)
        filter_mask = cv2.resize(filter_mask.astype(np.uint8), (img_dim[1], img_dim[0]), interpolation=cv2.INTER_NEAREST)
        # add padding to the filter mask as difference in size between original and padded image. pad with numpy
        pad_x = img_dim_padded[0] - img_dim[0]
        pad_y = img_dim_padded[1] - img_dim[1]
        filter_mask = np.pad(filter_mask, ((0, pad_x), (0, pad_y)), mode='constant', constant_values=1)
    else:
        filter_mask = np.ones(img_dim_padded, dtype=np.uint8)
    coords = []
    bco_filtermask = 0
    bco_colors = 0
    for x in range(0, img_dim_padded[0] - tile_size + 1, step_size):
        for y in range(0, img_dim_padded[1] - tile_size + 1, step_size):
            x_end = x + tile_size
            y_end = y + tile_size
            if x_end > img_dim[0]:
                x_end = img_dim[0]
            if y_end > img_dim[1]:
                y_end = img_dim[1]

            newLocation = (int(int(read_origin[0]) + y * level_downsampling), int(int(read_origin[1]) + x * level_downsampling))
            tile = np.array(slide.read_region(newLocation, level, ((y_end - y), (x_end - x))))
            tile[tile[:, :, 3] == 0] = 255
            tile = cv2.cvtColor(tile, cv2.COLOR_RGBA2RGB)

            # save as png
            # Image.fromarray(tile).save(f'/storage/research/igmp_slide_workspace/GRP Zlobec/Amjad/qupath/metassist-v1/MetAssist_expansion/crc-ugi/results/debug/tiles/tile_{x}_{y}.png')
            # divide tile in quadrants and check if any of them has colors. For the ones that don't, turn into white
            h, w, _ = tile.shape
            half_h, half_w = h // 2, w // 2
            quadrants = [
                tile[0:half_h, 0:half_w],  # Top-left
                tile[0:half_h, half_w:w],  # Top-right
                tile[half_h:h, 0:half_w],  # Bottom-left
                tile[half_h:h, half_w:w]   # Bottom-right
            ]
            # check for colors in each quadrant
            try:
                has_colors_q = [detect_colors(q) for q in quadrants]   
            except Exception as e:
                print(f"Error in detect_colors: {e}")  

            if sum(has_colors_q) > 2:
            # if any(has_colors_q):
                # replace for white the quadrants that don't have colors
                for i, q in enumerate(quadrants):
                    if not has_colors_q[i]:
                        quadrants[i] = 255 * np.ones_like(q, dtype=np.uint8)
                tile = np.vstack([np.hstack([quadrants[0], quadrants[1]]), np.hstack([quadrants[2], quadrants[3]])])
            else:
                bco_colors += 1
                continue

            has_colors = detect_colors(tile) if rm_background else True
            if has_colors:
                if np.any(filter_mask[x:x_end, y:y_end]):
                    coords.append([x, x_end, y, y_end])
                else:
                    bco_filtermask += 1
            else:
                bco_colors += 1
    
    return coords

# np.save('/storage/research/igmp_slide_workspace/GRP Zlobec/Amjad/qupath/metassist-v1/MetAssist_expansion/crc-ugi/results/debug/coords_new_old.npy', np.array(coords))
def calc_expected_max_overlap_num(coords: List[Tuple[int, int, int, int]], padded_shape: Tuple[int, int], crop_size: int) -> int:
    count_map = np.zeros(padded_shape, dtype=np.uint8)
    for (row_start, row_end, col_start, col_end) in coords:
        if row_start != 0:
            row_start += crop_size
        if row_end != padded_shape[0]:
            row_end -= crop_size
        if col_start != 0:
            col_start += crop_size
        if col_end != padded_shape[1]:
            col_end -= crop_size
        count_map[row_start:row_end, col_start:col_end] += 1
    return int(count_map.max())

def post_process(segmentation_mask:np.ndarray, 
                 lymph_node_class:int=1, 
                 classes_to_merge:list=None, 
                 merge_thresholds:list=[0.1], 
                 erase_thresholds:list=[0.01], 
                 apply_opening:list=[False],
                 min_ln_area:int=None,
                 complexity_threshold:float=2.9) -> np.ndarray:
    
    def remove_noise(labeled_ln_mask:np.ndarray, ln_mask:np.ndarray):
        kernel = np.ones((5, 5), np.uint8)
        num_labels = np.unique(labeled_ln_mask)
        for i in num_labels[1:]:
            # create mask for current label
            current_label_mask = (labeled_ln_mask == i).astype(np.uint8)
            box = cv2.boundingRect(current_label_mask)
            x, y, w, h = box
            box = current_label_mask[y:y+h, x:x+w]
            # apply two morphological openings
            box = cv2.morphologyEx(box, cv2.MORPH_OPEN, kernel, iterations=4)
            if box.sum()==0:
                ln_mask[current_label_mask==1]=0
                labeled_ln_mask[current_label_mask==1]=0

        return ln_mask, labeled_ln_mask

    def fill_holes(aux_mask, lns, filled_lns, class_to_merge, ln_class):
            class_to_merge_mask = (aux_mask == class_to_merge).astype(np.uint8)
            # Step 1: class completely inside ln, merge into ln
            where = np.where((filled_lns == 1) & (aux_mask == class_to_merge))
            aux_mask[where] = ln_class
            class_to_merge_mask[where] = 0
            # Step 2: ln completely inside class, merge into class
            box = cv2.boundingRect(class_to_merge_mask)
            class_to_merge_mask_box = class_to_merge_mask[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
            aux_max_box = aux_mask[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
            filled_class_to_merge = binary_fill_holes(class_to_merge_mask_box).astype(np.uint8)
            where = np.where((filled_class_to_merge == 1) & (aux_max_box == ln_class))
            aux_mask[box[1]:box[1]+box[3], box[0]:box[0]+box[2]][where] = class_to_merge
            class_to_merge_mask[box[1]:box[1]+box[3], box[0]:box[0]+box[2]][where] = 1
            lns[box[1]:box[1]+box[3], box[0]:box[0]+box[2]][where] = 0
            return aux_mask
    
    def open_mask(mask: np.ndarray, ln_mask: np.ndarray, min_area: int) -> tuple[np.ndarray, np.ndarray]:
        # Ensure binary masks
        mask = (mask > 0).astype(np.uint8)
        ln_mask = cv2.dilate(ln_mask, np.ones((5, 5), np.uint8), iterations=1)
        #apply dilation

        # Output for lost contact
        lost_ln_contact = np.zeros_like(mask, dtype=np.uint8)

        # Connected component labeling
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        kernel = np.ones((5, 5), np.uint8)
        for label_id in range(1, num_labels):  # skip background
            area = stats[label_id, cv2.CC_STAT_AREA]
            x, y, w, h = stats[label_id, cv2.CC_STAT_LEFT:cv2.CC_STAT_HEIGHT+1]

            roi = (labels[y:y+h, x:x+w] == label_id).astype(np.uint8)
            ln_roi = ln_mask[y:y+h, x:x+w]

            # Check initial LN contact
            if np.any(roi & ln_roi):
                if area >= min_area:
                    roi_opened = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=25)

                    # Check post-opening LN contact
                    if np.any((roi_opened > 0) & ln_roi):
                        mask[y:y+h, x:x+w] = (roi_opened > 0).astype(np.uint8)
                    else:
                        # Keep original region in lost_ln_contact
                        lost_ln_contact[y:y+h, x:x+w] |= roi
                else:
                    # Object too small — leave untouched
                    continue
            else:
                # No initial LN contact — skip
                continue

        return mask, lost_ln_contact
    
    def smart_open_preserve_small_objects(orig_mask: np.ndarray, kernel_size=5, iterations=1, min_size=100) -> np.ndarray:
        # Ensure binary mask
        orig_mask = (orig_mask > 0).astype(np.uint8)

        # Step 1: Apply elliptical opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        opened = cv2.morphologyEx(orig_mask, cv2.MORPH_OPEN, kernel, iterations=iterations)

        # Step 2: Label both original and opened masks
        num_labels, labels_orig = cv2.connectedComponents(orig_mask)
        # _, labels_opened = cv2.connectedComponents(opened)

        # Step 3: Recover objects from original mask that disappeared in opened

        for label_id in range(1, num_labels):  # skip background
            obj_mask = (labels_orig == label_id).astype(np.uint8)

            # If this object is not in opened (all pixels removed), add it back
            if np.sum(opened * obj_mask) == 0:
                area = cv2.countNonZero(obj_mask)
                if area > min_size:
                    opened = cv2.bitwise_or(opened, obj_mask)

        return opened

    assert isinstance(segmentation_mask, np.ndarray), "segmentation_mask must be a numpy array" 
    assert isinstance(lymph_node_class, int), "lymph_node_class must be an integer"
    assert isinstance(classes_to_merge, list), "classes_to_merge must be a list of integers"
    assert isinstance(merge_thresholds, list), "merge_thresholds must be a list of floats"
    assert isinstance(erase_thresholds, list), "erase_thresholds must be a list of floats"
    assert len(classes_to_merge) == len(merge_thresholds), "classes_to_merge and merge_thresholds must have the same length"
    assert len(classes_to_merge) == len(erase_thresholds), "classes_to_merge and erase_thresholds must have the same length"
    assert len(classes_to_merge) == len(apply_opening), "classes_to_merge and apply_opening must have the same length"
    
    classes_to_consider = [lymph_node_class] + classes_to_merge
    # HYPERPARAMETERS
    ln_mask = np.isin(segmentation_mask, classes_to_consider).astype(np.uint8) # create mask for lymph node
    if not np.any(ln_mask):
        return segmentation_mask


    # we will not make changes outside what is ln_mask, so we can create a smaller box 
    x1_original, y1_original, w_original, h_original = cv2.boundingRect(ln_mask)
    aux_mask = segmentation_mask[y1_original:y1_original+h_original, x1_original:x1_original+w_original]    

    lns = (aux_mask == lymph_node_class).astype(np.uint8) 
    num_labels, __ = cv2.connectedComponents(lns)
    lns, __ = remove_noise(__, lns)
    where = np.where((aux_mask == lymph_node_class) & (lns == 0))
    aux_mask[where] = 0

    filled_lns = binary_fill_holes(lns).astype(np.uint8)
    for class_to_merge in classes_to_merge:
        # Step 1: class completely inside ln, merge into ln or ln completely inside class, merge into class
        aux_mask = fill_holes(aux_mask, lns, filled_lns, class_to_merge, lymph_node_class)

    # Create binary mask for the class
    binary = (aux_mask == lymph_node_class).astype(np.uint8)

    # Morphological opening
    opened = smart_open_preserve_small_objects(binary, kernel_size=5, iterations=15)

    # Zero out original class in aux_mask
    aux_mask[aux_mask == lymph_node_class] = 0

    # Restore class label only where opened > 0
    aux_mask[opened > 0] = lymph_node_class
    different = True
    done = 0
    while different and done < 5:
        done += 1
        old = aux_mask.copy()

        for i, class_to_merge in enumerate(classes_to_merge):
            if class_to_merge not in aux_mask:
                continue

            class_to_merge_mask = (aux_mask == class_to_merge).astype(np.uint8)
            # apply opening to separate almost separate objects
            if done==1 :
                if apply_opening[i]:
                    class_to_merge_mask, lost_contact = open_mask(class_to_merge_mask, (aux_mask==lymph_node_class).astype(np.uint8), min_ln_area*50)
                else: 
                    lost_contact = np.zeros_like(class_to_merge_mask, dtype=np.uint8)
                if np.any(lost_contact): print('Opening removed contact with LN')
            num_class_to_merge, labeled_class_to_merge = cv2.connectedComponents(class_to_merge_mask-lost_contact)

            for r in range(1, num_class_to_merge):
                lns = (aux_mask == lymph_node_class).astype(np.uint8)
                dilated_lns = cv2.dilate(lns, np.ones((5, 5), np.uint8), iterations=1)
                __, labeled_dilated_lns = cv2.connectedComponents(dilated_lns)
                labeled_lns = lns * labeled_dilated_lns
                class_to_merge_mask = (labeled_class_to_merge == r).astype(np.uint8)
                class_to_merge_mask[aux_mask != class_to_merge] = 0
                if not np.any(class_to_merge_mask):
                    continue
                # ln ids with contact with class_to_merge
                ln_ids = np.unique(labeled_dilated_lns[class_to_merge_mask == 1])[1:]
                if len(ln_ids) == 0:
                    continue
                
                combined_lns = np.isin(labeled_lns, ln_ids).astype(np.uint8)
                combined_mask = cv2.bitwise_or(combined_lns, class_to_merge_mask)
                box = cv2.boundingRect(combined_mask)
                
                combined_mask = combined_mask[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                combined_lns = combined_lns[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                class_to_merge_mask = class_to_merge_mask[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                # calculate areas
                combined_area = cv2.countNonZero(combined_mask)
                ln_area = cv2.countNonZero(combined_lns)
                ratio = ln_area / combined_area

                if ratio > merge_thresholds[i]:
                    aux_mask[box[1]:box[1]+box[3], box[0]:box[0]+box[2]][class_to_merge_mask == 1] = lymph_node_class
                elif ratio < erase_thresholds[i]:
                    aux_mask[box[1]:box[1]+box[3], box[0]:box[0]+box[2]][combined_lns == 1] = class_to_merge
                elif ratio > merge_thresholds[i]*0.4:
                    box_circle = cv2.boundingRect(combined_lns)
                    radius = (1.2* min(box_circle[2], box_circle[3])) // 2 
                    # calculate center with moments
                    M = cv2.moments(combined_lns)
                    if M["m00"] == 0:
                        continue
                    cX_ln = int(M["m10"] / M["m00"])
                    cY_ln = int(M["m01"] / M["m00"])

                    # create circular mask
                    circular_mask = np.zeros_like(combined_lns)
                    # calculate intersection with combined_mask
                    cv2.circle(circular_mask, (cX_ln, cY_ln), int(radius), 1, -1)

                    intersection = circular_mask & class_to_merge_mask
                    aux_mask[box[1]:box[1]+box[3], box[0]:box[0]+box[2]][intersection == 1] = lymph_node_class

                else:
                    n_indv_ln, labeled_indv_lns = cv2.connectedComponents(combined_lns)
                    
                    for j in range(1, n_indv_ln):
                        indv_ln = (labeled_indv_lns == j).astype(np.uint8)
                        combined_indiv_mask = cv2.bitwise_or(indv_ln, class_to_merge_mask)
                        area_ln = cv2.countNonZero(indv_ln)
                        area_combined = cv2.countNonZero(combined_indiv_mask)
                        ratio = area_ln / area_combined
                        if ratio < erase_thresholds[i]*0.5:
                            aux_mask[box[1]:box[1]+box[3], box[0]:box[0]+box[2]][indv_ln == 1] = class_to_merge
                        
        if not np.any(aux_mask != old):
            different = False
        
    # 1. Identify lymph node regions
    lns = (aux_mask == lymph_node_class).astype(np.uint8)

    # 2. Dilate the lymph node mask
    lns_dilated = cv2.dilate(lns, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=5)

    # 3. Keep only the dilated parts that overlap background
    dilated_into_background = (lns_dilated == 1) & (aux_mask == 0)
    aux_mask[dilated_into_background] = lymph_node_class

    # 4. Recompute the lymph node mask and fill all holes
    lns_filled = binary_fill_holes((aux_mask == lymph_node_class)).astype(np.uint8)

    # 5. Overwrite entire filled region with lymph node class
    aux_mask[lns_filled == 1] = lymph_node_class

    # 6. Remove small objects and complex objects
    num_labels, labeled_lns = cv2.connectedComponents((aux_mask == lymph_node_class).astype(np.uint8))
    for i in range(1, num_labels):
        # create mask for current label
        current_label_mask = (labeled_lns == i).astype(np.uint8)
        # Find contour with full resolution
        contours, _ = cv2.findContours(current_label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue  # skip empty regions
        contour = max(contours, key=cv2.contourArea)

        # Compute area and perimeter
        area = cv2.contourArea(contour)
        if area == 0:
            continue
        perimeter = cv2.arcLength(contour, closed=True)

        # Compute complexity ratio
        expected_circle_perimeter = 2 * np.sqrt(np.pi * area)
        complexity_ratio = perimeter / expected_circle_perimeter
        is_complex = complexity_ratio > complexity_threshold

        if area < min_ln_area:
            aux_mask[current_label_mask == 1] = 0
        elif is_complex:
            aux_mask[current_label_mask == 1] = 2
            print(f'removed complex object, complexity ratio {complexity_ratio:.2f}')

    segmentation_mask[y1_original:y1_original+h_original, x1_original:x1_original+w_original] = aux_mask
    return segmentation_mask


def unpack_coords(coord_list:list, polygon_dict:Dict[str, List[np.ndarray]], category:str, downsampling_level:int, boundaries:tuple=None) -> Dict[str, List[np.ndarray]]:
    for coords in coord_list:
        if len(coords) == 2 and isinstance(coords[0], (int, float, complex)):
            coords = (np.array(coord_list) / downsampling_level).astype(np.int32)
            coords_u = np.unique(coords, axis=0)
            if coords_u.shape[0] > 2:
                if boundaries is not None:
                    x_min, x_max, y_min, y_max = boundaries
                    if np.any((coords_u[:,1] > x_min) & (coords_u[:,1] < x_max)) and np.any((coords_u[:,0] > y_min) & (coords_u[:,0] < y_max)):
                        coords[:,0] = np.clip(coords[:,0], y_min, y_max) - y_min
                        coords[:,1] = np.clip(coords[:,1], x_min, x_max) - x_min
                    else: 
                        break
                polygon_dict[category].append(coords)
            break
        else:
            unpack_coords(coords, polygon_dict, category, downsampling_level, boundaries)
    return polygon_dict
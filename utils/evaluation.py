import numpy as np
import cv2
from typing import Tuple

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
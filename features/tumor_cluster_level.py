import numpy as np
from skimage import measure, morphology
from scipy.spatial import distance
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.ndimage import convolve
import cv2
from typing import Dict, Tuple
from skimage import morphology
from skimage import measure

def extract_shape_features(mask:np.ndarray):
    """Compute shape features for a binary mask with a single metastasis object."""
    area = np.sum(mask)

    perimeter = np.sum(measure.perimeter(mask, neighbourhood=8))
    
    region = measure.regionprops(mask.astype(int))[0]
    solidity = region.solidity
    eccentricity = region.eccentricity
    
    # Boundary irregularity (roughness index)
    roughness = perimeter ** 2 / (4 * np.pi * area) if area > 0 else 0
    
    return {
        "area": area,
        "perimeter": perimeter,
        "solidity": solidity,
        "eccentricity": eccentricity,
        "roughness": roughness,
    }

def detect_branching_points(skeleton):
    # Define a 3x3 neighborhood kernel to count neighbors
    kernel = np.array([[1, 1, 1], 
                       [1, 10, 1],  # The center is set to 10 to preserve it
                       [1, 1, 1]])
    
    # Convolve the kernel with the binary skeleton
    neighbor_count = convolve(skeleton, kernel, mode='constant', cval=0)
    
    # Branching points: pixels with more than 2 neighbors (values 13+ in the convolved result)
    branching_points = (skeleton == 1) & (neighbor_count >= 13)

    
    mask = np.zeros_like(skeleton, dtype=np.uint8)    
    # Set the branching points in the mask to 255
    mask[branching_points > 0] = 255

    kernelSize = 3
    # Set operation iterations:
    opIterations = 4
    # Get the structuring element:
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    # Perform Dilate:
    pointsMask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101)

    return np.max(measure.label(pointsMask))

def extract_skeleton_features(mask, ratio=0.05):
    """Compute graph-based features using the skeleton of the metastasis region."""
    skeleton = morphology.skeletonize(mask).astype(np.uint8)
    points = np.column_stack(np.where(skeleton))
    adjust = False
    # getting the spanning tree is an expensive operation, so if there are many points, we should downsample the mask
    if len(points) > 1000:
        # downsample the mask
        adjust = True
        downsampled_mask = cv2.resize(mask.astype(np.uint8), (0, 0), fx=ratio, fy=ratio)
        skeleton = morphology.skeletonize(downsampled_mask).astype(np.uint8)
        points = np.column_stack(np.where(skeleton))
    
    if len(points) < 2:
        return {"mst_length": 0, "num_branches": 0, "num_points": 0}
    
    dist_matrix = distance.cdist(points, points, metric='euclidean')
    mst = minimum_spanning_tree(dist_matrix)
    mst_length = mst.sum()
    
    return {
        "mst_length": mst_length if not adjust else mst_length / ratio,
        "num_points": len(points) if not adjust else len(points) / ratio ,
        "num_branches": detect_branching_points(skeleton),
    }

def extract_boundary_irregularity(mask) -> float:
    """Compute fractal dimension to quantify boundary irregularity."""
    perimeter = np.sum(measure.perimeter(mask, neighbourhood=8))
    area = np.sum(mask)
    
    return np.log(perimeter) / np.log(area) if area > 1 else 0



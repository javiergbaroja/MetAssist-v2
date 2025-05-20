import openslide
import os
import pathlib
import sys
from itertools import combinations
from typing import List, Tuple
import argparse
from tqdm import tqdm
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.dirname(SCRIPT_DIR))

# Third-party library imports
from natsort import natsorted
import cv2
import geojson
import matplotlib.font_manager as fm
import networkx as nx
import numpy as np
import openslide
from scipy.stats import skew, kurtosis
from scipy.spatial import Delaunay
from scipy.spatial.distance import directed_hausdorff
from skimage import measure
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity
from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform, warp

# Project-specific imports
from features.tumor_cluster_level import extract_skeleton_features
from utils.utils import create_mask_from_contours, get_ann_files, check_wsi_exists_all_formats, divide_list_slurm_array

def hole_fraction(mask: np.ndarray) -> float:
    """
    Calculate the fraction of the object area that is covered by holes.
    
    Parameters:
    - mask (np.ndarray): Binary mask of dtype np.uint8, values 0 or 1, containing one object with possible holes.
    
    Returns:
    - float: Ratio of hole area to total object area (including holes).
    """
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is None:
        return 0.0  # No object found
    
    hierarchy = hierarchy[0]
    
    object_area = 0.0
    hole_area = 0.0

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if hierarchy[i][3] == -1:
            object_area += area  # outer contour
        else:
            hole_area += area  # hole

    return hole_area / object_area if object_area > 0 else 0.0

def create_ann_mask(path_to_geojson:str, level_downsampling:int=None) -> np.ndarray:
    with open(path_to_geojson) as f:
        data = geojson.load(f)

    for feature in data['features']:
        if "classification" not in feature['properties']:
            feature['properties']['classification'] = {'name': feature['properties']['type']}

    
    __, level_downsampling_original = get_level_and_downsampling(path_to_geojson)
    ratio = 1
    if level_downsampling is None:
        level_downsampling = level_downsampling_original
    elif level_downsampling != level_downsampling_original:
        ratio = level_downsampling_original / level_downsampling

    mask_shape = (int(data['mask_shape'][0]*ratio), int(data['mask_shape'][1]*ratio))
    return create_mask_from_contours(data, 
                                     data['category_dict'],
                                     mask_shape,
                                     level_downsampling=level_downsampling)

def compute_euler_number(binary_image: np.ndarray) -> int:
    """
    Computes the Euler number (1 - number_of_holes) for a binary image with a single object.
    
    Args:
        binary_image (np.ndarray): A binary image (0 and 255 or 0 and 1), single object.

    Returns:
        int: Euler number.
    """
    # Ensure binary format (0 and 255)
    bin_img = (binary_image > 0).astype(np.uint8) * 255

    # Find contours and hierarchy
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # No contours found
    if hierarchy is None:
        return 0

    # Hierarchy: [Next, Previous, First_Child, Parent]
    hierarchy = hierarchy[0]
    
    # Count number of holes (contours with a parent)
    num_holes = sum(1 for h in hierarchy if h[3] != -1)

    # Euler number = 1 - number of holes
    return 1 - num_holes

def get_bounding_boxes(ln_data:dict, level_downsampling:int=None) -> List[Tuple[int, int, int, int]]:
    """
    Get the bounding box of the largest area in the mask.
    """
    if level_downsampling is None:
        if 'level_downsampling' in ln_data:
            level_downsampling = ln_data['level_downsampling']
        elif 'level' in ln_data:
            level_downsampling = 2**ln_data['level']
        else:
            raise ValueError('level_downsampling not found in data')
    b_boxes = []
    for feature in ln_data['features']:
        coordinates = feature['geometry']['coordinates']
        if len(coordinates) == 0:
            continue
        elif len(coordinates) >= 1 and isinstance(coordinates[0][0], (int,float)):
            coordinates = [coordinates]

        contour = np.array(coordinates[0], dtype=np.int32) // level_downsampling
        x, y, w, h = cv2.boundingRect(contour)
        b_boxes.append([x, y, w, h])
    return b_boxes


def get_level_and_downsampling(ann_file:str) -> Tuple[int, int]:
    """
    Get the level and downsampling factor from the annotation file.
    """
    with open(ann_file) as f:
        data = geojson.load(f)
    
    if 'level' in data:
        level = data['level']
    elif 'level_downsampling' in data:
        level = int(np.log2(data['level_downsampling']))
    else:
        raise ValueError('level or level_downsampling not found in data')
    
    if 'level_downsampling' in data:
        downsampling = data['level_downsampling']
    elif 'level' in data:
        downsampling = 2**level
    else:
        raise ValueError('level_downsampling not found in data')
    
    return level, downsampling


def get_graph_features(centroids:np.ndarray, distance_threshold:float) -> dict:

    if len(centroids) == 0:
        num_clusters = 0
        inter_cluster_mean = 0
        inter_cluster_std = 0
        cluster_centroids = []
        cluster_variability = []
        hull_vertices = []
        labels = []
        return {
            "num_clusters": 0,
            "inter_cluster_mean": 0,
            "inter_cluster_std": 0,
            "cluster_centroids": [],
            "cluster_variability": [],
            "cluster_hull_vertices": [], 
            "labels": [],
        }
    elif len(centroids) == 1:
        return {
            "num_clusters": 1,
            "inter_cluster_mean": 0,
            "inter_cluster_std": 0,
            "cluster_centroids": centroids,
            "cluster_variability": [(0, 0)],
            "cluster_hull_vertices": [],
            "labels": np.array([0]),
        }
    elif len(centroids) < 4:
        # If there are fewer than 4 points, use a simple distance threshold
        dist_matrix = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=-1)
        np.fill_diagonal(dist_matrix, 0)
        # Create a graph where nodes are centroids
        G = nx.Graph()
        G.add_nodes_from(range(len(centroids)))
        # add edges based on distance threshold
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                if dist_matrix[i, j] <= distance_threshold:
                    G.add_edge(i, j)
    else:       
        # Construct Delaunay triangulation
        tri = Delaunay(centroids)

        # Create a graph where nodes are centroids
        G = nx.Graph()
        G.add_nodes_from(range(len(centroids)))

        # Add edges based on Delaunay triangulation
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    p1, p2 = simplex[i], simplex[j]
                    dist = np.linalg.norm(centroids[p1] - centroids[p2])
                    if dist <= distance_threshold:
                        G.add_edge(p1, p2)

    # Identify connected components as clusters
    clusters = list(nx.connected_components(G))
    num_clusters = len(clusters)
    print(f"Number of clusters detected: {num_clusters}")

    # Assign cluster labels for visualization
    labels = np.zeros(len(centroids), dtype=int)
    cluster_centroids = []
    cluster_variability = []
    for cluster_idx, cluster in enumerate(clusters):
        cluster_points = centroids[list(cluster)]
        cluster_centroid = np.mean(cluster_points, axis=0)
        cluster_centroids.append(cluster_centroid)
        
        # Compute intra-cluster distances
        intra_cluster_distances = np.linalg.norm(cluster_points[:, None] - cluster_points[None, :], axis=-1)
        intra_cluster_distances = intra_cluster_distances[np.triu_indices(len(cluster_points), k=1)]
        mean_intra_cluster_distance = np.mean(intra_cluster_distances) if len(intra_cluster_distances) > 0 else 0
        std_intra_cluster_distance = np.std(intra_cluster_distances) if len(intra_cluster_distances) > 0 else 0
        cluster_variability.append((mean_intra_cluster_distance, std_intra_cluster_distance))
        
        for node in cluster:
            labels[node] = cluster_idx

    # Compute distance matrix between cluster centroids
    cluster_centroids = np.array(cluster_centroids)
    distance_matrix = np.linalg.norm(cluster_centroids[:, None] - cluster_centroids[None, :], axis=-1)

    # Extract inter-cluster distances (excluding diagonal)
    inter_cluster_distances = distance_matrix[np.triu_indices(num_clusters, k=1)] if num_clusters > 1 else []
    inter_cluster_std = np.std(inter_cluster_distances) if len(inter_cluster_distances) > 1 else 0
    inter_cluster_mean = np.mean(inter_cluster_distances) if len(inter_cluster_distances) > 1 else 0
    print(f"Inter-Cluster Distance Mean: {inter_cluster_mean:.4f}")
    print(f"Inter-Cluster Distance Variability (std dev): {inter_cluster_std:.4f}")

    # Print intra-cluster variability
    for i, (mean_dist, std_dist) in enumerate(cluster_variability):
        print(f"Cluster {i}: Mean Intra-Cluster Distance = {mean_dist:.4f}, Std Dev = {std_dist:.4f}, Size = {len(clusters[i])}")
    
    from scipy.spatial import ConvexHull
    hull_vertices = []
    for i in range(num_clusters):
        cluster_points = centroids[labels == i]
        if len(cluster_points) < 3:
            hull_vertices.append([])
            continue
        hull = ConvexHull(cluster_points)
        hull_vertices.append(cluster_points[hull.vertices])

    
    features = {
        "num_clusters": num_clusters,
        "inter_cluster_mean": inter_cluster_mean,
        "inter_cluster_std": inter_cluster_std,
        "cluster_centroids": cluster_centroids,
        "cluster_variability": cluster_variability,
        "cluster_hull_vertices": hull_vertices,
        "labels": labels,
    }
    return features


def get_centroid(binary_mask: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the centroid of the non-zero pixels in a binary mask using OpenCV.

    Args:
        binary_mask (np.ndarray): A binary mask (2D array) where non-zero pixels represent the region of interest.

    Returns:
        Tuple[float, float]: The (x, y) coordinates of the centroid. If no non-zero pixels are found, returns (None, None).
    """
    # Calculate moments of the binary mask
    moments = cv2.moments(binary_mask, binaryImage=True)
    
    # Check if the area (m00) is zero to avoid division by zero
    if moments["m00"] == 0:
        return None, None

    # Calculate the centroid coordinates
    x_centroid = moments["m10"] / moments["m00"]
    y_centroid = moments["m01"] / moments["m00"]

    return x_centroid, y_centroid

def get_complexity_ratio(mask:np.ndarray) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # get only the largest contour
    if len(contours) == 0:
        print("No contours found")
    elif len(contours) > 1:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    contours = contours[0].squeeze()
    perimeter = cv2.arcLength(contours, True)

    return perimeter / (2 * np.sqrt(np.pi * mask.sum()))

def find_closest_point(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Finds the point in array `b` that is closest to point `a` based on Euclidean distance.

    Args:
        a (np.ndarray): A 1D array of shape (2,) representing a single point (x, y).
        b (np.ndarray): A 2D array of shape (n, 2) representing a set of points.

    Returns:
        np.ndarray: The point in `b` that is closest to `a`.
    """
    # Compute the Euclidean distances between `a` and all points in `b`
    distances = np.linalg.norm(b - a, axis=1)
    
    # Find the index of the minimum distance
    closest_index = np.argmin(distances)
    
    # Return the closest point
    return b[closest_index], distances[closest_index]

def get_read_origin_wsi_crop(wsi:openslide.OpenSlide, wsi_format:str) -> Tuple[int, int]:
    if wsi_format == '.mrxs':
        x, y = int(wsi.properties[openslide.PROPERTY_NAME_BOUNDS_X]), int(wsi.properties[openslide.PROPERTY_NAME_BOUNDS_Y])
    elif wsi_format in ['.svs', '.ndpi']:
        x, y = (0, 0)
    else:
        raise ValueError(f'Unknown WSI format {wsi_format}')
    return x, y

def main(args):
    ann_files = get_ann_files(args.path_to_geojson)
    # This shoould be a list of annotations, with format slidename_metastasis.geojson or slidename_ln.geojson.
    # There should therefore be two annotations per slide, one for metastasis and one for lymph node
    # we need to group them by slide name in a list of tuples [(slidename_metastasis.geojson, slidename_ln.geojson), ...]
    ann_pairs = []
    added_ids = []
    for i, ann_file in enumerate(ann_files):
        if i in added_ids:
            continue
        wsi_name = os.path.splitext(os.path.basename(ann_file))[0]
        wsi_name = '_'.join(wsi_name.split('_')[:-1])
        # now find the index of the other annotation file
        idx = [j for j, ann_file2 in enumerate(ann_files) if wsi_name in ann_file2 and i != j]
        if len(idx) == 0:
            print(f'No matching annotation file found for {ann_file}')
            continue
        elif len(idx) > 1:
            print(f'Multiple matching annotation files found for {ann_file}')
            continue
        else:
            met_ann = ann_file if 'metastasis' in ann_file else ann_files[idx[0]]
            ln_ann = ann_file if 'metastasis' not in ann_file else ann_files[idx[0]]
            ann_pairs.append((wsi_name, ln_ann, met_ann))
            added_ids.extend([i, idx[0]])

    assert len(ann_pairs) > 0, 'No annotation files found'
    
    ann_wsi_pairs = []
    for wsi_name, ln_ann, met_ann in ann_pairs:

        if args.wsi_list is not None and wsi_name not in args.wsi_list:
            print(f'WSI for annotation {wsi_name} was not found in provided list')
            continue
        wsi_file_exists, wsi = check_wsi_exists_all_formats(wsi_name, args.wsi_path)
        if wsi_file_exists:
            ann_wsi_pairs.append((wsi, ln_ann, met_ann))
        else:
            print(f'WSI for annotation {wsi_name} was not found')

    # divide list of files to process in parallel
    ann_wsi_pairs = divide_list_slurm_array(ann_wsi_pairs)
    os.makedirs(args.output_dir, exist_ok=True)

    with tqdm(total=len(ann_wsi_pairs)) as pbar:
        for wsi, ln_ann, met_ann in ann_wsi_pairs:
            if os.path.exists(os.path.join(args.output_dir, f'{os.path.splitext(os.path.basename(wsi))[0]}.geojson')):
                print(f'File {os.path.splitext(os.path.basename(wsi))[0]}.geojson already exists, skipping')
                pbar.update(1)
                continue
            try:
                wsi_name = os.path.splitext(os.path.basename(wsi))[0]
                wsi_format = os.path.splitext(os.path.basename(wsi))[1]
                wsi = openslide.open_slide(wsi)
                if args.resolution is None:
                    level, level_downsampling = get_level_and_downsampling(met_ann)
                    resolution = level_downsampling * 0.5 * (float(wsi.properties[openslide.PROPERTY_NAME_MPP_X]) + float(wsi.properties[openslide.PROPERTY_NAME_MPP_Y]))
                else:
                    mpps = [0.5*(float(wsi.properties[openslide.PROPERTY_NAME_MPP_X]) + float(wsi.properties[openslide.PROPERTY_NAME_MPP_X])) * i for i in wsi.level_downsamples]
                    level = int(np.argmin([np.abs(args.resolution - mpps[i]) for i in range(len(mpps))]))
                    level_downsampling = int(wsi.level_downsamples[level])
                resolution = level_downsampling * 0.5 * (float(wsi.properties[openslide.PROPERTY_NAME_MPP_X]) + float(wsi.properties[openslide.PROPERTY_NAME_MPP_Y]))
                
                ln_mask = create_ann_mask(ln_ann, level_downsampling=level_downsampling)
                met_mask = create_ann_mask(met_ann, level_downsampling=level_downsampling)
                
                # get dimensions of H&E
                x_, y_ = get_read_origin_wsi_crop(wsi, wsi_format)
                with open(ln_ann, 'r') as f:
                    bboxes_ln = get_bounding_boxes(geojson.load(f), level_downsampling=level_downsampling)

                if args.identify_sequential_cuts:
                    raise NotImplementedError('Sequential cut identification is not implemented yet')

                feature_dict = {
                    "level_downsampling": level_downsampling,
                    "bboxes": bboxes_ln,
                }
                for i, bbox in enumerate(bboxes_ln):
                    feature_dict[i] = {
                        "met_object_area": [],
                        "met_object_solidity": [],
                        "met_object_eccentricity": [],
                        "met_object_complexity_ratio": [],
                        "met_object_perimeter": [],
                        "met_object_fractal": [],
                        "met_object_euler": [],
                        "met_object_hole_fraction": [],
                        "met_object_mst_length": [],
                        "met_object_num_branches": [],
                        "met_object_num_points": [],
                        "met_object_h_coef_variation": [],
                        "met_object_e_coef_variation": [],
                        "met_object_h_coef_entropy": [],
                        "met_object_e_coef_entropy": [],
                        "met_area_percent": 0,
                        "met_num_objects": 0,
                        "met_rel_dist_centroid": 0,
                        "met_rel_dist_contour": 0,
                        "met_rel_dist_ratio": 0,
                        "met_num_clusters": 0,
                        "met_inter_cluster_dist_mean": 0,
                        "met_inter_cluster_dist_std": 0,
                        "ln_complexity_ratio": 0,
                        "met_cluster_num_objects": [],
                        "met_cluster_intra_cluster_dist_mean": [],
                        "met_cluster_intra_cluster_dist_std": [],
                        "met_cluster_area": [],
                        "met_cluster_eccentricity": [],
                        "met_cluster_complexity_ratio": [],
                        "met_cluster_fractal": [],
                    }
                    x, y, w, h = bbox
                    ln_mask_crop = ln_mask[y:y+h, x:x+w]
                    met_mask_crop = met_mask[y:y+h, x:x+w]
                    wsi_crop = np.array(wsi.read_region((x_+x*level_downsampling, y_+y*level_downsampling), level, (w, h)))
                    wsi_crop[wsi_crop[:,:,3] == 0] = 255 
                    wsi_crop = cv2.cvtColor(wsi_crop, cv2.COLOR_RGBA2RGB)

                    # commence feature extraction
                    contours, __ = cv2.findContours(ln_mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # get only the largest contour
                    if len(contours) == 0:
                        continue
                    elif len(contours) > 1:
                        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
                    contours = contours[0].squeeze()
                    # check min number of points for contour
                    if contours.shape[0] < 3:
                        continue

                    lymph_node_centroid = get_centroid(ln_mask_crop)
                    lymph_node_radius = float(max(w, h) / 2)
                    lymph_node_area = float(ln_mask_crop.sum())

                    met_mask_crop_labeled = measure.label(met_mask_crop)
                    ids = np.unique(met_mask_crop_labeled)[1:]
                    centroids = []
                    if len(ids) == 0:
                        continue
                    min_object_area = ((2.5 / resolution) **2 * np.pi) # we consider single cell needs to have a radius of at least 2.5 um, smaller -> noise
                    for j in ids:
                        region =  measure.regionprops((met_mask_crop_labeled == int(j)).astype(np.uint8))[0]
                        minr, minc, maxr, maxc = region.bbox
                        # extract the region of interest
                        met_roi = met_mask_crop_labeled[minr:maxr, minc:maxc] == j
                        rgb_roi = wsi_crop[minr:maxr, minc:maxc]
                        # calculate the properties of the region
                        
                        met_object_area = int(region.area)
                        if met_object_area < min_object_area:
                            continue
                        met_object_solidity = float(region.solidity)
                        met_object_eccentricity = float(region.eccentricity)
                        met_object_perimeter = float(region.perimeter )
                        met_object_complexity_ratio = float(met_object_perimeter / (2 * np.sqrt(np.pi * met_object_area))) if met_object_area > 0 else 0
                        met_object_fractal = float(np.log(met_object_perimeter) / np.log(met_object_area)) if met_object_area > 0 else 0
                        hole_fraction_value = hole_fraction(met_roi.astype(np.uint8))
                        met_object_euler = compute_euler_number(met_roi.astype(np.uint8))
                        skeleton_feats = extract_skeleton_features(met_roi.astype(np.uint8))
                        met_ojbect_mst_length = float(skeleton_feats['mst_length'])
                        met_object_num_branches = int(skeleton_feats['num_branches'])
                        met_object_num_points = int(skeleton_feats['num_points'])
                        centroids.append(region.centroid)

                        feature_dict[i]["met_object_area"].append(met_object_area)
                        feature_dict[i]["met_object_solidity"].append(met_object_solidity)  
                        feature_dict[i]["met_object_eccentricity"].append(met_object_eccentricity)
                        feature_dict[i]["met_object_complexity_ratio"].append(met_object_complexity_ratio)
                        feature_dict[i]["met_object_perimeter"].append(met_object_perimeter)
                        feature_dict[i]["met_object_fractal"].append(met_object_fractal)
                        feature_dict[i]["met_object_euler"].append(met_object_euler)
                        feature_dict[i]["met_object_mst_length"].append(met_ojbect_mst_length)
                        feature_dict[i]["met_object_num_branches"].append(met_object_num_branches)
                        feature_dict[i]["met_object_num_points"].append(met_object_num_points)

                        hed_roi = rgb2hed(rgb_roi)                    
                        h, e = hed_roi[..., 0][met_roi], hed_roi[..., 1][met_roi]
                        h = rescale_intensity(h, out_range=(0, 255))
                        e = rescale_intensity(e, out_range=(0, 255))

                        H = np.histogram(h, bins=256, range=[0, 255], density=True)[0]
                        E = np.histogram(e, bins=256, range=[0, 255], density=True)[0]

                        h_mean = np.mean(h, axis=0)
                        h_std = np.std(h, axis=0)
                        e_mean = np.mean(e, axis=0)
                        e_std = np.std(e, axis=0)
                        feature_dict[i]["met_object_h_coef_variation"].append(float(h_std / h_mean) if h_mean > 0 else 0)
                        feature_dict[i]["met_object_e_coef_variation"].append(float(e_std / e_mean) if e_mean > 0 else 0)
                        feature_dict[i]["met_object_h_coef_entropy"].append(float(-sum(np.multiply(H,np.log(H+1e-16)))))
                        feature_dict[i]["met_object_e_coef_entropy"].append(float(-sum(np.multiply(E,np.log(E+1e-16)))))
                    if len(centroids) == 0:
                        continue
                    centroids = np.array(centroids, dtype=np.int32)
                    graph_features = get_graph_features(centroids, distance_threshold=args.distance_threshold)
                    point, distance = find_closest_point(lymph_node_centroid, centroids)
                    
                    # LN level features
                    distance_to_contour = []
                    for centroid in centroids:
                        # Calculate the Euclidean distance from the centroid to each point in the LN contour
                        distances = np.linalg.norm(contours - centroid, axis=1)
                        # Find the minimum distance
                        min_distance = np.min(distances)
                        distance_to_contour.append(min_distance)
                    distance_to_contour = np.min(distance_to_contour)

                    feature_dict[i]["met_area_percent"] = met_mask_crop.sum() / lymph_node_area
                    feature_dict[i]["met_num_objects"] = len(centroids)
                    feature_dict[i]["met_rel_dist_centroid"] = float(distance / lymph_node_radius)
                    feature_dict[i]["met_rel_dist_contour"] = float(distance_to_contour / lymph_node_radius)
                    feature_dict[i]["met_rel_dist_ratio"] = float(distance / distance_to_contour) if distance_to_contour > 0 else 100
                    feature_dict[i]["met_num_clusters"] = graph_features["num_clusters"]
                    feature_dict[i]["met_inter_cluster_dist_mean"] = graph_features["inter_cluster_mean"]
                    feature_dict[i]["met_inter_cluster_dist_std"] = graph_features["inter_cluster_std"]
                    feature_dict[i]["ln_complexity_ratio"] = float(get_complexity_ratio(ln_mask_crop))

                    # cluster level features
                    feature_dict[i]["met_cluster_num_objects"] = [len(np.where(graph_features["labels"] == j)[0]) for j in range(graph_features["num_clusters"])]
                    feature_dict[i]["met_cluster_intra_cluster_dist_mean"] = [j[0] for j in graph_features["cluster_variability"]]
                    feature_dict[i]["met_cluster_intra_cluster_dist_std"] = [j[1] for j in graph_features["cluster_variability"]]
                    feature_dict[i]["met_cluster_area"] = [0 for _ in range(graph_features["num_clusters"])]
                    for idx, label in enumerate(graph_features["labels"]):
                        feature_dict[i]["met_cluster_area"][label] += feature_dict[i]["met_object_area"][idx]
                    feature_dict[i]["met_cluster_area"] = [j / lymph_node_area for j in feature_dict[i]["met_cluster_area"]]

                    for hull in graph_features["cluster_hull_vertices"]:
                        if len(hull) < 3:
                            eccentricity, complexity_ratio, fractal = 0, 0, 0
                            continue
                        hull = np.array(hull, dtype=np.int32)
                        # make empty array to enclose the hull
                        minc, minr, maxc, maxr = np.min(hull[:, 0]), np.min(hull[:, 1]), np.max(hull[:, 0]), np.max(hull[:, 1])
                        hull_mask = np.zeros((maxr - minr, maxc - minc), dtype=np.uint8)
                        cv2.fillConvexPoly(hull_mask, hull - np.array([minc, minr]), 1)
                        # calculate shape features
                        feats = measure.regionprops(hull_mask)
                        eccentricity = feats[0].eccentricity
                        complexity_ratio = feats[0].perimeter / (2 * np.sqrt(np.pi * hull_mask.sum()))
                        fractal = np.log(feats[0].perimeter) / np.log(feats[0].area)
                        feature_dict[i]["met_cluster_eccentricity"].append(eccentricity)
                        feature_dict[i]["met_cluster_complexity_ratio"].append(complexity_ratio)
                        feature_dict[i]["met_cluster_fractal"].append(fractal)
                pbar.update(1)

                # save features to file
                for ln in feature_dict.keys():
                    if ln in ['level', 'level_downsampling', 'mask_shape', 'category_dict', 'bboxes']:
                        continue
                    copy_of_dict = feature_dict[ln].copy()
                    for feature in feature_dict[ln].keys():
                        if isinstance(feature_dict[ln][feature], list):
                            if len(feature_dict[ln][feature]) == 0:
                                continue
                            if len(feature_dict[ln][feature]) == 1:
                                copy_of_dict[feature + '_mean'] = feature_dict[ln][feature][0]
                                copy_of_dict[feature + '_std'] = 0
                                copy_of_dict[feature + '_coefvar'] = 0
                                copy_of_dict[feature + '_min'] = feature_dict[ln][feature][0]
                                copy_of_dict[feature + '_max'] = feature_dict[ln][feature][0]
                                copy_of_dict[feature + '_range'] = 0
                                copy_of_dict[feature + '_median'] = feature_dict[ln][feature][0]
                                copy_of_dict[feature + '_IQR'] = 0
                                copy_of_dict[feature + '_kurtosis'] = 0
                                copy_of_dict[feature + '_skew'] = 0
                            else:
                                copy_of_dict[feature + '_mean'] = np.mean(feature_dict[ln][feature])
                                copy_of_dict[feature + '_std'] = np.std(feature_dict[ln][feature])
                                copy_of_dict[feature + '_coefvar'] = copy_of_dict[feature + '_mean'] / copy_of_dict[feature + '_std'] if copy_of_dict[feature + '_std'] > 0 else 0
                                copy_of_dict[feature + '_min'] = min(feature_dict[ln][feature])
                                copy_of_dict[feature + '_max'] = max(feature_dict[ln][feature])
                                copy_of_dict[feature + '_range'] = copy_of_dict[feature + '_max'] - copy_of_dict[feature + '_min']
                                copy_of_dict[feature + '_median'] = np.median(feature_dict[ln][feature])
                                copy_of_dict[feature + '_IQR'] = np.percentile(feature_dict[ln][feature], 75) - np.percentile(feature_dict[ln][feature], 25)
                                copy_of_dict[feature + '_kurtosis'] = kurtosis(feature_dict[ln][feature]) if len(np.unique(feature_dict[ln][feature])) > 1 else 0
                                copy_of_dict[feature + '_skew'] = skew(feature_dict[ln][feature]) if len(np.unique(feature_dict[ln][feature])) > 1 else 0
                    feature_dict[ln] = copy_of_dict

                with open(os.path.join(args.output_dir, f'{wsi_name}.geojson'), 'w') as f:
                    geojson.dump(feature_dict, f)
                f.close()
                # check you can successfully read the file
                with open(os.path.join(args.output_dir, f'{wsi_name}.geojson'), 'r') as f:
                    data = geojson.load(f)
                print(f'Processed {wsi_name} successfully')
            except Exception as e:
                print(f'Error processing {wsi_name}: {e}')
                continue



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Infer mask2former on WSI')
    parser.add_argument('--wsi_path', type=str, help='Path to input WSI folder')
    parser.add_argument('--path_to_geojson', type=str, help='Path to input prediction folder')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--resolution', type=float, help='Resolution', default=None)
    parser.add_argument('--wsi_list', type=str, help='Path to WSI list, a subset of WSIs to process, given as slidea.mrxs\nslideb.mrxs\nslidec.mrxs', default=None)
    parser.add_argument('--distance_threshold', type=float, help='Distance threshold for Delaunay triangulation', default=400)
    parser.add_argument('--identify_sequential_cuts', action='store_true', help='Identify sequential cuts')
    args = parser.parse_args()

    args.wsi_path = args.wsi_path.split(',')

    if args.wsi_list is not None:
        assert os.path.exists(args.wsi_list), f'Provided WSI list {args.wsi_list} does not exist'
        with open(args.wsi_list, 'r') as f:
            args.wsi_list = f.read().split('\n')
            if len(args.wsi_list) > 0:
                args.wsi_list = [os.path.splitext(wsi)[0] for wsi in args.wsi_list]
                # args.wsi_list = [os.path.splitext(os.path.basename(i))[0] for i in args.wsi_list if i != '']
    
    main(args)
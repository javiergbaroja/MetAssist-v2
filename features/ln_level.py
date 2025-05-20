import numpy as np
from skimage import measure
from scipy.spatial import distance
from scipy.cluster.hierarchy import fcluster, linkage
from skimage import morphology
from scipy.ndimage import distance_transform_edt

def compute_spatial_distribution(masks, radius=50):
    """Compute spatial distribution of metastases across the LN."""
    centroids = [measure.regionprops(mask.astype(int))[0].centroid for mask in masks]
    
    if len(centroids) < 2:
        return {"num_regions": 1, "mean_distance": 0, "num_clusters": 1}
    
    dist_matrix = distance.cdist(centroids, centroids)
    num_regions = sum((dist_matrix < radius).sum(axis=1) > 1)
    mean_distance = np.mean(dist_matrix[dist_matrix > 0])

    linkage_matrix = linkage(centroids, method='single')
    cluster_labels = fcluster(linkage_matrix, t=radius, criterion='distance')
    num_clusters = len(set(cluster_labels))
    
    return {"num_regions": num_regions, "mean_distance": mean_distance, "num_clusters": num_clusters}

    
def compute_relative_position_in_ln(masks, ln_mask, radius=50):
    """Compute the relative position of metastases within the lymph node."""
    centroids = [measure.regionprops(mask.astype(int))[0].centroid for mask in masks]
    linkage_matrix = linkage(centroids, method='single')
    cluster_labels = fcluster(linkage_matrix, t=radius, criterion='distance')
    
    ln_contour = morphology.binary_dilation(ln_mask) - ln_mask  # Edge of LN
    ln_distance_map = distance_transform_edt(ln_mask)
    edge_distance_map = distance_transform_edt(ln_contour)
    
    relative_positions = []
    for i, mask in enumerate(masks):
        centroid = measure.regionprops(mask.astype(int))[0].centroid
        distance_to_edge = edge_distance_map[int(centroid[0]), int(centroid[1])]
        distance_to_center = ln_distance_map[int(centroid[0]), int(centroid[1])]
        relative_position = distance_to_edge / (distance_to_edge + distance_to_center)
        relative_positions.append({
            "cluster": cluster_labels[i],
            "relative_position": relative_position
        })
    
    return relative_positions
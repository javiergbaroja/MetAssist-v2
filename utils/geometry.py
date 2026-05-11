# Handles all geometric transformations, contour operations, and mask encoding/decoding.
from __future__ import annotations

from copy import deepcopy
import numpy as np
import cv2
from shapely.geometry import shape, mapping
import json, geojson
from typing import List, Tuple, Dict, Union, Any, Optional

from .visualization import COLORMAP

GeoJSON = Dict[str, Any]

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
                    'color' : list(COLORMAP[category] if category in COLORMAP else (100,100,100)),
                    }
            },
            'geometry': {
                'type': 'Polygon',
                'coordinates': [contour]
            }
        }

        if len(hole) > 0:
            feature['geometry']['coordinates'].extend(hole)
            feature['geometry'] = fix_geometry(feature['geometry'])
    
        output_geojson['features'].append(feature)
    output_geojson = split_qupath_annotations_geojson(output_geojson, id_suffix='', keep_properties=True, normalize_mislabeled_polygons=True)
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


def fix_geometry(polygon: dict) -> dict:
    geom = shape(polygon)
    if not geom.is_valid:
        geom = geom.buffer(0)
        if not geom.is_valid:
            print("Failed to fix geometry")
        polygon = mapping(geom)
        
    return polygon
    

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


def _is_position(x: Any) -> bool:
    # Accept [x, y] or [x, y, z] etc.
    return (
        isinstance(x, list)
        and len(x) >= 2
        and all(isinstance(v, (int, float)) for v in x[:2])
    )


def _looks_like_linear_ring(x: Any) -> bool:
    # A ring is a list of positions
    return isinstance(x, list) and len(x) >= 4 and _is_position(x[0])


def _looks_like_polygon_coords(coords: Any) -> bool:
    # Polygon coords: [ring1, ring2, ...]
    return isinstance(coords, list) and len(coords) >= 1 and _looks_like_linear_ring(coords[0])


def _looks_like_multipolygon_coords(coords: Any) -> bool:
    # MultiPolygon coords: [poly1, poly2, ...] where poly = [ring1, hole1, ...]
    return isinstance(coords, list) and len(coords) >= 1 and _looks_like_polygon_coords(coords[0])


def _normalize_mislabeled_polygon_geometry(geom: GeoJSON) -> GeoJSON:
    """
    If geom is type=Polygon but coordinates are MultiPolygon-shaped, convert to MultiPolygon.
    Otherwise return geom unchanged.
    """
    if not isinstance(geom, dict):
        return geom
    if geom.get("type") != "Polygon":
        return geom

    coords = geom.get("coordinates")
    if _looks_like_multipolygon_coords(coords) and not _looks_like_polygon_coords(coords):
        # The check above is conservative; but in practice a MultiPolygon-shaped
        # coords will also satisfy polygon_coords if you only check the first item.
        # So we do an explicit structural check below.
        pass

    # Structural check:
    # - Polygon: coords[0][0] should be a position [x,y]
    # - Mis-MultiPolygon: coords[0][0][0] should be a position [x,y]
    try:
        if (
            isinstance(coords, list)
            and len(coords) > 0
            and isinstance(coords[0], list)
            and len(coords[0]) > 0
            and isinstance(coords[0][0], list)
            and len(coords[0][0]) > 0
            and _is_position(coords[0][0][0])  # extra nesting indicates multipolygon
            and not _is_position(coords[0][0]) # ensures it's not a normal polygon ring
        ):
            new_geom = deepcopy(geom)
            new_geom["type"] = "MultiPolygon"
            new_geom["coordinates"] = coords
            return new_geom
    except Exception:
        # If anything unexpected happens, just keep original geometry
        return geom

    return geom


def split_qupath_annotations_geojson(
    geojson: GeoJSON,
    *,
    id_suffix: str = "_part",
    keep_properties: bool = True,
    normalize_mislabeled_polygons: bool = True,
) -> GeoJSON:
    """
    Split QuPath-exported GeoJSON annotations that contain multi-part geometries
    into separate single-part Feature objects.

    Additionally, if normalize_mislabeled_polygons=True, convert geometries with:
        type == "Polygon"
        coordinates shaped like MultiPolygon
    into proper MultiPolygon before splitting.
    """
    if not isinstance(geojson, dict) or geojson.get("type") != "FeatureCollection":
        raise ValueError("Expected a GeoJSON FeatureCollection (dict with type='FeatureCollection').")

    features = geojson.get("features")
    if not isinstance(features, list):
        raise ValueError("FeatureCollection must contain a list in 'features'.")

    out_features: List[GeoJSON] = []

    for feat_index, feature in enumerate(features):
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            out_features.append(feature)
            continue

        geom = feature.get("geometry")
        if not isinstance(geom, dict) or "type" not in geom:
            out_features.append(feature)
            continue

        if normalize_mislabeled_polygons:
            geom = _normalize_mislabeled_polygon_geometry(geom)

        gtype = geom.get("type")
        fid: Optional[Any] = feature.get("id", None)

        def new_feature(part_geom: GeoJSON, part_i: int) -> GeoJSON:
            f: GeoJSON = {"type": "Feature", "geometry": part_geom}
            if keep_properties and "properties" in feature:
                f["properties"] = deepcopy(feature["properties"])
            if "bbox" in feature:
                f["bbox"] = deepcopy(feature["bbox"])
            if fid is not None:
                f["id"] = f"{fid}{id_suffix}{part_i}"
            else:
                f["id"] = f"feature{feat_index}{id_suffix}{part_i}"
            return f

        # Split multipart
        if gtype == "MultiPolygon":
            coords = geom.get("coordinates", [])
            if not isinstance(coords, list) or len(coords) <= 1:
                # 0/1 part -> keep original (but use normalized geometry if we changed it)
                kept = deepcopy(feature)
                kept["geometry"] = deepcopy(geom)
                out_features.append(kept)
                continue
            for i, poly_coords in enumerate(coords):
                out_features.append(new_feature({"type": "Polygon", "coordinates": poly_coords}, i))

        elif gtype == "MultiLineString":
            coords = geom.get("coordinates", [])
            if not isinstance(coords, list) or len(coords) <= 1:
                kept = deepcopy(feature)
                kept["geometry"] = deepcopy(geom)
                out_features.append(kept)
                continue
            for i, ls_coords in enumerate(coords):
                out_features.append(new_feature({"type": "LineString", "coordinates": ls_coords}, i))

        elif gtype == "MultiPoint":
            coords = geom.get("coordinates", [])
            if not isinstance(coords, list) or len(coords) <= 1:
                kept = deepcopy(feature)
                kept["geometry"] = deepcopy(geom)
                out_features.append(kept)
                continue
            for i, pt_coords in enumerate(coords):
                out_features.append(new_feature({"type": "Point", "coordinates": pt_coords}, i))

        elif gtype == "GeometryCollection":
            geoms = geom.get("geometries", [])
            if not isinstance(geoms, list) or len(geoms) <= 1:
                kept = deepcopy(feature)
                kept["geometry"] = deepcopy(geom)
                out_features.append(kept)
                continue
            for i, g in enumerate(geoms):
                if isinstance(g, dict) and "type" in g:
                    out_features.append(new_feature(deepcopy(g), i))
                else:
                    out_features.append(feature)
                    break

        else:
            # Single-part -> keep original (but use normalized geometry if we changed it)
            kept = deepcopy(feature)
            kept["geometry"] = deepcopy(geom)
            out_features.append(kept)

    out = deepcopy(geojson)
    out["features"] = out_features
    return out
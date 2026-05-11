
from typing import List, Tuple, Optional, Union
from glob import glob
import os
from utils.wsi import check_wsi_exists_all_formats
from utils.hpc import divide_list_slurm_array

def get_file_list(path:str, file_type:str)->list:
    """This function returns a list of files with a specified file type in a directory. It searches in: path/**/*file_type

    Args:
        path (str): directory
        file_type (str): file type

    Returns:
        list: list of files with the specified file type
    """
    return glob(os.path.join(path, '**', f'*.{file_type}'), recursive=True)


def get_ann_files(path_to_ann:Union[str, List[str]], file_type:str='geojson')->List[str]:
    if isinstance(path_to_ann, list):
        path_to_ann = sorted([p for p in path_to_ann if p.endswith(f'.{file_type}')])
    elif path_to_ann.endswith(f'*.{file_type}'):
        return sorted(glob(path_to_ann))
    elif path_to_ann.endswith(f'.{file_type}'):
        return [path_to_ann] 
    elif os.path.isdir(path_to_ann):
        return get_file_list(path_to_ann, file_type)
    else:
        raise ValueError('Invalid path to annotation file(s). Please provide a directory or a list of files with the correct file type.')
    

def find_wsi(input_dir: str, base: str, file_list:List[str]):
    if file_list is not None and base not in file_list:
        print(f'WSI for annotation {base} was not found in provided list')
        return False, None
    wsi_file_exists, wsi = check_wsi_exists_all_formats(base, input_dir)
    if wsi_file_exists:
        return True, wsi
    else:
        print(f'WSI for annotation {base} was not found')
        return False, None


def get_annotation_pairs(input_dir:str, wsi_dir:str, wsi_list:Optional[List[str]]) -> List[Tuple[str, str, str]]:
    """Find matching LN and metastasis GeoJSON pairs.

    Returns a list of tuples: (wsi_path, ln_geojson_path, met_geojson_path)
    """
    ann_files = get_ann_files(input_dir)
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
            wsi_exists, wsi_path = find_wsi(wsi_dir, wsi_name, wsi_list)
            if not wsi_exists:
                continue
            ann_pairs.append((wsi_path, ln_ann, met_ann))


    assert len(ann_pairs) > 0, 'No annotation files found'
    return divide_list_slurm_array(ann_pairs)
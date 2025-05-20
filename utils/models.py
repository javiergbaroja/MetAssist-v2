import importlib
import torch
from typing import List, Tuple
from torchvision.transforms import functional as F

from utils.augmentations import TestTimeAugmentation
from utils.utils import ACCEPTED_MODEL_CLASSES

def get_model_class_from_checkpoint(checkpoint_path:str) -> str:
    """
    Get the model class from the checkpoint path.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file.
    
    Returns:
        str: The model class name.
    """
    for model_class in ACCEPTED_MODEL_CLASSES:
        if model_class in checkpoint_path:
            return model_class
    # WARNING: No model class found in checkpoint path. Defaulting to 'Mas2FormerForUniversalSegmentation'
    return 'Mask2FormerForUniversalSegmentation'

def get_model_class_from_model(model) -> str:
    """
    Get the model class from the model instance.
    
    Args:
        model: The model instance.
    
    Returns:
        str: The model class name.
    """
    for model_class in ACCEPTED_MODEL_CLASSES:
        if model.__class__.__name__ == model_class:
            return model_class
    raise ValueError(f"Model class {model.__class__.__name__} not in accepted model classes: {ACCEPTED_MODEL_CLASSES}")

def get_model_funcs(model_class:str) -> Tuple:
    """
    Get the model functions based on the model class.

    Args:
        model_class (str): The model class name.

    Returns:
        Tuple: A tuple containing the image processor, model creation function, post-process output function, and custom post-process function.
    """
    if not isinstance(model_class, str):
        model_class = get_model_class_from_model(model_class)
    elif model_class not in ACCEPTED_MODEL_CLASSES:
        raise ValueError(f"Model class {model_class} not in accepted model classes: {ACCEPTED_MODEL_CLASSES}")
    
    module_name = model_class.lower() if model_class != 'Mask2FormerForUniversalSegmentation' else 'mask2former'
    module = importlib.import_module(f"models.{module_name}")
    create_img_processor = getattr(module, 'create_img_processor', None)
    create_model = getattr(module, 'create_model', None)
    post_process_output = getattr(module, 'post_process_output', None)
    custom_post_process_semantic_segmentation = getattr(module, 'custom_post_process_semantic_segmentation', None)
    return create_img_processor, create_model, post_process_output, custom_post_process_semantic_segmentation



def infer_collate_fn(batch: List[Tuple[torch.Tensor, Tuple[int, int, int, int], TestTimeAugmentation]]) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]], List[TestTimeAugmentation]]:

    tiles, coords, augmentations = zip(*batch)
    
    # Stack tiles into a single tensor
    tiles = torch.stack(tiles)
    tiles = F.normalize(tiles, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return tiles, list(coords), list(augmentations)


def create_mask2former_from_checkpoint(checkpoint_path, label2id:dict, encoder_name:str, decoder_model:str, out_indices:list, device:str='cuda' if torch.cuda.is_available() else 'cpu'):
    
    if not isinstance(checkpoint_path, str):
        return checkpoint_path
    
    model_class = get_model_class_from_checkpoint(checkpoint_path)
    _, create_model, _, _ = get_model_funcs(model_class)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = create_model(
        encoder_model=encoder_name, 
        decoder_model=decoder_model, 
        label2id=label2id, 
        id2label={v: k for k, v in label2id.items()}, 
        out_indices=out_indices,
        freeze_encoder=True)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device(device)
    model.to(device)
    return model

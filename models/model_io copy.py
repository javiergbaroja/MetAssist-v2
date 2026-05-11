import importlib
import torch
from typing import List, Tuple
from torchvision.transforms import functional

from data.augmentations import TestTimeAugmentation
ACCEPTED_MODEL_CLASSES = ['UNet', 'SETR', 'UNETR', 'Mask2FormerForUniversalSegmentation', "VISTAPATH"]


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
    # WARNING: No model class found in checkpoint path. Defaulting to 'Mask2FormerForUniversalSegmentation'
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
    module = importlib.import_module(f"models.architectures.{module_name}")
    create_img_processor = getattr(module, 'create_img_processor', None)
    create_model = getattr(module, 'create_model', None)
    # post_process_output = getattr(module, 'post_process_output', None)
    # custom_post_process_semantic_segmentation = getattr(module, 'custom_post_process_semantic_segmentation', None)
    train_collator = getattr(module, 'TrainCollator', None)
    return create_img_processor, create_model, train_collator

class InferCollator:
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        # ImageNet constants
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, batch: List[Tuple[torch.Tensor, Tuple[int, int, int, int], TestTimeAugmentation]]):
        tiles, coords, augmentations = zip(*batch)
        
        # Stack tiles into a single tensor
        tiles = torch.stack(tiles)
        
        if self.normalize:
            tiles = functional.normalize(tiles, mean=self.mean, std=self.std)
        
        return tiles, list(coords), list(augmentations)


def infer_collate_fn(batch: List[Tuple[torch.Tensor, Tuple[int, int, int, int], TestTimeAugmentation]]) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]], List[TestTimeAugmentation]]:

    tiles, coords, augmentations = zip(*batch)
    
    # Stack tiles into a single tensor
    tiles = torch.stack(tiles)
    tiles = functional.normalize(tiles, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return tiles, list(coords), list(augmentations)


def create_mask2former_from_checkpoint(checkpoint_path, label2id:dict, encoder_name:str, decoder_model:str, out_indices:list=[], device:str='cuda' if torch.cuda.is_available() else 'cpu'):
    
    if not isinstance(checkpoint_path, str):
        return checkpoint_path
    
    model_class = get_model_class_from_checkpoint(checkpoint_path)
    _, create_model, _ = get_model_funcs(model_class)
    
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



import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple

def _process_mask2former_output(outputs, target_sizes, return_logits):
    """
    Internal helper to handle Hugging Face Mask2Former outputs.
    Logic extracted from models/mask2former.py
    """
    masks_classes:torch.Tensor = outputs.class_queries_logits
    masks_probs:torch.Tensor = outputs.masks_queries_logits

    # 1. Scale back to preprocessed image size (384, 384)
    masks_probs = F.interpolate(
        masks_probs, size=(384, 384), mode="bilinear", align_corners=False
    )

    # 2. Remove the null class and compute probs
    masks_classes = masks_classes.softmax(dim=-1)[..., :-1]
    masks_probs = masks_probs.sigmoid()

    # 3. Compute semantic segmentation logits via Einstein summation
    segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
    del masks_classes, masks_probs  
    
    # 4. Resize to target sizes
    processed_outputs = []
    if target_sizes is None:
        raise ValueError("Please provide target sizes for resizing the logits")
    else:
        if len(segmentation) != len(target_sizes):
            raise ValueError("Batch size mismatch with target_sizes")
        resized = F.interpolate(segmentation, size=target_sizes[0], mode="bilinear", align_corners=False)  # (B, C, H, W) — no loop needed
        if return_logits:
            return resized
        return resized.argmax(dim=1)
            
    #     for idx in range(len(segmentation)):
    #         resized_logits = torch.nn.functional.interpolate(
    #             segmentation[idx].unsqueeze(0), size=target_sizes[idx], mode="bilinear", align_corners=False
    #         )
    #         if return_logits:
    #             processed_outputs.append(resized_logits[0])
    #         else:
    #             processed_outputs.append(resized_logits[0].argmax(dim=0))
    
    # return torch.stack(processed_outputs)
    



def post_process_output(outputs, 
                        target_sizes: Optional[List[Tuple[int, int]]] = None, 
                        return_logits: bool = False) -> torch.Tensor:
    """
    Unified post-processing function for all models (Mask2Former, UNet, UNETR, SETR).
    
    Args:
        outputs: Raw output from the model (ModelOutput object or Mask2FormerOutput).
        target_sizes: List of (H, W) tuples for resizing (primarily for Mask2Former).
        return_logits: If True, returns (B, C, H, W) logits. If False, returns (B, H, W) class indices.
    
    Returns:
        torch.Tensor: The processed output tensor.
    """
    
    # --- Step 1: Extract Predictions based on Model Type ---
    
    # Case A: Mask2Former (Hugging Face Output)
    if hasattr(outputs, 'class_queries_logits') and hasattr(outputs, 'masks_queries_logits'):
        processed = _process_mask2former_output(outputs, target_sizes, return_logits)
        
    # Case B: Standard Segmentation Models (UNet, UNETR, SETR using ModelOutput)
    elif hasattr(outputs, 'preds'):
        if return_logits:
            processed = outputs.preds
        else:
            processed = outputs.y_pred
            
        # Note: UNet/SETR/UNETR implementations currently ignore target_sizes. 
        # If resizing is needed for them, it should be added here using F.interpolate.
        
    else:
        # Fallback for raw tensors
        processed = outputs

    # --- Step 2: Standardize Shape ---
    
    # Remove unnecessary dimensions (e.g. batch dim of 1)
    processed = processed.squeeze()
    
    # Determine expected dimensions
    # Logits: (B, C, H, W) -> 4 dims
    # Preds:  (B, H, W)    -> 3 dims
    expected_dim = 4 if return_logits else 3
    
    # Restore dimensions if squeeze was too aggressive (e.g. Batch=1 cases)
    # If we have (C, H, W) but want (B, C, H, W), we unsqueeze.
    # If we have (H, W) but want (B, H, W), we unsqueeze.
    while processed.ndim < expected_dim:
        processed = processed.unsqueeze(0)

    # Validate final shape
    assert processed.ndim == expected_dim, f"Output shape mismatch. Expected {expected_dim}D, got {processed.shape}"

    # Ensure float for consistency (UNETR/M2F did this)
    return processed.float()



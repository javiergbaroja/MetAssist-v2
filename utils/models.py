import torch
from typing import List, Tuple
from torchvision.transforms import functional as F

from utils.augmentations import TestTimeAugmentation
from utils.utils import ACCEPTED_MODEL_CLASSES

def infer_collate_fn(batch: List[Tuple[torch.Tensor, Tuple[int, int, int, int], TestTimeAugmentation]]) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]], List[TestTimeAugmentation]]:

    tiles, coords, augmentations = zip(*batch)
    
    # Stack tiles into a single tensor
    tiles = torch.stack(tiles)
    tiles = F.normalize(tiles, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return tiles, list(coords), list(augmentations)


def create_mask2former_from_checkpoint(checkpoint_path, label2id:dict, encoder_name:str, decoder_model:str, out_indices:list, device:str='cuda' if torch.cuda.is_available() else 'cpu'):
    
    if not isinstance(checkpoint_path, str):
        return checkpoint_path
    
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

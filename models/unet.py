from typing import Literal, Dict

from segmentation_models_pytorch.unet import Unet

import torch
import torch.nn as nn
from losses_unet import Mask2FormerStyleLoss, AsymUnifiedFocalLoss
from unetr_base import ModelOutput, BASE_CFG

class UNet(nn.Module):
    def __init__(self, 
                 backbone_name:str,
                 num_classes:int, 
                 freeze_backbone: bool = False,
                 loss_weights_dict:dict = {'asymmetric_ftl':0.5, 'asymmetric_fl':0.5},
                 criterion:Literal['mask2former', 'focaltversky'] = 'focaltversky',
                 **kwargs):
        
        super().__init__()
        self.num_classes = num_classes
        self.criterion = self._get_criterion(criterion, loss_weights_dict)
        self.model = Unet(backbone_name, classes=num_classes, activation=None, encoder_weights=kwargs['encoder_weights'])
        self.head = nn.Identity() if self.criterion.needs_logits else nn.Softmax(dim=1)

        if freeze_backbone:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, x:torch.Tensor, y_true:torch.Tensor=None) -> ModelOutput:
        x = self.head(self.model(x))
        return ModelOutput(preds=x,
                           logits=self.criterion.needs_logits,
                           losses_dict=self.criterion(x, y_true) if y_true is not None else None,
                           loss_weights_dict=self.criterion.loss_weights)


    
    def _get_criterion(self, criterion, loss_weights):
        if criterion == 'mask2former':
            return Mask2FormerStyleLoss(num_classes=self.num_classes, loss_weights=loss_weights)
        elif criterion == 'focaltversky':
            return AsymUnifiedFocalLoss(loss_weights=loss_weights)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        

def create_model(encoder_model:str,  
                 label2id:Dict[str,int], 
                 freeze_encoder:bool,
                 **kwargs) -> UNet:
    
    config = BASE_CFG["_".join([encoder_model, "UNet"])]
    config['num_classes'] = len(label2id)
    config['id2label'] = {v: k for k, v in label2id.items()}
    config['label2id'] = label2id
    config['backbone_name'] = encoder_model 
    config['freeze_backbone'] = freeze_encoder
    config['criterion'] = 'focaltversky'
    config['loss_weights'] = {'asymmetric_ftl':0.5, 'asymmetric_fl':0.5}

    model = UNet(**config)
    
    # Freeze Backbone
    if freeze_encoder: 
        for name, param in model.backbone.named_parameters():
            param.requires_grad = False

    return model

class UNetImageProcessor:
    def __init__(self):
        self.dummy = True

    def __call__(self, data):
        return data


def create_img_processor(decoder_model:str, ignore_index:int=None) -> UNetImageProcessor:

    return UNetImageProcessor()



def custom_post_process_semantic_segmentation(outputs, target_sizes, return_logits:bool=False) -> torch.Tensor:
    """
    Custom post-processing function for semantic segmentation outputs.
    
    Args:
        outputs: The model outputs.
        target_sizes: The target sizes for resizing the outputs.
        return_logits: If True, returns logits. If False, returns probabilities.
    
    Returns:
        A list of dictionaries containing the resized segmentation maps and logits.
    """
    if return_logits:
        return outputs.logits
    else:
        return outputs.y_pred
    
    
def post_process_output(outputs, target_sizes, return_logits=False):
    """
    Post-process the model outputs to create the final segmentation mask.

    Args:
        outputs: Model outputs.
        target_sizes (list): List of target sizes.
        return_logits (bool, optional): Whether to return the logits. Defaults to False.

    Returns:
        torch.Tensor: The final segmentation mask.
    """
    outputs = custom_post_process_semantic_segmentation(outputs, target_sizes=target_sizes, return_logits=return_logits)
    outputs = torch.stack(outputs).squeeze().cpu()
    while len(outputs.shape) < 4:
        outputs = outputs.unsqueeze(0)
    # unsqueeze_first = True if len(outputs.shape) < 4 else False
    # outputs = outputs.unsqueeze(0) if unsqueeze_first else outputs
    assert len(outputs.shape) == 4, f"Expected 4D tensor (BCHW), got {outputs.shape}"
    return outputs.float()
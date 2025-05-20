
from typing import Literal, Union, Dict

from transformers import Dinov2Backbone
import torch
import torch.nn as nn
from losses_unet import Mask2FormerStyleLoss, AsymUnifiedFocalLoss
from unetr_base import BASE_CFG, get_embedding_combiner, ConvBlock, DeconvBlock, ModelOutput, ViTEncoder, foundation_backbones, facebook_backbones 
from timm import create_model

class SETR(nn.Module):
    """
    SEgmentation TRansformer (SETR).

    This class implements a SETR model that uses a ViT as the backbone and a series of deconvolutional layers as the decoder for Progressive UPsampling (SETR-PUP). 
    The output of the network is a segmentation map where each pixel is assigned a class label.

    From Zheng S et al. (2021), Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers, Proceedings of the IEEE/CVF conference on computer vision and pattern recognition

    Parameters
    ----------
    num_classes : int, optional
        The number of output classes for the segmentation task. Default is 1.
    embedding_combiner : str, optional
        The method to combine the embeddings from the ViT backbone. Options are 'linear', 'modulation', 'drop_cls_token'. Default is 'linear'.
    freeze_backbone : bool, optional
        If True, the weights of the ViT backbone will be frozen during training, i.e., they will not be updated. Default is False.
    criterion: str, Literal['mask2former', 'focaltversky'], optional. Default is 'focaltversky'.
    **kwargs
        Additional keyword arguments for the ViT backbone.

    Attributes
    ----------
    backbone : nn.Module
        The ViT backbone.
    embedding_combiner : nn.Module
        The module used to combine the cls and patch tokens from the ViT backbone. Options are 'linear', 'modulation' or 'drop_cls_token'. 
        'linear' uses a linear layer to combine the embeddings, 'modulation' uses a conditional gating module, and 'drop_cls_token' drops the cls token.
    decoder : nn.Module
        The decoder that upsamples the combined embeddings to the original image size.
    segmentation_head : nn.Module
        The final layer that maps the decoder output to class labels.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Performs a forward pass through the network.

    Notes
    -----
    The SETR expects input tensors of shape (B, C, H, W) where B is the batch size, C is the number of channels, and H and W are the height and width of the images, respectively.
    """
    def __init__(self, 
                 num_classes: int = 1,
                 embedding_combiner: Literal['linear', 'modulation', 'drop_cls_token'] = 'drop_cls_token',
                 backbone_name: Literal['dinov2_h_virchow2', 'dinov2_l', 'dinov2-h-virchow2', 'dinov2-l'] = 'dinov2_h_virchow2',
                 freeze_backbone: bool = False,
                 criterion: Literal['mask2former', 'focaltversky'] = 'focaltversky',
                 loss_weights: dict = {'asymmetric_ftl':0.5, 'asymmetric_fl':0.5},
                 **kwargs) -> None:
        super().__init__()


        self.num_classes = num_classes
        self.criterion = self._get_criterion(criterion, loss_weights)
        self.backbone = self._get_backbone(backbone_name, **kwargs)
        # vit produces embeddings of shape (1,257,self.backbone.embed_dim). We need to deal with the cls token
        self.embedding_combiner = get_embedding_combiner(embedding_combiner, self.backbone.embed_dim, backbone_name)

        self.decoder = nn.Sequential(
            nn.Sequential(DeconvBlock(self.backbone.embed_dim, 512)), #32x32
            nn.Sequential(DeconvBlock(512, 256)), #64x64
            nn.Sequential(DeconvBlock(256, 128)), #128x128  
            nn.Sequential(DeconvBlock(128, 64)),  #256x256
            )
        self.segmentation_head = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(64, 32),
            nn.Conv2d(32, self.num_classes, 1),
            nn.Identity() if self.criterion.needs_logits else nn.Softmax(dim=1)
            )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _get_criterion(self, criterion, loss_weights):
        if criterion == 'mask2former':
            return Mask2FormerStyleLoss(num_classes=self.num_classes, loss_weights=loss_weights)
        elif criterion == 'focaltversky':
            return AsymUnifiedFocalLoss(loss_weights=loss_weights)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
    def _get_backbone(self, backbone_name, **kwargs) -> Union[ViTEncoder, Dinov2Backbone]:
        if backbone_name in foundation_backbones:
            backbone = ViTEncoder(num_classes=0, **kwargs)
            backbone.load_state_dict(create_model(kwargs['encoder_weights'], pretrained=True, mlp_layer=kwargs['mlp_layer'], act_layer=kwargs['act_layer'], out_indices=kwargs['feature_layers'], features_only=False).state_dict())
        elif backbone_name in facebook_backbones:
            backbone = Dinov2Backbone.from_pretrained(kwargs['encoder_weights'], out_indices=kwargs['feature_layers'], label2id=kwargs['label2id'], id2label=kwargs['id2label'], num_labels=self.num_classes, ignore_mismatched_sizes=True)
            backbone.embed_dim = backbone.config.hidden_size
            backbone._out_channels = [backbone.embed_dim]*backbone.config.num_hidden_layers
            backbone._in_channels = kwargs['in_chans']
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        return backbone
    
        
    def forward(self, x: torch.Tensor, y_true:torch.Tensor=None) -> ModelOutput:
        B, C, height, width = x.shape

        x = self.backbone(x).feature_maps[0]
        x = self.embedding_combiner(x) # combine cls token and patch tokens

        # reshape to BCHW output format if not already in that format
        if len(x.shape[2:]) < 2: 
            H, W = self.backbone.patch_embed.dynamic_feat_size((height, width))
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.decoder(x)
        # check if matching height and width for concatenation. If not matching crop and then concatenate
        if x.shape[2:] != (height, width):
            diffY = height - x.shape[2]
            diffX = width - x.shape[3]
            x = nn.functional.pad(x, (diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2))
        x = self.segmentation_head(x)

        return ModelOutput(preds=x,
                           logits=self.criterion.needs_logits,
                           losses_dict=self.criterion(x, y_true) if y_true is not None else None,
                           loss_weights_dict=self.criterion.loss_weights)



def create_model(encoder_model:str,  
                 label2id:Dict[str,int], 
                 freeze_encoder:bool,
                 **kwargs) -> SETR:
    
    config = BASE_CFG["_".join([encoder_model, "SETR"])]
    config['num_classes'] = len(label2id)
    config['id2label'] = {v: k for k, v in label2id.items()}
    config['label2id'] = label2id
    config['backbone_name'] = encoder_model 
    config['freeze_backbone'] = freeze_encoder
    config['criterion'] = 'focaltversky'
    config['loss_weights'] = {'asymmetric_ftl':0.5, 'asymmetric_fl':0.5}

    model = SETR(**config)
    
    # Freeze Backbone
    if freeze_encoder: 
        for name, param in model.backbone.named_parameters():
            param.requires_grad = False

    return model

class SETRImageProcessor:
    def __init__(self):
        self.dummy = True

    def __call__(self, data):
        return data


def create_img_processor(decoder_model:str, ignore_index:int=None) -> SETRImageProcessor:

    return SETRImageProcessor()



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
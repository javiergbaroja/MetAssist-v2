
from typing import Literal, Union, Dict

import numpy as np
from transformers import Dinov2Backbone
import torch
import torch.nn as nn
from models.losses_unet import Mask2FormerStyleLoss, AsymUnifiedFocalLoss
from models.unetr_base import BASE_CFG, get_embedding_combiner, ConvBlock, DeconvBlock, ModelOutput, ViTEncoder, foundation_backbones, facebook_backbones 
from timm import create_model as timm_create_model

class UNETR(nn.Module):
    """
    UNETR: A U-Net-like architecture with Vision Transformer (ViT) encoder for image segmentation tasks.
    Based  Hatamiyadeh et al. "UNETR: Transformers for 3D Medical Image Segmentation" (2022). in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2022, pp. 574-584

    This model leverages the power of Vision Transformers (ViT) for encoding the input image into feature representations
    and then uses a U-Net-like decoder to upsample these features back to the original image resolution for segmentation.

    Args:
        depth (int): The depth of the Vision Transformer encoder.
        num_classes (int): The number of output classes for segmentation.
        logits (bool): If True, the model outputs raw logits. If False, the model outputs class probabilities using softmax.
        **kwargs: Additional keyword arguments for configuring the Vision Transformer encoder.

    Attributes:
        vit (ViTEncoder): The Vision Transformer encoder.
        skip_blocks (nn.ModuleList): A list of modules for processing skip connections from the encoder.
        decoder (nn.ModuleList): A list of modules for decoding the feature maps back to the original resolution.
        segmentation_head (nn.Sequential): The final segmentation head that outputs the segmentation map.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass of the model. Takes an input tensor `x` and returns the segmentation map.

    Example:
        >>> model = UNETR(depth=32, num_classes=3, logits=False)
        >>> input_tensor = torch.randn(1, 3, 224, 224)
        >>> output = model(input_tensor)
        >>> print(output.shape)
        torch.Size([1, 3, 224, 224])
    """

    def __init__(self, 
                #  logits: bool = False,
                 backbone_name: Literal['dinov2_h_virchow2', 'dinov2_l','dinov2-h-virchow2', 'dinov2-l'] = 'dinov2_h_virchow2',
                 freeze_backbone: bool = False,
                 num_classes: int = 0,
                 embedding_combiner: Literal['linear', 'modulation', 'drop_cls_token'] = 'drop_cls_token',
                 criterion: Literal['mask2former', 'focaltversky'] = 'focaltversky',
                 loss_weights: dict = {'asymmetric_ftl':0.5, 'asymmetric_fl':0.5},
                 **kwargs) -> None:
        super().__init__()

        # redefine indices to have the features from the last block (==depth) and the rest equally spaced to a total of 4
        # create module list for upsampling blocks
        self.num_classes = num_classes
        self.criterion = self._get_criterion(criterion, loss_weights)
        self._out_channels = [64, 128, 256, 512] # corresponding to 
        self._in_channels = kwargs['in_chans']
        self._depth = kwargs['depth']
        self.backbone = self._get_backbone(backbone_name, **kwargs)
        
        convtrans_kwargs = {'kernel_size': 2, 'stride': 2, 'padding': 0, 'output_padding': 0}
        self.embedding_combiner = get_embedding_combiner(embedding_combiner, self.backbone.embed_dim, backbone_name)

        self.skip_blocks = nn.ModuleList([
            nn.Sequential(ConvBlock(self.backbone._in_channels, 32), ConvBlock(32, 64)),                                    # receives input
            nn.Sequential(DeconvBlock(self.backbone.embed_dim, 512), DeconvBlock(512, 256), DeconvBlock(256, 128)),         # receives from f8
            nn.Sequential(DeconvBlock(self.backbone.embed_dim, 512), DeconvBlock(512, 256)),                                # receives from f16 
            DeconvBlock(self.backbone.embed_dim, 512),                                                                      # receives from f24
            nn.ConvTranspose2d(self.backbone.embed_dim, 512, **convtrans_kwargs)])                                          # receives from f32  
        
        self.decoder = nn.ModuleList([
            nn.Sequential(ConvBlock(self._out_channels[0]*2, self._out_channels[0]), ConvBlock(self._out_channels[0], self._out_channels[0])),                                                                                          # receiving from conv_input and upsample_1
            nn.Sequential(ConvBlock(self._out_channels[1]*2, self._out_channels[1]), ConvBlock(self._out_channels[1], self._out_channels[1]), nn.ConvTranspose2d(self._out_channels[1], self._out_channels[0], **convtrans_kwargs)),    # receiving from upsample_1 and upsample_2
            nn.Sequential(ConvBlock(self._out_channels[2]*2, self._out_channels[2]), ConvBlock(self._out_channels[2], self._out_channels[2]), nn.ConvTranspose2d(self._out_channels[2], self._out_channels[1], **convtrans_kwargs)),    # receiving from upsample_2 and upsample_3
            nn.Sequential(ConvBlock(self._out_channels[3]*2, self._out_channels[3]), ConvBlock(self._out_channels[3], self._out_channels[3]), nn.ConvTranspose2d(self._out_channels[3], self._out_channels[2], **convtrans_kwargs)),    # receiving from upsample_3 and bottleneck
            nn.Identity()                                                                                                                                                                                                               # receiving from bottleneck                                                             
        ])

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(self._out_channels[0], num_classes, 1),
            nn.Identity() if self.criterion.needs_logits else nn.Softmax(dim=1),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    @property
    def device(self):
        """
        Returns the device of the model parameters.
        """
        # Get the device of the first parameter in the model
        return next(self.parameters()).device

    def _get_criterion(self, criterion, loss_weights) -> Union[Mask2FormerStyleLoss, AsymUnifiedFocalLoss]:
        if criterion == 'mask2former':
            return Mask2FormerStyleLoss(num_classes=self.num_classes, loss_weights=loss_weights)
        elif criterion == 'focaltversky':
            return AsymUnifiedFocalLoss(loss_weights=loss_weights)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
    
    def _get_backbone(self, backbone_name, **kwargs) -> Union[ViTEncoder, Dinov2Backbone]:
        if backbone_name in foundation_backbones:
                backbone = ViTEncoder(num_classes=0, **kwargs)
                backbone.load_state_dict(timm_create_model(kwargs['encoder_weights'], pretrained=True, mlp_layer=kwargs['mlp_layer'], act_layer=kwargs['act_layer'], out_indices=kwargs['feature_layers'], features_only=False).state_dict())
        elif backbone_name in facebook_backbones:
            backbone = Dinov2Backbone.from_pretrained(kwargs['encoder_weights'], out_indices=kwargs['feature_layers'], label2id=kwargs['label2id'], id2label=kwargs['id2label'], num_labels=self.num_classes, ignore_mismatched_sizes=True)
            backbone.embed_dim = backbone.config.hidden_size
            backbone._out_channels = [backbone.embed_dim]*backbone.config.num_hidden_layers
            backbone._in_channels = kwargs['in_chans']
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        return backbone

        
    def forward(self, pixel_values: torch.Tensor, y_true:torch.Tensor=None) -> ModelOutput:

        B, _, height, width = pixel_values.shape

        intermediates = list(self.backbone(pixel_values).feature_maps)
        # combine embeddings from cls token and patch tokens from the last block
        intermediates[-1] = self.embedding_combiner(intermediates[-1])
        # reshape to BCHW output format if not already in that format
        if len(intermediates[0].shape[2:]) < 2:
            H, W = self.backbone.patch_embed.dynamic_feat_size((height, width))
            intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]

        intermediates.insert(0, pixel_values) # we have [input, f8, f16, f24, f32]

        # get skip connections features
        # skip_features = [block(intermediates[i]) for i, block in enumerate(self.skip_blocks)]
        skip_features = []
        for i, block in enumerate(self.skip_blocks):
            skip_features.append(block(intermediates[i]))
        
        # pass through decoder (bottom up, which requires reversing the loop order)
        pixel_values = skip_features[-1]
        for i, block in enumerate(reversed(self.decoder)):
            if i == 0:
                pixel_values = block(pixel_values)
            elif i == len(self.decoder) - 1:
                # check if matching height and width for concatenation. If not matching crop and then concatenate
                if pixel_values.shape[2:] != skip_features[-i-1].shape[2:]:
                    diffY = skip_features[-i-1].shape[2] - pixel_values.shape[2]
                    diffX = skip_features[-i-1].shape[3] - pixel_values.shape[3]
                    pixel_values = nn.functional.pad(pixel_values, (diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2))
                pixel_values = block(torch.cat([pixel_values, skip_features[-i-1]], dim=1))
            else:
                pixel_values = block(torch.cat([pixel_values, skip_features[-i-1]], dim=1))

        pixel_values = self.segmentation_head(pixel_values)     
        return ModelOutput(preds=pixel_values,
                           logits=self.criterion.needs_logits,
                           losses_dict=self.criterion(pixel_values, y_true) if y_true is not None else None,
                           loss_weights_dict=self.criterion.loss_weights)
    

def create_model(encoder_model:str,  
                 label2id:Dict[str,int], 
                 freeze_encoder:bool,
                 **kwargs) -> UNETR:
    
    config = BASE_CFG["_".join([encoder_model, "UNETR"])]
    config['num_classes'] = len(np.unique(list(label2id.values())))
    config['id2label'] = {v: k for k, v in label2id.items()}
    config['label2id'] = label2id
    config['backbone_name'] = encoder_model 
    config['freeze_backbone'] = freeze_encoder
    config['criterion'] = 'focaltversky'
    config['loss_weights'] = {'asymmetric_ftl':0.5, 'asymmetric_fl':0.5}

    model = UNETR(**config)
    
    return model

class UNETRImageProcessor:
    def __init__(self):
        self.dummy = True

    def __call__(self, data):
        return data


def create_img_processor(decoder_model:str, ignore_index:int=None) -> UNETRImageProcessor:

    return UNETRImageProcessor()



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
        return outputs.preds
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
    outputs = outputs.squeeze().cpu()
    while len(outputs.shape) < 4:
        outputs = outputs.unsqueeze(0)
    # unsqueeze_first = True if len(outputs.shape) < 4 else False
    # outputs = outputs.unsqueeze(0) if unsqueeze_first else outputs
    assert len(outputs.shape) == 4, f"Expected 4D tensor (BCHW), got {outputs.shape}"
    return outputs.float()
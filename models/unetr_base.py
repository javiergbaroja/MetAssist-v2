import torch
from typing import List, Literal, Optional, Type, Union
import torch.nn as nn
from timm.layers import PatchEmbed
from functools import partial
from timm.models.vision_transformer import (Block, 
                                            Mlp,
                                            AttentionPoolLatent,
                                            resample_abs_pos_embed,
                                            get_act_layer,
                                            get_norm_layer,
                                            feature_take_indices,
                                            global_pool_nlc,
                                            PatchDropout)
from segmentation_models_pytorch.unet import Unet

from timm import create_model
from transformers import Dinov2Backbone
import torch
import torch.nn as nn
from timm.layers import PatchEmbed, LayerType, SwiGLUPacked
from losses_unet import Mask2FormerStyleLoss, AsymUnifiedFocalLoss



BASE_CFG ={
    'resnet34_UNet': {'encoder_weights': 'imagenet'},
    'dinov2_l_UNETR': {'encoder_weights': 'facebook/dinov2-large', 'get_head_embeddings':True, 'feature_layers':[6,12,18,24], 'depth':24, 'in_chans':3},
    'dinov2_h_virchow2_UNETR': {'encoder_weights': 'hf-hub:paige-ai/Virchow2', 'get_head_embeddings':True, 'feature_layers':[7,15,23,31], 'depth':32, 'in_chans':3, 'mlp_layer':SwiGLUPacked, 'act_layer':nn.modules.activation.SiLU},
    'dinov2_l_SETR': {'encoder_weights': 'facebook/dinov2-large', 'feature_layers':[24], 'depth':24, 'in_chans':3},
    'dinov2_h_virchow2_SETR': {'encoder_weights': 'hf-hub:paige-ai/Virchow2', 'feature_layers':[31], 'depth':32, 'mlp_layer':SwiGLUPacked, 'act_layer':nn.modules.activation.SiLU,'in_chans':3},
    'resnet34-UNet': {'encoder_weights': 'imagenet'},
    'dinov2-l_UNETR': {'encoder_weights': 'facebook/dinov2-large', 'get_head_embeddings':True, 'feature_layers':[6,12,18,24], 'depth':24, 'in_chans':3},
    'dinov2-h-virchow2_UNETR': {'encoder_weights': 'hf-hub:paige-ai/Virchow2', 'get_head_embeddings':True, 'feature_layers':[7,15,23,31], 'depth':32, 'in_chans':3, 'mlp_layer':SwiGLUPacked, 'act_layer':nn.modules.activation.SiLU},
    'dinov2-l_SETR': {'encoder_weights': 'facebook/dinov2-large', 'feature_layers':[24], 'depth':24, 'in_chans':3},
    'dinov2-h-virchow2_SETR': {'encoder_weights': 'hf-hub:paige-ai/Virchow2', 'feature_layers':[31], 'depth':32, 'mlp_layer':SwiGLUPacked, 'act_layer':nn.modules.activation.SiLU,'in_chans':3},
}

foundation_backbones = ['dinov2_h_virchow2','dinov2-h-virchow2']
facebook_backbones = ['dinov2_l', 'dinov2-l']

dino_backbones = {
    'dinov2_b': 'facebook/dinov2-base',
    'dinov2_l': 'facebook/dinov2-large',
    'dinov2-l': 'facebook/dinov2-large',
    'dinov2_g': 'facebook/dinov2-giant',
    'dinov2_s': 'facebook/dinov2-small',
    'dinov2_h_virchow2': 'hf-hub:paige-ai/Virchow2',
    'dinov2-h-virchow2': 'hf-hub:paige-ai/Virchow2',
}

# Specify the names of the decoder (Mask2Former) models
mask2former_cityscapes_semantic = {
    'swin-tiny-cityscapes-semantic': 'facebook/mask2former-swin-tiny-cityscapes-semantic',
    'swin-small-cityscapes-semantic': 'facebook/mask2former-swin-small-cityscapes-semantic',
    'swin-large-cityscapes-semantic': 'facebook/mask2former-swin-large-cityscapes-semantic',
    'swin-base-IN21k-cityscapes-semantic': 'facebook/mask2former-swin-base-IN21k-cityscapes-semantic',
}


class BackboneOutput:
    def __init__(self,
                 feature_maps:List[torch.Tensor]):
        self.feature_maps = tuple(feature_maps)

class ViTEncoder(nn.Module):
    def __init__(self, 
                 img_size:int=224, 
                 patch_size:int=14, 
                 in_chans:int=3, 
                 embed_dim=1280, 
                 depth:int=32, 
                 num_heads:int=16, 
                 num_classes:int=0,
                 mlp_ratio:float=5.3375,
                 drop_rate: float = 0.,
                 patch_drop_rate:float=0.0,
                 class_token=True,
                 pre_norm=False,
                 global_pool: Literal['', 'avg', 'avgmax', 'max', 'token', 'map'] = '',
                 fc_norm: Optional[bool] = False,
                 reg_tokens: int = 4,
                 dynamic_img_size: bool = True,
                 dynamic_img_pad: bool = False,
                 pos_drop_rate:float=0.0,
                 pos_embed:str = 'learn',
                 no_embed_class: bool = False,
                 drop_path_rate: float = 0.,
                 qkv_bias: bool = True,
                 qk_norm: bool = False,
                 init_values: Optional[float] = 1e-5,
                 proj_drop_rate:float = 0.,
                 attn_drop_rate:float = 0.,
                 mlp_layer: Type[nn.Module] = SwiGLUPacked,
                 norm_layer: Optional[LayerType] = None,
                 act_layer: Optional[LayerType] = torch.nn.modules.activation.SiLU,
                 feature_layers: List[int] = [],
                 get_head_embeddings: bool = False,
                 **kwargs):
        """
        Custom Vision Transformer encoder for segmentation_models_pytorch UNet.
        
        Parameters:
            img_size (int): Input image size (default: 224).
            patch_size (int): Patch size (default: 16).
            in_chans (int): Number of input channels (default: 3 for RGB).
            embed_dim (int): Dimension of the embedding vector for each patch.
            depth (int): Number of transformer blocks.
            num_heads (int): Number of attention heads in the transformer layers.
            mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
        """
        super().__init__()
        use_fc_norm = global_pool in ('avg', 'avgmax', 'max') if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU

        # EncoderMixin attributes
        self._in_channels = in_chans
        self._depth = depth
        self._out_channels = []

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.dynamic_img_size = dynamic_img_size
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.feature_layers = feature_layers # feature indices to return
        self.get_head_embeddings = get_head_embeddings
        if self.get_head_embeddings and depth-1 not in self.feature_layers:
            self.feature_layers.append(depth-1)  # always get the last feature

        # Patch embedding layer
        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args)
        num_patches = self.patch_embed.num_patches
        reduction = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if not pos_embed or pos_embed == 'none':
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)

        self.pos_drop = nn.Dropout(p=0.1)

        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()
        # Transformer encoder blocks        
        # decide number of blocks to use: depth if self.indices is empty, else max(self.indices) + 1
        depth = max(self.feature_layers) + 1 if self.feature_layers else depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self._out_channels = [embed_dim]*depth
        self._depth = depth
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(depth)
        ])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=reduction) for i in range(depth)]
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional embedding to the input tensor.

        This method applies positional embeddings to the input tensor `x`. If positional embeddings are not provided,
        it reshapes the input tensor. If dynamic image size is enabled, it resamples the positional embeddings to match
        the input tensor's spatial dimensions. It also handles the addition of class and regression tokens if they are present.

        Args:
            x (torch.Tensor): The input tensor of shape (B, H, W, C) where B is the batch size, H and W are the height and width,
                            and C is the number of channels.

        Returns:
            torch.Tensor: The tensor with positional embeddings, cls_token, and reg_token applied.

        Attributes:
            pos_embed (torch.Tensor or None): The positional embeddings. If None, the input tensor is reshaped.
            dynamic_img_size (bool): Flag indicating whether to dynamically resample positional embeddings based on input size.
            cls_token (torch.Tensor or None): The class token to be added to the input tensor.
            reg_token (torch.Tensor or None): The regression token to be added to the input tensor.
            no_embed_class (bool): Flag indicating whether to exclude the class token from positional embeddings.
            num_prefix_tokens (int): The number of prefix tokens to be added to the input tensor.
            pos_drop (nn.Module): Dropout layer applied to the positional embeddings.

        Example:
            >>> model = YourModel()
            >>> input_tensor = torch.randn(1, 224, 224, 768)
            >>> output_tensor = model._pos_embed(input_tensor)
            >>> print(output_tensor.shape)
            torch.Size([1, 50176, 768])
        """
        
        if self.pos_embed is None:
            return x.view(x.shape[0], -1, x.shape[-1])

        if self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)
    
    def pool(self, x: torch.Tensor, pool_type: Optional[str] = None) -> torch.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
            return x
        pool_type = self.global_pool if pool_type is None else pool_type
        x = global_pool_nlc(x, pool_type=pool_type, num_prefix_tokens=self.num_prefix_tokens)
        return x

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:

        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), self.feature_layers)

        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in take_indices:
                intermediates.append(x)
        
        if self.get_head_embeddings: # apply head if needed
            x = self.norm(x)
            x = self.pool(x)
            x = self.fc_norm(x)
            x = self.head_drop(x)
            
        # process intermediates
        if self.num_prefix_tokens:
            # split prefix (e.g. class token, register tokens) and spatial feature tokens
            intermediates = [y[:, self.num_prefix_tokens:] if idx<len(intermediates)-1 and self._depth-1 == max_index else y for idx, y in enumerate(intermediates)]
            cls_token = x[:, 0]
            x = x[:, 1+self.num_reg_tokens:]
            # concatenate cls token to patch tokens
            x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
            intermediates[-1] = x

        return BackboneOutput(intermediates)

# create a custom ModelOutput class to be the output of the models forwad pass
class ModelOutput:
    def __init__(self, 
                 preds:torch.Tensor, 
                 logits:bool,
                 losses_dict:dict, 
                 loss_weights_dict:dict):
        self.preds = preds
        self.logits = logits
        self.losses_dict = losses_dict
        self.loss_weights_dict = loss_weights_dict
    
    @property
    def loss(self):
        if self.losses_dict is None:
            return None
        for k in self.losses_dict.keys():
            if k in self.loss_weights_dict:
                self.losses_dict[k] *= self.loss_weights_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                self.losses_dict.pop(k)
        return sum(self.losses_dict.values())
    
    @property
    def probs(self):
        return torch.softmax(self.preds, dim=1) if self.logits else self.preds
    
    @property
    def y_pred(self):
        return torch.argmax(self.preds, dim=1)
        
    def __repr__(self) -> str:
        f = "ModelOutput(\n"
        f += f"  preds: {self.preds.shape}\n"
        f += f"  recieved logits: {self.logits}\n"
        f += f"  losses: {list(self.losses_dict.keys())}\n"
        f += f"  loss_weights: {self.loss_weights_dict}\n"
        f += f"  loss: {self.loss}\n)"
        return f

class ModulationCombiner(nn.Module):
    """
    Conditional Gating Module with constraints for stability.
    The class token generates scaling (gamma) and shifting (beta) parameters,
    constrained using sigmoid and tanh activations, respectively.
    """
    def __init__(self, embed_dim=1260, num_patches=256):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        # Constrained Linear layers for gamma and beta
        self.gamma_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()  # Constrains gamma in range [0, 1]
        )
        self.beta_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh()  # Constrains beta in range [-1, 1]
        )

    def forward(self, tokens:torch.Tensor) -> torch.Tensor:
        """
        Modulate patch tokens based on class token, with constraints for stability.
        
        Args:
            patch_tokens: Tensor of shape (batch_size, num_patches+1, embed_dim).
        Returns:
            Tensor of shape (batch_size, num_patches, embed_dim) after conditioning.
        """
        class_token = tokens[:,0,:].unsqueeze(1)  # Shape: (batch_size, 1, embed_dim)
        patch_tokens = tokens[:,1:,:]  # Shape: (batch_size, num_patches, embed_dim)

        # Compute constrained gamma and beta from class token
        gamma = self.gamma_fc(class_token)  # Shape: (batch_size, 1, embed_dim)
        beta = self.beta_fc(class_token)    # Shape: (batch_size, 1, embed_dim)
        
        # Repeat gamma and beta to match the shape of patch tokens
        gamma = gamma.repeat(1, self.num_patches, 1)  # Shape: (batch_size, num_patches, embed_dim)
        beta = beta.repeat(1, self.num_patches, 1)    # Shape: (batch_size, num_patches, embed_dim)

        # Apply residual modulation for stability
        conditioned_patches = patch_tokens + gamma * patch_tokens + beta
        return conditioned_patches

class LinearEmbeddingCombiner(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.relu = nn.ReLU()
    
    def forward(self, tokens:torch.Tensor) -> torch.Tensor:
        class_token = tokens[:,0,:].unsqueeze(1)  # Shape: (batch_size, 1, embed_dim)
        patch_tokens = tokens[:,1:,:]  # Shape: (batch_size, num_patches, embed_dim)
        combined = torch.cat((class_token.expand(-1, patch_tokens.shape[1], -1), patch_tokens), dim=-1)
        combined = self.relu(self.fc1(combined))
        return combined

class DropClsToken(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    
    def forward(self, tokens:torch.Tensor) -> torch.Tensor:
        return tokens[:,1:,:] # Shape: (batch_size, num_patches, embed_dim)

def get_embedding_combiner(embedding_combiner:str, embedding_dim:int, backbone_name:str):
    if backbone_name not in foundation_backbones:
        return nn.Identity()
    if embedding_combiner == 'modulation':
        return ModulationCombiner(embedding_dim)
    elif embedding_combiner == 'linear':
        return LinearEmbeddingCombiner(embedding_dim)
    elif embedding_combiner == 'drop_cls_token':
        return DropClsToken()
    else:
        raise ValueError(f"Unknown embedding combiner: {embedding_combiner}")


class DeconvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)
    
class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)
    


    

   

import torch
from transformers import (Dinov2Model, 
                          Dinov2Config, 
                          AutoImageProcessor, 
                          Mask2FormerConfig, 
                          Mask2FormerForUniversalSegmentation, 
                          Mask2FormerImageProcessor)
from typing import Dict, Optional, List, Tuple
from timm.layers import SwiGLUPacked  
import timm

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


def create_virchow2_model(label2id:Dict[str,int], id2label:Dict[int,str], num_labels:int, out_indices:list) -> Mask2FormerForUniversalSegmentation:
    ####### VIRCHOW MODEL #######
    model_config = Mask2FormerConfig(
        backbone='vit_huge_patch14_224',
        use_timm_backbone=True,
        use_pretrained_backbone=False,
        label2id=label2id,
        ignore_value=0,
        id2label=id2label,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        backbone_kwargs={'out_indices':out_indices,
                        'mlp_layer':SwiGLUPacked, 
                        'act_layer':torch.nn.SiLU,
                        'img_size': (224, 224),
                        'patch_size':14,
                        'init_values':1e-5,
                        'num_classes':0,
                        'mlp_ratio':5.3375,
                        'global_pool':"",
                        'dynamic_img_size': True,
                        'reg_tokens':4,
                        'pretrained_cfg': {"tag": 'orig_in21k',
                                            "custom_load": False,
                                            "input_size": [3, 224, 224],
                                            "fixed_input_size": False,
                                            "interpolation": "bicubic",
                                            "crop_pct": 1.0,
                                            "crop_mode": "center",
                                            "mean": [0.485, 0.456, 0.406],
                                            "std": [0.229, 0.224, 0.225],
                                            "num_classes": 0,
                                            "pool_size": None,
                                            "first_conv": "patch_embed.proj",
                                            "classifier": "head",
                                            "license": "CC-BY-NC-ND-4.0"}})
    # Instantiate Mask2Former model with ViT-H backbone with Virchow2 configuration
    model = Mask2FormerForUniversalSegmentation(model_config)
    # Load Virchow2 weights into Mask2Former backbone
    model.model.pixel_level_module.encoder._backbone.load_state_dict(timm.create_model('hf-hub:paige-ai/Virchow2', pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU, out_indices=out_indices, features_only=True).state_dict())
    return model


def create_facebook_model(encoder_model:str, decoder_model:str, label2id:Dict[str,int], id2label:Dict[int,str], num_labels:int, out_indices:list, freeze_encoder=True) -> Mask2FormerForUniversalSegmentation:
    model_config = Mask2FormerConfig.from_pretrained(decoder_model, ignore_mismatched_sizes=True, label2id=label2id, id2label=id2label, num_labels=num_labels)
    model_config.backbone_config = Dinov2Config.from_pretrained(encoder_model, out_indices=out_indices, label2id=label2id, id2label=id2label, 
                                                                    num_labels=num_labels, ignore_mismatched_sizes=True)
    # Instantiate Mask2Former model with Dinov2 backbone
    model = Mask2FormerForUniversalSegmentation(model_config)
    # Load Dinov2 weights into Mask2Former backbone
    model.model.pixel_level_module.encoder.load_state_dict(Dinov2Model.from_pretrained(encoder_model, out_indices=out_indices, 
                                                                label2id=label2id, id2label=id2label, 
                                                                num_labels=num_labels, ignore_mismatched_sizes=True).state_dict())
    return model


def create_model(encoder_model:str, decoder_model:str, label2id:Dict[str,int], id2label:Dict[int,str], out_indices:list, freeze_encoder:bool) -> Mask2FormerForUniversalSegmentation:
    unique_ids = list(set(label2id.values()))
    if len(label2id) != len(unique_ids):
        num_labels = len(unique_ids)    
    else:
        num_labels = len(label2id)

    if 'Unannotated' in label2id:
        training_region_id = label2id['Unannotated']
        # Remove 'Training region' from label2id and id2label
        del label2id['Unannotated']
        del id2label[training_region_id]

    if dino_backbones[encoder_model] == 'hf-hub:paige-ai/Virchow2':
        model = create_virchow2_model(label2id, id2label, num_labels, out_indices)
        
    elif 'facebook' in dino_backbones[encoder_model]:
        model = create_facebook_model(dino_backbones[encoder_model], mask2former_cityscapes_semantic[decoder_model], label2id, id2label, num_labels, out_indices)
    
    else:
        raise ValueError(f"Invalid encoder model: {encoder_model}")
    
    # Freeze Backbone
    if freeze_encoder: 
        for name, param in model.model.pixel_level_module.encoder.named_parameters():
            param.requires_grad = False

    return model
                        

def create_img_processor(decoder_model:str, ignore_index:int=None) -> Mask2FormerImageProcessor:
    if ignore_index is not None:
        kwargs = {'ignore_index': ignore_index, 'do_reduce_labels': False}
    else:
        kwargs = {}
    preprocessor = AutoImageProcessor.from_pretrained(mask2former_cityscapes_semantic[decoder_model], 
                                                  ignore_mismatched_sizes=True,
                                                  do_resize=False, do_rescale=False, do_normalize=False, **kwargs)
    return preprocessor


# def custom_post_process_semantic_segmentation(
#         outputs, target_sizes: Optional[List[Tuple[int, int]]] = None, 
#         return_logits:bool=False
#     ) -> torch.Tensor:
#         """
#         Modified from the Hugging Face Mask2FormerImageProcessor class. Unlike the original function, this function supports extraction of probability mask.
#         See: https://github.com/huggingface/transformers/blob/52ea4aa589324bae43dfb1b6db70335da7b68654/src/transformers/models/mask2former/image_processing_mask2former.py#L352
#         Converts the output of [`Mask2FormerForUniversalSegmentation`] into semantic segmentation maps. Only supports
#         PyTorch.

#         Args:
#             outputs ([`Mask2FormerForUniversalSegmentation`]):
#                 Raw outputs of the model.
#             target_sizes (`List[Tuple[int, int]]`, *optional*):
#                 List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
#                 final size (height, width) of each prediction. If left to None, predictions will not be resized.
#         Returns:
#             `List[torch.Tensor]`:
#                 A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
#                 corresponding to the target_sizes entry (if `target_sizes` is specified). Each entry of each
#                 `torch.Tensor` correspond to a semantic class id.
#         """
#         class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
#         masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

#         # Scale back to preprocessed image size - (384, 384) for all models
#         masks_queries_logits = torch.nn.functional.interpolate(
#             masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
#         )

#         # Remove the null class `[..., :-1]`
#         masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
#         masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

#         # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
#         segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
#         batch_size = class_queries_logits.shape[0]

#         # Resize logits and compute semantic segmentation maps
#         if target_sizes is not None:
#             if batch_size != len(target_sizes):
#                 raise ValueError(
#                     "Make sure that you pass in as many target sizes as the batch dimension of the logits"
#                 )

#             semantic_segmentation = []
#             for idx in range(batch_size):
#                 resized_logits = torch.nn.functional.interpolate(
#                     segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
#                 )
#                 if not return_logits:
#                     semantic_segmentation.append(resized_logits[0].argmax(dim=0))
#                 else:
#                     semantic_segmentation.append(resized_logits[0])
#         else:
#             raise ValueError("Please provide target sizes for resizing the logits")

#         return semantic_segmentation


# def post_process_output(outputs, target_sizes, return_logits=False):
#     """
#     Post-process the model outputs to create the final segmentation mask.

#     Args:
#         outputs: Model outputs.
#         target_sizes (list): List of target sizes.
#         return_logits (bool, optional): Whether to return the logits. Defaults to False.

#     Returns:
#         torch.Tensor: The final segmentation mask.
#     """
#     if return_logits:
#         final_shape = 4
#     else:
#         final_shape = 3
#     outputs = custom_post_process_semantic_segmentation(outputs, target_sizes=target_sizes, return_logits=return_logits)
#     outputs = torch.stack(outputs).squeeze()
#     while len(outputs.shape) < final_shape:
#         outputs = outputs.unsqueeze(0)
#     # unsqueeze_first = True if len(outputs.shape) < 4 else False
#     # outputs = outputs.unsqueeze(0) if unsqueeze_first else outputs
#     if return_logits:
#         assert len(outputs.shape) == final_shape, f"Expected 4D tensor (BCHW), got {outputs.shape}"
#     else:
#         assert len(outputs.shape) == final_shape, f"Expected 3D tensor (BHW), got {outputs.shape}"
#     return outputs.float()


class TrainCollator:
    def __init__(self, ignore_index:int):
        self.processor = create_img_processor('swin-large-cityscapes-semantic', ignore_index=ignore_index)
    def __call__(self, data) -> dict:
        batch = {}
        inputs = list(zip(*data))
        images = inputs[0]
        segmentation_maps = inputs[1]
        coords = inputs[-2]
        filenames = inputs[-1]
        # this function pads the inputs to the same size,
        # and creates a pixel mask
        # actually padding isn't required here since we are cropping
        data = self.processor(
            images,
            segmentation_maps=segmentation_maps,
            return_tensors="pt",
        )
        batch["pixel_values"] = data['pixel_values']
        batch["mask_labels"] = data['mask_labels']
        batch["class_labels"] = data['class_labels']
        batch["original_segmentation_maps"] = torch.stack(inputs[1])
        batch["coords"] = coords
        batch["filename"] = filenames
        
        return batch

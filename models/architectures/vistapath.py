import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.segment_anything.modeling import PromptEncoder
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoderLayer
from typing import Dict

class CustomSegmentationModel(nn.Module):
    def __init__(self, base_model_name, d_model, nhead, num_layers, bbx_random, vision_tune = False):
        super(CustomSegmentationModel, self).__init__()
        self.base_model = CLIPModel.from_pretrained(base_model_name, weights_only=False)
        
        # Unfreeze specific layers of PLIP for fine-tuning

        if vision_tune:
            for name, param in self.base_model.named_parameters():
                if "vision_model" in name:
                    param.requires_grad = True  # Unfreeze vision and text encoders
                else:
                    param.requires_grad = False

        else:
            for name, param in self.base_model.named_parameters():
                    param.requires_grad = False
        
        self.cross_attn_text = CrossAttentionLayer(d_model=d_model, nhead=nhead)
        self.cross_attn_bbx = CrossAttentionLayer(d_model=d_model, nhead=nhead)

        self.image_proj = self.base_model.visual_projection  # usually nn.Linear
        self.text_proj = self.base_model.text_projection 
        
        # Define a transformer encoder layer
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Define the segmentation decoder
        self.decoder = CustomDecoder(input_channels=d_model)

        # sam_model = sam_model_registry['vit_b'](checkpoint='/project/zhihuanglab/Peixian/Path_Seg/DL/MedSAM/work_dir/SAM/sam_vit_b_01ec64.pth')
        # self.prompt_encoder = sam_model.prompt_encoder

        self.prompt_encoder = PromptEncoder(
            embed_dim=512,
            image_embedding_size=(7, 7),
            input_image_size=(224, 224),
            mask_in_chans=16,
        )
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        

        self.bbx_random = bbx_random

    def forward(self, pixel_values, input_ids, attention_mask, box):

        with torch.no_grad():
            # box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box.shape) == 2:
                box = box[:, None, :]  # (B, 1, 4)
            

            if random.random() < self.bbx_random:
                box = None

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box,
                masks=None,
            )

        outputs = self.base_model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)

        # Extract vision transformer tokens: (B, 50, 512)
        image_tokens = outputs.vision_model_output.last_hidden_state[:, 1:, :]  # drop CLS token -> (B, 49, 512)

        # print(f"image_tokens: {image_tokens.shape}")
        image_proj = self.image_proj(image_tokens)  # (B, 49, d_model)
        # print(f"image_proj: {image_proj.shape}")

        # Project text and global-average pool for fusion
        text_tokens = outputs.text_model_output.last_hidden_state  # (B, T, 512)
        text_proj = self.text_proj(text_tokens)  # (B, T, d_model)

        # print(f"text_tokens: {text_tokens.shape}")
        # print(f"text_proj: {text_proj.shape}")

        # Cross-attention: let each image patch attend to the text
        fused_tokens = self.cross_attn_text(query=image_proj, key=text_proj, value=text_proj)  # (B, 49, d_model)

        fused_tokens = self.cross_attn_bbx(query=fused_tokens, key=sparse_embeddings, value=sparse_embeddings)  # (B, 49, d_model)

    

        # Transformer encoder (optional post-fusion modeling)
        fused_tokens = self.transformer_encoder(fused_tokens)  # (B, 49, d_model)

        B, N, D = fused_tokens.shape
        h = w = int(N ** 0.5)
        fused_feat = fused_tokens.permute(0, 2, 1).reshape(B, D, h, w)  # (B, d_model, 7, 7)


        # Segmentation output
        segmentation_output = self.decoder(fused_feat)  # (B, num_classes, H, W)

        if box is not None and box.shape[1] == 1:
            box = box[:, 0, :]  # (B, 4)

        return segmentation_output, box


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value):
        attn_output, _ = self.cross_attn(query=query, key=key, value=value)
        output = self.norm(query + self.dropout(attn_output))  # Add & Norm
        return output
    

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_queries):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, d_model)  # Learnable queries
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead) for _ in range(num_layers)
        ])

    def forward(self, memory):
        B = memory.size(0)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, num_queries, d_model)

        for layer in self.layers:
            queries = layer(queries, memory)

        return queries  



class CustomDecoder(nn.Module):
    def __init__(self, input_channels, num_classes=2):
        super(CustomDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(input_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 7x7 → 14x14

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 14x14 → 28x28

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 28x28 → 56x56

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),  # 56x56 → 224x224

            nn.Conv2d(32, num_classes, kernel_size=1)  # Final conv: per-pixel class scores
        )

    def forward(self, x):
        return self.decoder(x)  # (B, num_classes, H, W)
    

def create_img_processor(decoder_model:str="vinid/plip", **kwargs) -> CLIPProcessor:
    preprocessor = CLIPProcessor.from_pretrained(decoder_model)
    return preprocessor


class VISTAPATHProcessor:
    def __init__(self):
        self.dummy = True

    def __call__(self, data):
        return data


def create_img_processor(decoder_model:str, ignore_index:int=None) -> VISTAPATHProcessor:
    return VISTAPATHProcessor()


class VISTAPATH(torch.nn.Module):
    def __init__(self, label2id:dict):
        super().__init__()
        self.model = CustomSegmentationModel(
            base_model_name="vinid/plip",
            d_model=512,
            nhead=8,
            num_layers=4,
            bbx_random=1)
        self.processor = CLIPProcessor.from_pretrained("vinid/plip")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label2id = label2id

    # def forward(self, pixel_values, input_ids, attention_mask, labels=None, box=None):
    def forward(self, pixel_values:torch.Tensor):
        bboxes = [0, 0, 223, 223]
        # convert to tensor of appropriate shape
        boxes = torch.tensor(bboxes, dtype=torch.float32, device=pixel_values.device)
        boxes = boxes.unsqueeze(0).repeat(pixel_values.size(0), 1).to(self.device)
        original_shape = pixel_values.shape

        if pixel_values.min() < 0:
            raise ValueError("Input pixel_values have negative values. They should be in range 0-255.")
        if pixel_values.max() <= 1.0:
            pixel_values *= 255.0
        
        if pixel_values.size(2) != 224 or pixel_values.size(3) != 224:
            pixel_values = nn.functional.interpolate(pixel_values, size=(224, 224), mode='bilinear', align_corners=False).to(torch.uint8)
        # create preds_tensor, of shape batch, height, width, num_classes
        # shape = pixel_values.shape[2:] + (len(self.label2id),)
        preds  = torch.zeros((pixel_values.size(0), 1+len(self.label2id), pixel_values.size(2), pixel_values.size(3)), device=pixel_values.device)
        for class_name, class_idx in self.label2id.items():
            text_template = f"an image of {class_name}"
            text_templates = [text_template] * pixel_values.size(0)
            inputs = self.processor(text=text_templates, images=pixel_values, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
            pixel_values_ = inputs['pixel_values'].float().to(self.device)
            input_ids = inputs['input_ids'].long().to(self.device)
            attention_mask = inputs['attention_mask'].float().to(self.device)

            logits, _ = self.model(pixel_values_, input_ids, attention_mask, boxes)
            outputs = F.softmax(logits, dim=1)
            foreground_prob = outputs[:, 1, :, :]  # get foreground prob
            # reshape to original shape by interpolation
            preds[:, class_idx+1, :, :] = foreground_prob
        
        preds[:, 0, :, :] = 1 - torch.max(preds[:, 1:, :, :], dim=1)[0]  # background class prob  
    # reshape to original height and width by interpolation
        preds = nn.functional.interpolate(preds, size=original_shape[2:], mode='bilinear', align_corners=False)
        model_output = ModelOutput(
            preds=preds,
            logits=False,
            losses_dict={},
            loss_weights_dict={}
        )
        
        return model_output    
    
def create_model(
                label2id:Dict[str,int], 
                **kwargs) -> VISTAPATH:

    model = VISTAPATH(label2id=label2id)
    return model


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


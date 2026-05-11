import random
from typing import Tuple, List, Optional, Literal

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from skimage import exposure
import torch
import torchvision.transforms.functional as F
import torchvision.transforms._functional_tensor as F_t

RGB2HED = np.array([[0.65, 0.70, 0.29],
                                [0.07, 0.99, 0.11],
                                [0.27, 0.57, 0.78]])
HED2RGB = np.linalg.inv(RGB2HED)



###### PIL Image Augmentations ######

def identity_transformation(tile:Image.Image) -> Image.Image:
    return tile

def apply_pil_gamma_correction(tile:Image.Image):
    """ Apply gamma correction to the tile image. The gamma value is randomly selected from the range [0.8, 1.2].

    Args:
        tile (Image.Image): tile image to apply gamma correction to. Shape (H, W, C)

    Returns:
        Image.Image: augmented tile with gamma correction
    """

    gamma = random.uniform(0.8, 1.2)
    # inv_gamma = 1.0 / gamma
    np_tile = np.array(tile)
    np_tile = exposure.adjust_gamma(np_tile, gamma=gamma)
    
    return Image.fromarray(np_tile)
    
def apply_pil_additive_noise(tile:Image.Image) -> Image.Image:
    """ Apply Gaussian noise to the tile image. The standard deviation of the noise is randomly selected from the range [0, 0.1].

    Args:
        tile (Image.Image): tile image to apply Gaussian noise to
    
    Returns:
        Image.Image: augmented tile with Gaussian noise
    """

    np_tile = np.array(tile).astype(np.float32)
    std = random.uniform(0, 0.1)
    noise = np.random.normal(0, std, np_tile.shape)*255 # mean=0, std in range [0, 0.1]
    np_tile = np.clip(np_tile + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(np_tile)
    
def apply_pil_gaussian_blur(tile:Image.Image) -> Image.Image:
    """ Apply Gaussian blur to the tile image. The radius of the blur is randomly selected from the range [0, 0.1].

    Args:
        tile (Image.Image): tile image to apply Gaussian blur to
    
    Returns:
        Image.Image: augmented tile with Gaussian blur
    """
    
    return tile.filter(ImageFilter.GaussianBlur(radius=random.uniform(0., 0.1)))


def apply_pil_brightness_augmentation(tile:Image.Image) -> Image.Image:
    """ Return the original tile with brightness augmentation.

    Args:
        tile (Image.Image): tile image to return

    Returns:
        Image.Image: brightness adjusted tile
    """
    return ImageEnhance.Brightness(tile).enhance(random.uniform(0.65, 1.35))


def apply_pil_hsv_augmentation(tile:Image.Image) -> Image.Image:
    """ Apply color jittering to the tile image to simulate stain variations.
        The color augmentation is applied in the HSV color space:
        - Hue shift: [-0.1, 0.1]
        - Saturation shift: [-0.1, 0.1]
        - Brightness shift: [0.65, 1.35]

    Args:
        tile (Image.Image): tile image to apply color jittering to

    Returns:
        Image.Image: augmented tile with color jittering
    """
    # Apply color jittering to simulate stain variations
    saturation_shift = 1+random.uniform(-0.1, 0.1)
    hue_shift = 1+random.uniform(-0.1, 0.1)
    brightness_shift = random.uniform(0.65, 1.35)

    hsv_tile = tile.convert('HSV')
    h, s, v = hsv_tile.split()
    # shift hue by hue_shift percent. For example if hue is 100 and hue_shift is 0.1, new_hue = 100 + 0.1*100
    h = h.point(lambda i: i*hue_shift)
    # shift saturation
    s = s.point(lambda i: i*saturation_shift)
    # brightness shift
    v = v.point(lambda i: i*brightness_shift)
    
    # return to RGB
    hsv_tile = Image.merge('HSV', (h, s, v))
    # return to valid range 
    tile = hsv_tile.convert('RGB')
    
    return tile


def apply_pil_hed_augmentation(tile:Image.Image) -> Image.Image:
    """Apply augmentation to the tile image in the HED color space. 
    The augmentation is applied by shifting the values of the Hematoxylin (H), Eosin (E), and DAB (D) channels by a small random amount.
    Implementation in https://github.com/DIAGNijmegen/pathology-he-auto-augment/blob/main/he-randaugment/augmenters/color/utils/custom_hed_transform.py
    Why normalize to 0-1 and then use eps=2?, and why rescale from range (-1, 1)instead of (0,1)?

    Args:
        tile (Image.Image): tile image to apply augmentation to

    Returns:
        Image.Image: augmented tile in the RGB color space
    """

    tile = apply_pil_brightness_augmentation(tile)
    array = np.array(tile).astype(np.float32) / 255
    r, g, b = array[..., 0], array[..., 1], array[..., 2]
    eps = 2
    shift_ratio = 0.05 # shift ratio for HED-light transformation
    h_shift_a, h_shift_b = random.uniform(1-shift_ratio, 1+shift_ratio),random.uniform(-shift_ratio, shift_ratio)
    e_shift_a, e_shift_b = random.uniform(1-shift_ratio, 1+shift_ratio), random.uniform(-shift_ratio, shift_ratio)
    r_shift_a, r_shift_b = random.uniform(1-shift_ratio, 1+shift_ratio), random.uniform(-shift_ratio, shift_ratio)
    # create the Nx3 matrix P by stacking the R, G, B arrays
    p = np.stack([r.flatten(), g.flatten(), b.flatten()], axis=1)
    hed = -np.log((p + eps)) @ RGB2HED

    HED_shift = np.zeros_like(hed)
    HED_shift[:, 0] = h_shift_a*hed[:, 0] + h_shift_b
    HED_shift[:, 1] = e_shift_a*hed[:, 1] + e_shift_b
    HED_shift[:, 2] = r_shift_a*hed[:, 2] + r_shift_b

    # go back to RGB space
    RGB_aug = np.exp(-HED_shift @ HED2RGB) - eps
    # clip the values to be between 0 and 255
    RGB_aug = exposure.rescale_intensity(RGB_aug, in_range=(-1,1))
    RGB_aug = exposure.rescale_intensity(RGB_aug, out_range=(0,255)).astype(np.uint8)
    RGB_aug = RGB_aug.reshape(r.shape + (3,))
    return Image.fromarray(RGB_aug.astype(np.uint8))


###### PyTorch Image Augmentations ######


def _apply_torch_brightness_augmentation(tile:torch.Tensor) -> torch.Tensor:
    """ Return the original tile with brightness augmentation.

    Args:
        tile (torch.Tensor): tile image to return

    Returns:
        torch.Tensor: brightness augmented tile
    """
    return F.adjust_brightness(tile, random.uniform(0.65, 1.35))

def _apply_torch_gamma_correction(tile:torch.Tensor) -> torch.Tensor:
    """ Apply gamma correction to the tile image. The gamma value is randomly selected from the range [0.8, 1.2].

    Args:
        tile (torch.Tensor): tile image to apply gamma correction to. Shape (C, H, W)

    Returns:
        torch.Tensor: augmented tile with gamma correction
    """
    
    return F.adjust_gamma(tile, gamma=random.uniform(0.8, 1.2))


def _apply_torch_gaussian_blur(tile:torch.Tensor) -> torch.Tensor:
    """ Apply Gaussian blur to the tile image. The radius of the blur is randomly selected from the range [0, 0.1].

    Args:
        tile (Image.Image): tile image to apply Gaussian blur to
    
    Returns:
        Image.Image: augmented tile with Gaussian blur
    """
    
    return F.gaussian_blur(tile, sigma=random.uniform(0., 0.1), kernel_size=3)


def _apply_torch_additive_noise(tile:torch.Tensor) -> torch.Tensor:
    """ Apply Gaussian noise to the tile image. The standard deviation of the noise is randomly selected from the range [0, 0.1].

    Args:
        tile (torch.Tensor): tile image to apply Gaussian noise to
    
    Returns:
        torch.Tensor: augmented tile with Gaussian noise
    """
    sigma = random.uniform(0, 0.1)*255 # mean=0, std in range [0, 0.1]

    return torch.clamp(tile + torch.randn_like(tile) * sigma, 0, 255).to(torch.uint8)



def _torch_rescale_intensity(image: torch.Tensor, in_range: Tuple[float, float] = (0, 1), out_range: Tuple[float, float] = (0, 1)) -> torch.Tensor:
    """
    Rescale the intensity of the image to the specified range.

    Args:
        image (torch.Tensor): Input image tensor.
        in_range (Tuple[float, float]): Min and max values of the input intensity range.
        out_range (Tuple[float, float]): Min and max values of the output intensity range.

    Returns:
        torch.Tensor: Image tensor with rescaled intensity.
    """
    # Unpack the input and output ranges
    in_min, in_max = in_range
    out_min, out_max = out_range

    # Clip the image to the input range
    image = torch.clamp(image, min=in_min, max=in_max)

    # Rescale the intensity
    image = (image - in_min) / (in_max - in_min)  # Scale to [0, 1]
    image = image * (out_max - out_min) + out_min  # Scale to [out_min, out_max]

    return image


def _apply_torch_hed_augmentation(tile:torch.Tensor) -> torch.Tensor:
    """Apply augmentation to the tile image in the HED color space. 
    The augmentation is applied by shifting the values of the Hematoxylin (H), Eosin (E), and DAB (D) channels by a small random amount.
    Implementation in https://github.com/DIAGNijmegen/pathology-he-auto-augment/blob/main/he-randaugment/augmenters/color/utils/custom_hed_transform.py
    Why normalize to 0-1 and then use eps=2?, and why rescale from range (-1, 1)instead of (0,1)?

    Args:
        tile (torch.Tensor): tile image to apply augmentation to

    Returns:
        torch.Tensor: augmented tile in the RGB color space
    """

    tile = _apply_torch_brightness_augmentation(tile)
    tile = tile.to(torch.float32) / 255
    R, G, B = tile.unbind(-3)
    eps = 2
    shift_ratio = 0.05 # shift ratio for HED-light transformation
    h_shift_a, h_shift_b = random.uniform(1-shift_ratio, 1+shift_ratio),random.uniform(-shift_ratio, shift_ratio)
    e_shift_a, e_shift_b = random.uniform(1-shift_ratio, 1+shift_ratio), random.uniform(-shift_ratio, shift_ratio)
    r_shift_a, r_shift_b = random.uniform(1-shift_ratio, 1+shift_ratio), random.uniform(-shift_ratio, shift_ratio)
    # create the Nx3 matrix P by stacking the R, G, B arrays
    P = torch.stack([R.flatten(), G.flatten(), B.flatten()], axis=1)
    HED = -torch.log((P + eps)) @ torch.tensor(RGB2HED)

    HED_shift = torch.zeros_like(HED)
    HED_shift[:, 0] = h_shift_a*HED[:, 0] + h_shift_b
    HED_shift[:, 1] = e_shift_a*HED[:, 1] + e_shift_b
    HED_shift[:, 2] = r_shift_a*HED[:, 2] + r_shift_b

    # go back to RGB space
    RGB_aug = torch.exp(-HED_shift @ torch.tensor(HED2RGB)) - eps
    # clip the values to be between 0 and 255
    RGB_aug = _torch_rescale_intensity(RGB_aug, in_range=(-1,1))
    RGB_aug = _torch_rescale_intensity(RGB_aug, out_range=(0,255)).astype(np.uint8)
    RGB_aug = RGB_aug.reshape(R.shape + (3,))
    return Image.fromarray(RGB_aug.astype(np.uint8))


def _apply_torch_hsv_augmentation(tile:torch.Tensor) -> torch.Tensor:
    """ Apply color jittering to the tile image to simulate stain variations.
        The color augmentation is applied in the HSV color space:
        - Hue shift: [-0.1, 0.1]
        - Saturation shift: [-0.1, 0.1]
        - Brightness shift: [0.65, 1.35]

    Args:
        tile (torch.Tensor): tile image (RGB) to apply color jittering to

    Returns:
        torch.Tensor: augmented tile with HSV color jittering
    """
    # Apply color jittering to simulate stain variations
    saturation_shift = 1+random.uniform(-0.1, 0.1)
    hue_shift = 1+random.uniform(-0.1, 0.1)
    brightness_shift = random.uniform(0.65, 1.35)

    hsv_tile = F_t._rgb2hsv(tile)
    h, s, v = hsv_tile.unbind(-3)
    # shift hue by hue_shift percent. For example if hue is 100 and hue_shift is 0.1, new_hue = 100 + 0.1*100
    h *= hue_shift 
    # shift saturation
    s *= saturation_shift
    # brightness shift
    v *= brightness_shift
    
    # return to RGB
    hsv_tile = torch.stack((h, s, v), dim=-3)
    # clamp values to valid range
    hsv_tile = torch.clamp(hsv_tile, min=0, max=255)
    # return to valid range 
    tile = F_t._hsv2rgb(hsv_tile)
    
    return tile

class TestTimeAugmentation:
    def __init__(self, rotation:int, flip:str, color_jitter:Literal['hsv', 'hed'], noise:Optional[bool]=None, blur:Optional[bool]=None, gamma:Optional[bool]=None):
        self.rotation = rotation
        self.flip = flip
        if color_jitter not in ['hsv', 'hed']:
            raise ValueError(f"Color jitter {color_jitter} not supported. Please choose from ['hsv', 'hed']")
        self.color_jitter = color_jitter
        self.noise = noise
        self.blur = blur
        self.gamma = gamma

    def __call__(self, tile:Image.Image) -> Image.Image:

        if self.rotation != 0:
            tile = tile.rotate(self.rotation)
        if self.flip == 'h':
            tile = tile.transpose(Image.FLIP_LEFT_RIGHT)
        elif self.flip == 'v':
            tile = tile.transpose(Image.FLIP_TOP_BOTTOM)

        if self.color_jitter:
            tile = apply_pil_hsv_augmentation(tile) if self.color_jitter == 'hsv' else _apply_pil_hed_augmentation(tile)
        if self.noise:
            tile = apply_pil_additive_noise(tile)
        if self.blur:
            tile = apply_pil_gaussian_blur(tile)
        if self.gamma:
            tile = apply_pil_gamma_correction(tile)
        
        return tile
    
    def reverse_tta(self, tile:Image.Image) -> Image.Image:
        if self.flip == 'h':
            tile = F.hflip(tile)
        elif self.flip == 'v':
            tile = F.vflip(tile)
        if self.rotation != 0:
            tile = F.rotate(tile, -self.rotation)
        return tile
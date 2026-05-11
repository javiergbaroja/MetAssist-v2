import openslide
import cv2
import numpy as np
from typing import List, Tuple, Optional
from utils.wsi import detect_tissue_mask


def extract_tile_coords_new(slide: openslide.OpenSlide, 
                            img_dim: Tuple[int, int], 
                            img_dim_padded: Tuple[int, int],  
                            tile_size: int, 
                            step_size: int, 
                            read_origin: Tuple[int, int], 
                            level: int, 
                            filter_mask: Optional[np.ndarray] = None, 
                            rm_background: Optional[str] = 'saturation',
                            min_ratio_object: float = 1e-5) -> List[Tuple[int, int, int, int]]:
    """
    Extracts coordinates for tiles from a WSI using vectorized operations.

    Args:
        slide: The WSI object.
        img_dim: Original shape of the WSI region (rows, cols).
        img_dim_padded: Padded shape of the WSI region (rows, cols).
        tile_size: Size of the tile (square).
        step_size: Stride for tile extraction.
        read_origin: (x, y) origin of the read.
        level: WSI pyramid level.
        filter_mask: Optional binary mask to restrict extraction.
        rm_background: Strategy ('saturation' or 'fixed') to generate mask if filter_mask is None.

    Returns:
        List of tuples: (x_start, x_end, y_start, y_end)
    """

    # --- 1. Prepare Processing Mask & Scale Factors ---
    processing_mask = None
    scale_h = scale_w = 1.0

    if filter_mask is not None:
        processing_mask, (scale_h, scale_w) = _resize_processing_mask(filter_mask, img_dim)
        
    elif rm_background:
        processing_mask, thumb_ds = detect_tissue_mask(slide, read_origin, rm_background, min_ratio_object)
        
        # Calculate scale: Reading Level -> Thumbnail Level
        reading_ds = slide.level_downsamples[level]
        scale_ratio = reading_ds / thumb_ds
        scale_h = scale_w = scale_ratio

    # --- 2. Generate Vectorized Grid ---
    # Create grid based on the padded image dimensions
    x_range = np.arange(0, img_dim_padded[0] - tile_size + 1, step_size)
    y_range = np.arange(0, img_dim_padded[1] - tile_size + 1, step_size)
    
    # 'ij' indexing: x corresponds to rows (height), y to columns (width)
    xv, yv = np.meshgrid(x_range, y_range, indexing='ij')
    x_flat = xv.ravel()
    y_flat = yv.ravel()

    # --- 3. Filter Coordinates (Integral Image Method) ---
    if processing_mask is not None:
        # A. Compute Integral Image
        # This creates a summed-area table. Shape is (H+1, W+1).
        # Allows O(1) calculation of sum of pixels in any rectangle.
        img_integral = cv2.integral(processing_mask)
        mask_h, mask_w = processing_mask.shape

        # B. Map Grid Coordinates to Mask Coordinates
        # Use floor for start and ceil for end to handle fractional scaling conservatively.
        # This ensures that if a tile partially overlaps a mask pixel, we count it.
        x1 = np.floor(x_flat * scale_h).astype(int)
        y1 = np.floor(y_flat * scale_w).astype(int)
        x2 = np.ceil((x_flat + tile_size) * scale_h).astype(int)
        y2 = np.ceil((y_flat + tile_size) * scale_w).astype(int)

        # C. Clip to Mask Boundaries
        # Integral image has dimensions +1 compared to mask
        x1 = np.clip(x1, 0, mask_h)
        y1 = np.clip(y1, 0, mask_w)
        x2 = np.clip(x2, 0, mask_h)
        y2 = np.clip(y2, 0, mask_w)

        # D. Calculate Sum of Tissue Pixels in each Tile
        # Formula: Sum = BottomRight - BottomLeft - TopRight + TopLeft
        tile_sums = (img_integral[x2, y2] 
                   - img_integral[x1, y2] 
                   - img_integral[x2, y1] 
                   + img_integral[x1, y1])

        # E. Keep tiles with ANY valid tissue pixels
        valid_indices = tile_sums > 0
        x_flat = x_flat[valid_indices]
        y_flat = y_flat[valid_indices]

    # --- 4. Format Output ---
    # Construct [x_start, x_end, y_start, y_end]
    # Clamp end coordinates to original image dimensions
    x_end = np.minimum(x_flat + tile_size, img_dim[0])
    y_end = np.minimum(y_flat + tile_size, img_dim[1])
    
    # Stack and return
    coords_array = np.stack((x_flat, x_end, y_flat, y_end), axis=1)
    
    return coords_array.tolist()


def _resize_processing_mask(filter_mask: np.ndarray, 
                            img_dim: Tuple[int, int], 
                            target_downsample: int = 32) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Resizes a high-res user mask to a low-res thumbnail for efficient processing.
    Returns the resized mask and the scaling factors (scale_h, scale_w).
    """
    target_h = max(img_dim[0] // target_downsample, 1)
    target_w = max(img_dim[1] // target_downsample, 1)

    # Resize and dilate
    processed_mask = cv2.resize(filter_mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    processed_mask = cv2.dilate(processed_mask, np.ones((3, 3), np.uint8), iterations=1)

    # Calculate scale factors (New / Old)
    scale_factor_h = target_h / img_dim[0]
    scale_factor_w = target_w / img_dim[1]
    
    return processed_mask, (scale_factor_h, scale_factor_w)


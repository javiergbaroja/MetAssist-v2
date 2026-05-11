import openslide
import math
import os
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional
from glob import glob
import random
import geojson

# image processing and array manipulation
import numpy as np
from PIL import Image
import cv2

# torch
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

# utils
from data.augmentations import (
    identity_transformation,
    apply_pil_brightness_augmentation,
    apply_pil_hsv_augmentation,
    apply_pil_hed_augmentation,
    apply_pil_additive_noise,
    apply_pil_gaussian_blur,
    apply_pil_gamma_correction,
)
from data.augmentations import TestTimeAugmentation
from utils.wsi import ACCEPTED_WSI_TYPES, check_wsi_exists_all_formats, prepare_read_from_slide
from data.tiling import extract_tile_coords_new


class BaseSlideDataset:
    """
    A dataset class for performing inference on whole slide images (WSIs).

    Handles WSI preparation, tile extraction, and memory-efficient chunked stitching
    of model predictions. Supports test-time augmentations (TTA).

    Memory management strategy:
        The WSI prediction space is divided into non-overlapping chunks of size
        `chunk_size x chunk_size` (in prediction space). Each chunk maintains its
        own accumulation buffer (logit sum + count map). Once all tiles that
        contribute to a chunk have been processed, the chunk is finalised
        (softmax → argmax), written to the output array, and its buffer freed.
        This bounds peak memory to O(num_classes * chunk_size^2 * max_active_chunks)
        rather than O(num_classes * H * W).

    Attributes:
        downsample_factor (int): Factor by which to downsample the WSI.
        filter_mask (np.ndarray): Binary mask limiting tile extraction to relevant tissue.
        tile_size (int): Size of each tile in model-input pixels.
        step_size (int): Sliding-window step size in model-input pixels.
        chunk_size (int): Side length (in prediction pixels) of each stitching chunk.
        wsi (openslide.OpenSlide): The open WSI object.
        coords (List[Tuple[int, int, int, int]]): Tile coordinates (prediction space).
        crop_size (int): Pixels cropped from tile prediction edges before stitching.
        level (int): OpenSlide pyramid level used for reading.
        level_downsampling (int): Downsampling factor of the chosen level.
        tiling_downsample_factor (int): Additional downsample between read level and model input.
        exact_resolution (float): Actual MPP at the chosen level.
        original_shape (Tuple[int, int]): WSI dimensions at the chosen level (H, W).
        padded_shape (Tuple[int, int]): Padded WSI dimensions ensuring tile divisibility.
        read_origin (Tuple[int, int]): Top-left origin for WSI reads (accounts for WSI offset).
    """

    def __init__(
        self,
        wsi_path: str,
        filter_mask: np.ndarray,
        resolution: float,
        tile_size: int,
        step_size: int,
        downsample_factor: int,
        crop_size: int,
        apply_tta: bool,
        rotations: Optional[List[int]],
        flips: Optional[Tuple[str, str]],
        color_jitter: Optional[str],
        noise: Optional[bool],
        blur: Optional[bool],
        gamma: Optional[bool],
        chunk_size: int = 4096,
    ):
        self.downsample_factor = downsample_factor
        self.filter_mask = filter_mask
        self.tile_size = tile_size
        self.step_size = step_size
        self.chunk_size = chunk_size
        self._wsi = None  

        self._prepare_slide(wsi_path, resolution)

        self.coords = extract_tile_coords_new(
            slide=self.wsi,
            filter_mask=filter_mask,
            img_dim=self.original_shape,
            img_dim_padded=self.padded_shape,
            tile_size=self.tile_size * self.tiling_downsample_factor,
            step_size=self.step_size * self.tiling_downsample_factor,
            read_origin=self.read_origin,
            level=self.level,
        )
        self.coords = sorted(self.coords, key=lambda x: (x[0] // chunk_size, x[2] // chunk_size))
        self.num_augs = len(rotations) * (len(flips) + 1) if apply_tta else 1
        self.coords = [coord for coord in self.coords for _ in range(self.num_augs)] if self.num_augs > 1 else self.coords

        self.flips_values = [f for f in ('h', 'v') if f in flips]
        self.rotations = rotations
        self.color_jitter = color_jitter
        self.apply_tta = apply_tta
        self.noise = noise
        self.blur = blur
        self.gamma = gamma
        self.angles: List[int] = []
        self.flips: List[str] = []
        self.prepare_augmentation_series()

        # --- crop_size: clamp to valid range and warn if step_size leaves no overlap ---
        max_crop = max((tile_size - step_size) // 2, 0)
        if step_size >= tile_size and crop_size > 0:
            import warnings
            warnings.warn(
                f"crop_size={crop_size} requested but step_size={step_size} >= "
                f"tile_size={tile_size}: no tile overlap exists. Setting crop_size=0.",
                UserWarning,
            )
        self.crop_size = min(crop_size, max_crop)

        # --- Chunked stitching state ---
        # Prediction-space dimensions
        self._pred_H = self.padded_shape[0] // self.tiling_downsample_factor
        self._pred_W = self.padded_shape[1] // self.tiling_downsample_factor
        self._n_chunks_row = math.ceil(self._pred_H / self.chunk_size)
        self._n_chunks_col = math.ceil(self._pred_W / self.chunk_size)

        # Pre-assign tiles to chunks (must happen after crop_size is set)
        self._tile_to_chunks: List[List[Tuple[int, int]]] = []
        self._chunk_remaining: Dict[Tuple[int, int], int] = defaultdict(int)
        self._assign_tiles_to_chunks()

        # Lazily allocated chunk buffers: chunk_id -> {'pred': Tensor, 'count': Tensor}
        self._chunk_buffers: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}

        # Final output written incrementally as chunks are finalised
        self._final_prediction = np.zeros((self._pred_H, self._pred_W), dtype=np.uint8)

        # Global tile counter for stitch_predictions (maps batch tiles → precomputed chunk lists)
        self._tile_idx = 0

        # Number of model output classes — set on first stitch call
        self._num_classes: Optional[int] = None

    # ------------------------------------------------------------------
    # Slide preparation
    # ------------------------------------------------------------------
    @property
    def wsi(self):
        if self._wsi is None:
            self._wsi = openslide.open_slide(self.wsi_path)
        return self._wsi

    def _prepare_slide(self, wsi_path: str, resolution: float) -> None:
        """Open the WSI, determine the best pyramid level, and compute padded shape."""
        file_type = os.path.splitext(wsi_path)[1]
        # if os.getenv('TMPDIR') is not None:
        #     # If temporary environment exists (e.g., on an HPC), copy the WSI to local storage for faster access.
        #     local_wsi_path = os.path.join(os.getenv('TMPDIR'), os.path.basename(wsi_path))
        #     shutil.copy(wsi_path, local_wsi_path)
        
        #     if  file_type == '.mrxs':
        #         # For .mrxs files, copy the entire directory if it exists (handles associated files), named after the WSI without extension
        #         wsi_dir = os.path.splitext(wsi_path)[0]
        #         local_wsi_dir = os.path.join(os.getenv('TMPDIR'), os.path.splitext(os.path.basename(wsi_path))[0])
        #         if not os.path.exists(local_wsi_dir):
        #             shutil.copytree(wsi_dir, local_wsi_dir)

        #     wsi_path = local_wsi_path

        # self.wsi = openslide.open_slide(wsi_path)
        self.wsi_path = wsi_path
        (
            self.level,
            self.level_downsampling,
            self.exact_resolution,
            self.tiling_downsample_factor,
            self.original_shape,
            self.read_origin,
        ) = prepare_read_from_slide(self.wsi, resolution, file_type=file_type)

        pad_x = (
            self.tile_size * self.tiling_downsample_factor
            - self.original_shape[0] % (self.tile_size * self.tiling_downsample_factor)
        ) % (self.tile_size * self.tiling_downsample_factor)

        pad_y = (
            self.tile_size * self.tiling_downsample_factor
            - self.original_shape[1] % (self.tile_size * self.tiling_downsample_factor)
        ) % (self.tile_size * self.tiling_downsample_factor)

        self.padded_shape = (
            self.original_shape[0] + pad_x,
            self.original_shape[1] + pad_y,
        )

    # ------------------------------------------------------------------
    # Chunk assignment
    # ------------------------------------------------------------------

    def _cropped_pred_coords(
        self, row_start_orig: int, col_start_orig: int
    ) -> Tuple[int, int, int, int]:
        """
        Convert a raw tile coordinate (original space) to its cropped prediction-space
        bounding box.  This is the single source of truth for crop logic, shared between
        pre-assignment and stitching.

        Returns:
            (rs, re, cs, ce) — row/col start/end in prediction space after edge cropping.
        """
        rs = row_start_orig // self.tiling_downsample_factor
        cs = col_start_orig // self.tiling_downsample_factor
        re = rs + self.tile_size
        ce = cs + self.tile_size

        crop = self.crop_size
        rs = rs + crop if rs != 0 else rs
        re = re - crop if re != self._pred_H else re
        cs = cs + crop if cs != 0 else cs
        ce = ce - crop if ce != self._pred_W else ce

        return rs, re, cs, ce

    def _assign_tiles_to_chunks(self) -> None:
        """
        Pre-compute which chunks each tile contributes to (using cropped coordinates)
        and build the per-chunk tile-count map used to detect when a chunk is complete.
        """
        for coord in self.coords:
            row_start_orig, _, col_start_orig, _ = coord
            rs, re, cs, ce = self._cropped_pred_coords(row_start_orig, col_start_orig)

            # Chunk indices spanned by this tile's (cropped) footprint
            cr_start = rs // self.chunk_size
            cr_end = min((re - 1) // self.chunk_size, self._n_chunks_row - 1)
            cc_start = cs // self.chunk_size
            cc_end = min((ce - 1) // self.chunk_size, self._n_chunks_col - 1)

            tile_chunks: List[Tuple[int, int]] = []
            for cr in range(cr_start, cr_end + 1):
                for cc in range(cc_start, cc_end + 1):
                    chunk_id = (cr, cc)
                    tile_chunks.append(chunk_id)
                    self._chunk_remaining[chunk_id] += 1

            self._tile_to_chunks.append(tile_chunks)

    # ------------------------------------------------------------------
    # Chunk helpers
    # ------------------------------------------------------------------

    def _chunk_bounds(self, cr: int, cc: int) -> Tuple[int, int, int, int]:
        """Return (row_start, row_end, col_start, col_end) for chunk (cr, cc)."""
        rs = cr * self.chunk_size
        cs = cc * self.chunk_size
        re = min(rs + self.chunk_size, self._pred_H)
        ce = min(cs + self.chunk_size, self._pred_W)
        return rs, re, cs, ce

    def _allocate_chunk(self, chunk_id: Tuple[int, int]) -> None:
        """Lazily allocate accumulation buffers for a chunk."""
        cr, cc = chunk_id
        rs, re, cs, ce = self._chunk_bounds(cr, cc)
        h, w = re - rs, ce - cs
        self._chunk_buffers[chunk_id] = {
            "pred": torch.zeros((self._num_classes, h, w), dtype=torch.float32),
            "count": torch.zeros((h, w), dtype=torch.float32),
        }

    def _finalize_chunk(self, chunk_id: Tuple[int, int]) -> None:
        """
        Average accumulated logits, apply softmax, take argmax, write to output,
        and immediately free the chunk buffer to release memory.
        """
        buf = self._chunk_buffers.pop(chunk_id)
        cr, cc = chunk_id
        rs, re, cs, ce = self._chunk_bounds(cr, cc)

        count = buf["count"].clamp(min=1.0).unsqueeze(0)  # (1, H, W)
        averaged_logits = buf["pred"] / count               # (C, H, W)

        # Argmax over averaged logits (no softmax needed for argmax)
        self._final_prediction[rs:re, cs:ce] = (
            averaged_logits.argmax(dim=0).numpy().astype(np.uint8)
        )

    # ------------------------------------------------------------------
    # Core stitching
    # ------------------------------------------------------------------

    def stitch_predictions(
        self,
        tile_predictions: torch.Tensor,
        coords: List[Tuple[int, int, int, int]],
        augmentations: List[Optional[TestTimeAugmentation]],
    ) -> None:
        """
        Accumulate a batch of tile predictions into per-chunk buffers.

        For each tile:
          1. Inverse-transform predictions if TTA was applied.
          2. Crop tile edges to reduce boundary artefacts.
          3. Accumulate raw logits (not softmax) into every chunk the tile overlaps.
          4. Decrement remaining-tile counter; finalise and free any completed chunk.

        Args:
            tile_predictions: Model output logits, shape (B, C, tile_size, tile_size).
            augmentations: Per-tile TTA objects (or None) for inverse transform.
            coords: Per-tile coordinates in original (non-downsampled) space.
        """
        if self._num_classes is None:
            self._num_classes = tile_predictions.shape[1]

        crop = self.crop_size

        for pred_logits, (row_start_orig, __, col_start_orig, __), augmentation in zip(
            tile_predictions, coords, augmentations
        ):
            # --- 1. Inverse TTA ---
            if augmentation is not None:
                pred_logits = augmentation.reverse(pred_logits.unsqueeze(0)).squeeze(0)

            # --- 2. Compute cropped prediction-space bounding box ---
            rs, re, cs, ce = self._cropped_pred_coords(row_start_orig, col_start_orig)

            # Corresponding slice into the (uncropped) pred_logits tensor
            raw_rs = row_start_orig // self.tiling_downsample_factor
            raw_cs = col_start_orig // self.tiling_downsample_factor
            tile_crop_rs = rs - raw_rs   # 0 or crop
            tile_crop_re = re - raw_rs   # tile_size or tile_size - crop
            tile_crop_cs = cs - raw_cs
            tile_crop_ce = ce - raw_cs

            cropped_logits = pred_logits[
                :, tile_crop_rs:tile_crop_re, tile_crop_cs:tile_crop_ce
            ]  # (C, h_crop, w_crop)

            # --- 3. Accumulate into overlapping chunks ---
            chunk_ids = self._tile_to_chunks[self._tile_idx]
            self._tile_idx += 1

            for chunk_id in chunk_ids:
                if chunk_id not in self._chunk_buffers:
                    self._allocate_chunk(chunk_id)

                cr, cc = chunk_id
                c_rs, c_re, c_cs, c_ce = self._chunk_bounds(cr, cc)

                # Intersection of cropped tile with this chunk (prediction space)
                i_rs = max(rs, c_rs)
                i_re = min(re, c_re)
                i_cs = max(cs, c_cs)
                i_ce = min(ce, c_ce)

                # Slice within cropped_logits
                t_rs = i_rs - rs
                t_re = i_re - rs
                t_cs = i_cs - cs
                t_ce = i_ce - cs

                # Slice within chunk buffer
                b_rs = i_rs - c_rs
                b_re = i_re - c_rs
                b_cs = i_cs - c_cs
                b_ce = i_ce - c_cs

                self._chunk_buffers[chunk_id]["pred"][:, b_rs:b_re, b_cs:b_ce] += (
                    cropped_logits[:, t_rs:t_re, t_cs:t_ce]
                )
                self._chunk_buffers[chunk_id]["count"][b_rs:b_re, b_cs:b_ce] += 1

                # --- 4. Finalise chunk if all its tiles have been processed ---
                self._chunk_remaining[chunk_id] -= 1
                if self._chunk_remaining[chunk_id] == 0:
                    self._finalize_chunk(chunk_id)

    # ------------------------------------------------------------------
    # Final prediction assembly
    # ------------------------------------------------------------------

    def create_final_predictions(
        self, return_probs: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Return the completed segmentation map.

        Under normal operation all chunks are finalised during `stitch_predictions`.
        Any residual chunks (e.g. if inference was interrupted) are finalised here
        as a safety net.

        Args:
            return_probs: Not supported with chunked stitching (buffers have already
                          been freed).  Passing True raises a ValueError.

        Returns:
            (final_prediction, None): uint8 segmentation map of shape
            (padded_H // tiling_downsample_factor, padded_W // tiling_downsample_factor).
        """
        if return_probs:
            raise ValueError(
                "return_probs=True is not supported with chunked stitching: "
                "chunk buffers are freed incrementally during stitch_predictions. "
                "If you need full probability maps, set chunk_size to cover the "
                "entire WSI or accumulate probabilities externally."
            )

        # Safety net: finalise any chunks not yet written (should not occur in practice)
        for chunk_id in list(self._chunk_buffers.keys()):
            self._finalize_chunk(chunk_id)

        return self._final_prediction, None

    # ------------------------------------------------------------------
    # Augmentation helpers
    # ------------------------------------------------------------------

    def prepare_augmentation_series(self) -> None:
        """Build per-tile angle and flip lists for TTA."""
        existing_flips = [0] + self.flips_values
        angles = [
            angle
            for angle in self.rotations
            for _ in existing_flips
        ]
        flips = existing_flips * len(self.rotations)
        n_repeats = len(self.coords) / len(angles)
        self.angles = angles * int(n_repeats)
        self.flips = flips * int(n_repeats)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Tuple[int, int, int, int], Optional[TestTimeAugmentation]]:
        coord = self.coords[idx]
        tile_size_wh = (coord[3] - coord[2]) * self.tiling_downsample_factor, (coord[1] - coord[0]) * self.tiling_downsample_factor

        new_location = (
            int(int(self.read_origin[0]) + coord[2] * self.level_downsampling),
            int(int(self.read_origin[1]) + coord[0] * self.level_downsampling),
        )
        tile = np.array(self.wsi.read_region(new_location, self.level, tile_size_wh))
        # tile = cv2.resize(tile, (self.tile_size, self.tile_size), interpolation=cv2.INTER_LINEAR)

        if tile.shape[2] == 4:
            tile[:, :, 3] = 255
            tile = cv2.cvtColor(tile, cv2.COLOR_RGBA2RGB)

        if tile.shape[0] != self.tile_size:
            pad = self.tile_size - tile.shape[0]
            tile = np.pad(tile, ((0, pad), (0, 0), (0, 0)), mode="constant", constant_values=255)
        if tile.shape[1] != self.tile_size:
            pad = self.tile_size - tile.shape[1]
            tile = np.pad(tile, ((0, 0), (0, pad), (0, 0)), mode="constant", constant_values=255)

        tile = Image.fromarray(tile)
        if self.apply_tta:
            augmentation = TestTimeAugmentation(
                self.angles[idx],
                self.flips[idx],
                self.color_jitter,
                self.noise,
                self.blur,
                self.gamma,
            )
            tile = augmentation(tile)
        else:
            augmentation = None

        tile = F.to_tensor(tile)
        return tile, coord, augmentation

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def get_gt_mask_tiles(
        self, mask_gt: np.ndarray, coords: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        return np.array([mask_gt[x:x_end, y:y_end] for x, x_end, y, y_end in coords])


# ---------------------------------------------------------------------------
# BaseTileDataset — unchanged from original
# ---------------------------------------------------------------------------

class BaseTileDataset(Dataset):
    def __init__(
        self,
        list_of_masks: list,
        wsi_root: Union[str, List[str]],
        resolution: float = 2.0,
        tile_size: int = 384,
        step_size: int = 1,
        label2id: dict = {"Background": 0},
        num_classes: int = 2,
        dataset_save_path: str = None,
        data_augs: Dict[str, Union[str, List[str]]] = None,
        print_function=print,
        infer_mode: bool = False,
    ):
        super().__init__()

        self.list_of_masks = list_of_masks
        self.wsi_root = self._get_wsi_roots(wsi_root)
        self.resolution = resolution
        self.tile_size = tile_size
        self.step_size = step_size
        self.classes = list(label2id.values())
        self.categories_to_use = list(label2id.keys())
        self.label2id = label2id
        self.ignored_categories = []
        self.num_classes = num_classes
        self.img_transform = transforms.Compose(
            [
                transforms.Resize((self.tile_size, self.tile_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.mask_transform = transforms.Compose(
            [transforms.Resize((self.tile_size, self.tile_size))]
        )
        self.sampling_weights = None
        self.indices = None
        self.current_process = None
        self.level = 0
        self.level_downsampling = 0
        self.working_resolution = 0
        self.slide_cache = {}
        self.dataset_save_path = dataset_save_path
        self.printf = print_function
        os.makedirs(self.dataset_save_path, exist_ok=True)
        self.set_data_augs(data_augs)
        self.set_current_process()
        self.get_level_from_mpp()

    def _get_wsi_roots(self, wsi_root: Union[str, List[str]]) -> List[str]:
        if isinstance(wsi_root, str):
            wsi_root = glob(wsi_root, recursive=True)
        elif isinstance(wsi_root, list):
            wsi_root = [glob(wsi, recursive=True) for wsi in wsi_root]
            wsi_root = [item for sublist in wsi_root for item in sublist]
        have_wsi = [
            any([len(glob(os.path.join(wsi, f"*.{ext}"))) > 0 for ext in ACCEPTED_WSI_TYPES])
            for wsi in wsi_root
        ]
        wsi_root = [wsi for wsi, check in zip(wsi_root, have_wsi) if check]
        if len(wsi_root) == 0:
            raise ValueError("No WSI folders found. Please check the path.")
        return wsi_root

    def _set_color_augmentation(self, color_jitter: str):
        if color_jitter is None:
            self._apply_color_jitter = apply_pil_brightness_augmentation
        elif color_jitter == "hsv":
            self._apply_color_jitter = apply_pil_hsv_augmentation
        elif color_jitter == "hed":
            self._apply_color_jitter = apply_pil_hed_augmentation
        elif color_jitter == "valid":
            self._apply_color_jitter = identity_transformation

    def set_data_augs(self, data_augs: Union[None, Dict[str, Union[str, bool, List[str]]]]) -> None:
        if data_augs is not None:
            for key, value in data_augs.items():
                if key == "color":
                    if value not in ["hsv", "hed", None, "valid"]:
                        raise ValueError(f"Invalid color augmentation: {value}")
                    self._set_color_augmentation(data_augs["color"])
                elif key == "rotation":
                    if not isinstance(value, list) or not all(i in [0, 90, 180, 270] for i in value):
                        raise ValueError(f"Invalid rotation augmentation: {value}")
                elif key == "flip":
                    if not isinstance(value, list) or not all(i in ["h", "v", "None"] for i in value):
                        raise ValueError(f"Invalid flip augmentation: {value}")
                elif key in ("contrast", "noise", "blur"):
                    if not isinstance(value, bool):
                        raise ValueError(f"Invalid {key} augmentation: {value}. Must be bool.")
                else:
                    raise ValueError(f"Invalid augmentation key: {key}")
        self.data_augs = data_augs

    def set_current_process(self):
        if not torch.cuda.is_available():
            self.current_process = "main"
        elif torch.cuda.device_count() == 1:
            self.current_process = "main"
        elif torch.distributed.get_rank() == 0:
            self.current_process = "main"
        else:
            self.current_process = "worker"

    def get_level_from_mpp(self):
        filename = os.path.splitext(os.path.basename(self.list_of_masks[0]))[0]
        slide = self._get_wsi(filename)
        mpp = np.array(slide.properties[openslide.PROPERTY_NAME_MPP_X], dtype=np.float32)
        downsampling = self.resolution / mpp
        self.level = slide.get_best_level_for_downsample(downsampling)
        self.level_downsampling = int(slide.level_downsamples[self.level])
        self.working_resolution = round(mpp * self.level_downsampling)
        if self.working_resolution != self.resolution:
            self.printf(
                f"Warning: requested {self.resolution} mpp not available at level {self.level}. "
                f"Using {self.working_resolution} mpp."
            )

    def _get_wsi(self, filename: str, return_openslide: bool = True):
        wsi_exists, wsi_path = check_wsi_exists_all_formats(filename, self.wsi_root)
        assert wsi_exists, f"WSI {filename} not found in {self.wsi_root}."
        return openslide.open_slide(wsi_path) if return_openslide else wsi_path

    def apply_transforms(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.fromarray(image.astype(np.uint8))
        mask = Image.fromarray(mask)
        if self.data_augs is not None:
            if self.data_augs.get("rotation"):
                degree = random.choice(self.data_augs["rotation"])
                image, mask = image.rotate(degree), mask.rotate(degree)
            if self.data_augs.get("flip"):
                flip = random.choice(self.data_augs["flip"])
                if flip == "h":
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                elif flip == "v":
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                    mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            if self.data_augs.get("color"):
                image = self._apply_color_jitter(image)
            if self.data_augs.get("noise"):
                image = apply_pil_additive_noise(image)
            if self.data_augs.get("blur"):
                image = apply_pil_gaussian_blur(image)
            if self.data_augs.get("contrast"):
                image = apply_pil_gamma_correction(image)
        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(
                torch.from_numpy(np.array(mask)).unsqueeze(0)
            ).squeeze().long()
        return image, mask

    def _load_geojson(self, path_to_geojson: str) -> dict:
        with open(path_to_geojson) as f:
            return geojson.load(f)

    def __len__(self):
        return len(self.indices)

    def get_weights_and_coords(self):
        raise NotImplementedError
    


class BaseTileDataset(Dataset):
    def __init__(
        self,
        list_of_masks:list,
        wsi_root:Union[str, List[str]],
        resolution:float=2.0,
        tile_size:int=384,
        step_size:int=1,
        label2id:dict={'Background':0},
        num_classes:int=2,
        dataset_save_path:str=None,
        data_augs:Dict[str, Union[str, List[str]]]=None,
        print_function=print,
        infer_mode:bool=False,):
        super().__init__()

        self.list_of_masks = list_of_masks
        self.wsi_root = self._get_wsi_roots(wsi_root)
        self.resolution = resolution
        self.tile_size = tile_size
        self.step_size = step_size
        self.classes = list(label2id.values())
        self.categories_to_use = list(label2id.keys())
        self.label2id = label2id
        self.ignored_categories = []
        self.num_classes = num_classes
        self.img_transform = transforms.Compose([
            transforms.Resize((self.tile_size, self.tile_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.tile_size, self.tile_size)),
        ])
        self.sampling_weights = None
        self.indices = None
        self.current_process = None
        self.level = 0
        self.level_downsampling = 0
        self.working_resolution = 0
        self.slide_cache = {}  # cache slide objects to reduce file opens overhead
        self.dataset_save_path = dataset_save_path
        self.printf = print_function
        os.makedirs(self.dataset_save_path, exist_ok=True)
        # initialization functions
        self.set_data_augs(data_augs)
        self.set_current_process()
        self.get_level_from_mpp()
        # if not infer_mode:
        #     self.get_weights_and_coords()

    def _get_wsi_roots(self, wsi_root:Union[str, List[str]]) -> List[str]:
        if isinstance(wsi_root, str):
            wsi_root = glob(wsi_root, recursive=True)
        elif isinstance(wsi_root, list):
            wsi_root = [glob(wsi, recursive=True) for wsi in wsi_root]
            wsi_root = [item for sublist in wsi_root for item in sublist]

        have_wsi = [any([len(glob(os.path.join(wsi, f'*.{ext}'))) > 0 for ext in ACCEPTED_WSI_TYPES]) for wsi in wsi_root]
        wsi_root = [wsi for wsi, check in zip(wsi_root, have_wsi) if check]
        if len(wsi_root) == 0:
            raise ValueError("No WSI folders found. Please check the path.")
        return wsi_root

    def _set_color_augmentation(self, color_jitter: str):
        if color_jitter is None:
            self._apply_color_jitter = apply_pil_brightness_augmentation
        elif color_jitter == 'hsv':
            self._apply_color_jitter = apply_pil_hsv_augmentation
        elif color_jitter == 'hed':
            self._apply_color_jitter = apply_pil_hed_augmentation
        elif color_jitter == 'valid':
            self._apply_color_jitter = identity_transformation
    
    def set_data_augs(self, data_augs:Union[None, Dict[str, Union[str, bool, List[str]]]]) -> None:
        if data_augs is not None:
            for key, value in data_augs.items():
                if key == 'color':
                    if value not in ['hsv', 'hed', None, 'valid']:
                        raise ValueError(f"Invalid color augmentation value: {value}. Must be one of ['hsv', 'hed', None]")
                    self._set_color_augmentation(data_augs['color'])
                elif key == 'rotation':
                    if not isinstance(value, list):
                        raise ValueError(f"Invalid rotation augmentation value: {value}. Must be a list of integers")
                    if not all([i in [0, 90, 180, 270] for i in value]):
                        raise ValueError(f"Invalid rotation augmentation value: {value}. Must be a list of integers from [0, 90, 180, 270]")
                elif key == 'flip':
                    if not isinstance(value, list):
                        raise ValueError(f"Invalid flip augmentation value: {value}. Must be a tuple of strings")
                    if not all([i in ['h', 'v', 'None'] for i in value]):
                        raise ValueError(f"Invalid flip augmentation value: {value}. Must be a tuple of strings from ['h', 'v', 'None']")
                elif key == 'contrast':
                    if not isinstance(value, bool):
                        raise ValueError(f"Invalid contrast augmentation value: {value}. Must be a boolean")
                elif key == 'noise':
                    if not isinstance(value, bool):
                        raise ValueError(f"Invalid noise augmentation value: {value}. Must be a boolean")
                elif key == 'blur':
                    if not isinstance(value, bool):
                        raise ValueError(f"Invalid blur augmentation value: {value}. Must be a boolean")
                else:
                    raise ValueError(f"Invalid data augmentation key: {key}. Must be one of ['color', 'rotation', 'flip', 'contrast', 'noise', 'blur']")
        self.data_augs = data_augs

    def set_current_process(self):
        # main if not using gpu
        if not torch.cuda.is_available(): self.current_process='main'
        # or running on single gpu
        elif torch.cuda.is_available() and torch.cuda.device_count() == 1: self.current_process='main'
        # or running on multiple gpus and main process
        elif torch.cuda.is_available() and torch.cuda.device_count() > 1 and torch.distributed.get_rank() == 0: self.current_process='main'
        else: self.current_process='worker'


    def get_level_from_mpp(self):
        # get level from mpp
        filename = os.path.splitext(os.path.basename(self.list_of_masks[0]))[0]
        # define wsi root somewhere else, or another way of getting the wsi
        
        slide = self._get_wsi(filename)
        # get mpp from openslide properties
        mpp = slide.properties[openslide.PROPERTY_NAME_MPP_X]
        # convert to float numpy array
        mpp = np.array(mpp, dtype=np.float32)
        # get downsampling factor
        downsampling = self.resolution / mpp
        # get best level for downsampling
        self.level = slide.get_best_level_for_downsample(downsampling)
        self.level_downsampling = int(slide.level_downsamples[self.level])
        self.working_resolution = round(mpp * self.level_downsampling)
        if self.working_resolution != self.resolution:
            self.printf(f"Warning: Input resolution of {self.resolution} mpp is not available for selected WSI level ({self.level}). Using {self.working_resolution} mpp instead.")
    
    def _get_wsi(self, filename:str, return_openslide:bool=True) -> Union[openslide.OpenSlide, openslide.ImageSlide]:
        wsi_exists, wsi_path = check_wsi_exists_all_formats(filename, self.wsi_root)
        assert wsi_exists, f"WSI file {filename} not found in {self.wsi_root}. Please check the path."
        return openslide.open_slide(wsi_path) if return_openslide else wsi_path

    def _get_wsi_path(self, filename:str) -> str:
        # define wsi root somewhere else, or another way of getting the wsi
        for wsi_root_i in self.wsi_root:
            wsi_path = os.path.join(wsi_root_i, filename + ".mrxs")
            if os.path.exists(wsi_path):
                return wsi_path
        raise FileNotFoundError(f"WSI file not found for {filename}")

    
    
    def apply_transforms(self, image:np.ndarray, mask:np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        # image transformations

        image = Image.fromarray(image.astype(np.uint8))
        mask = Image.fromarray(mask)
        if self.data_augs is not None:
            if self.data_augs['rotation'] is not None:
                degree = random.choice(self.data_augs['rotation'])
                image = image.rotate(degree)
                mask = mask.rotate(degree)
            if self.data_augs['flip'] is not None:
                flip = random.choice(self.data_augs['flip'])
                if flip == 'h':
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                elif flip == 'v':
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                    mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            if self.data_augs['color'] is not None:
                image = self._apply_color_jitter(image)
            if self.data_augs['noise']:
                image = apply_pil_additive_noise(image)
            if self.data_augs['blur']:
                image = apply_pil_gaussian_blur(image)
            if self.data_augs['contrast']:
                image = apply_pil_gamma_correction(image)
        
        if self.img_transform:
            image = self.img_transform(image)  
        if self.mask_transform:    
            mask = self.mask_transform(torch.from_numpy(np.array(mask)).unsqueeze(0)).squeeze().long()

        return image, mask     

    def _load_geojson(self, path_to_geojson:str) -> dict:
        with open(path_to_geojson) as f:
            data = geojson.load(f)
        return data
    
    def __len__(self):
        return len(self.indices)  
    

    def get_weights_and_coords(self):
        raise NotImplementedError
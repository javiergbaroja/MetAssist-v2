"""
dataset_seg_base_chunked.py  — optimised version
=================================================

Changes applied to BaseSlideDataset vs original
-------------------------------------------------

[B3]  Fork-safe OpenSlide handle  (correctness + perf)
      _prepare_slide() stores the resolved WSI path in self._wsi_path and opens a
      *temporary* handle (self._wsi) only for the duration of __init__ (metadata
      reading + coord extraction).  At the end of __init__ that handle is closed
      and set to None.  A @property wsi then lazily reopens *per process/worker*,
      giving each DataLoader worker its own independent file descriptor.
      openslide is not fork-safe; the original code silently shared a handle
      across workers, causing data corruption or deadlocks on num_workers > 1.

[B5]  GPU-resident chunk buffers  (memory bandwidth)
      self._stitch_device is auto-detected (CUDA when available, CPU otherwise).
      Chunk accumulation buffers (pred + count) are allocated on that device with
      FP16 on GPU (5-class 4096x4096 chunk: 160 MB FP16 vs 320 MB FP32).
      _finalize_chunk computes argmax on _stitch_device and transfers only the
      uint8 result to CPU — 8x less data than moving the full logit volume.

[B6]  Pre-computed stitch metadata  (CPU loop overhead)
      _assign_tiles_to_chunks() now builds self._stitch_meta: one pre-baked tuple
      per tile encoding the prediction-space crop window, the pred_logits slice,
      and all per-chunk intersection slices.  stitch_predictions performs zero
      arithmetic per tile — only index lookup + tensor ops.
      self._tile_to_chunks is removed.

[B8]  Fix tuple-multiply bug in __getitem__  (correctness)
      The original code computed `tile_size_wh * self.tiling_downsample_factor`
      where tile_size_wh is a 2-tuple.  Python tuple multiplication repeats
      elements: (w, h) * 2 = (w, h, w, h).  This passes a 4-tuple to
      openslide.read_region when tiling_downsample_factor > 1, causing either a
      TypeError or wrong reads.  Fixed by explicit scalar multiplication of each
      dimension.

BaseTileDataset
---------------
Unchanged from original.
"""

import openslide
import math
import os
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional
from glob import glob
import random
import geojson

import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

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


# ---------------------------------------------------------------------------
# _stitch_meta element layout (one plain tuple per tile, built once at init)
# ---------------------------------------------------------------------------
#
# (rs, re, cs, ce,           prediction-space crop window
#  t_row_s, t_row_e,         row slice into raw pred_logits tensor
#  t_col_s, t_col_e,         col slice into raw pred_logits tensor
#  chunk_data)               inner tuple; each element:
#                              (chunk_id,
#                               b_rs, b_re, b_cs, b_ce,     chunk-buffer slice
#                               lt_rs, lt_re, lt_cs, lt_ce)  tile-logits slice
#
# Because all values are pre-baked integers, stitch_predictions does zero
# arithmetic per tile: only tuple unpacking and tensor slice-and-add.
# ---------------------------------------------------------------------------


class BaseSlideDataset:
    """
    Dataset for WSI inference with chunked prediction stitching.

    GPU-resident accumulation buffers, fork-safe OpenSlide handles, and fully
    pre-computed stitch metadata.  See module docstring for detailed change log.
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
        self.filter_mask       = filter_mask
        self.tile_size         = tile_size
        self.step_size         = step_size
        self.chunk_size        = chunk_size

        # [B3] Initialise to None; _prepare_slide opens a temporary handle
        self._wsi:      Optional[openslide.OpenSlide] = None
        self._wsi_path: Optional[str]                 = None

        # [B5] Auto-detect stitch device — CUDA preferred for GPU accumulation
        self._stitch_device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Open WSI temporarily for metadata + coord extraction
        self._prepare_slide(wsi_path, resolution)

        self.coords = extract_tile_coords_new(
            slide=self._wsi,                      # temporary init-time handle
            filter_mask=filter_mask,
            img_dim=self.original_shape,
            img_dim_padded=self.padded_shape,
            tile_size=self.tile_size * self.tiling_downsample_factor,
            step_size=self.step_size * self.tiling_downsample_factor,
            read_origin=self.read_origin,
            level=self.level,
        )
        self.coords = sorted(
            self.coords, key=lambda x: (x[0] // chunk_size, x[2] // chunk_size)
        )
        self.num_augs = len(rotations) * (len(flips) + 1) if apply_tta else 1
        self.coords = (
            [c for c in self.coords for _ in range(self.num_augs)]
            if self.num_augs > 1
            else self.coords
        )

        # [B3] Close init-time handle; each worker/main-process reopens via .wsi
        self._wsi.close()
        self._wsi = None

        self.flips_values = [f for f in ("h", "v") if f in flips]
        self.rotations    = rotations
        self.color_jitter = color_jitter
        self.apply_tta    = apply_tta
        self.noise        = noise
        self.blur         = blur
        self.gamma        = gamma
        self.angles: List[int] = []
        self.flips:  List[str] = []
        self.prepare_augmentation_series()

        # --- crop_size: clamp to valid range ---
        max_crop = max((tile_size - step_size) // 2, 0)
        if step_size >= tile_size and crop_size > 0:
            import warnings
            warnings.warn(
                f"crop_size={crop_size} requested but step_size={step_size} >= "
                f"tile_size={tile_size}: no overlap. Setting crop_size=0.",
                UserWarning,
            )
        self.crop_size = min(crop_size, max_crop)

        # --- Chunked stitching state ---
        self._pred_H       = self.padded_shape[0] // self.tiling_downsample_factor
        self._pred_W       = self.padded_shape[1] // self.tiling_downsample_factor
        self._n_chunks_row = math.ceil(self._pred_H / self.chunk_size)
        self._n_chunks_col = math.ceil(self._pred_W / self.chunk_size)

        # [B6] Pre-computed stitch metadata (one tuple per tile)
        self._stitch_meta:     List[tuple]                        = []
        self._chunk_remaining: Dict[Tuple[int, int], int]         = defaultdict(int)
        self._assign_tiles_to_chunks()   # populates _stitch_meta + _chunk_remaining

        # Lazily allocated chunk buffers: chunk_id -> {'pred': Tensor, 'count': Tensor}
        self._chunk_buffers: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}

        # Final segmentation output (written incrementally as chunks finalise)
        self._final_prediction = np.zeros((self._pred_H, self._pred_W), dtype=np.uint8)

        # Global tile index — maps batch position to _stitch_meta entries
        self._tile_idx = 0

        # Number of model output classes — resolved on first stitch call
        self._num_classes: Optional[int] = None

    # ------------------------------------------------------------------
    # [B3] Fork-safe lazy WSI handle
    # ------------------------------------------------------------------

    @property
    def wsi(self) -> openslide.OpenSlide:
        """
        Lazy, per-process WSI handle.

        The init-time handle is closed at the end of __init__, leaving
        self._wsi = None in every forked DataLoader worker.  Each worker
        (and the main process for num_workers=0) transparently reopens the
        file on the first __getitem__ call.  This is the minimal change
        required to make openslide access fork-safe.
        """
        if self._wsi is None:
            self._wsi = openslide.open_slide(self._wsi_path)
        return self._wsi

    # ------------------------------------------------------------------
    # Slide preparation
    # ------------------------------------------------------------------

    def _prepare_slide(self, wsi_path: str, resolution: float) -> None:
        """
        Resolve local path (TMPDIR copy if applicable), open a temporary
        handle in self._wsi, and populate all WSI geometry attributes.

        The handle must remain open until coord extraction completes in
        __init__.  __init__ closes it immediately afterwards.
        """
        file_type = os.path.splitext(wsi_path)[1]
        local_wsi_path = wsi_path

        if os.getenv("TMPDIR") is not None:
            local_wsi_path = os.path.join(
                os.getenv("TMPDIR"), os.path.basename(wsi_path)
            )
            if not os.path.exists(local_wsi_path):
                shutil.copy(wsi_path, local_wsi_path)

            if file_type == ".mrxs":
                wsi_dir       = os.path.splitext(wsi_path)[0]
                local_wsi_dir = os.path.join(
                    os.getenv("TMPDIR"),
                    os.path.splitext(os.path.basename(wsi_path))[0],
                )
                if not os.path.exists(local_wsi_dir):
                    shutil.copytree(wsi_dir, local_wsi_dir)

        # [B3] Persist resolved path for lazy per-worker reopening
        self._wsi_path = local_wsi_path
        # Open temporary init-time handle (closed at end of __init__)
        self._wsi = openslide.open_slide(local_wsi_path)

        (
            self.level,
            self.level_downsampling,
            self.exact_resolution,
            self.tiling_downsample_factor,
            self.original_shape,
            self.read_origin,
        ) = prepare_read_from_slide(self._wsi, resolution, file_type=file_type)

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
    # [B6] Pre-computed stitch metadata
    # ------------------------------------------------------------------

    def _cropped_pred_coords(
        self, row_start_orig: int, col_start_orig: int
    ) -> Tuple[int, int, int, int]:
        """
        Convert raw tile origin to its prediction-space crop window.

        Called only during _assign_tiles_to_chunks (init-time); the results are
        cached in _stitch_meta and never recomputed during inference.
        """
        rs = row_start_orig // self.tiling_downsample_factor
        cs = col_start_orig // self.tiling_downsample_factor
        re = rs + self.tile_size
        ce = cs + self.tile_size

        crop = self.crop_size
        rs = rs + crop if rs != 0          else rs
        re = re - crop if re != self._pred_H else re
        cs = cs + crop if cs != 0          else cs
        ce = ce - crop if ce != self._pred_W else ce

        return rs, re, cs, ce

    def _assign_tiles_to_chunks(self) -> None:
        """
        [B6] Build self._stitch_meta — one pre-baked tuple per tile.

        Replaces self._tile_to_chunks entirely.  Every intersection and
        slice index is computed once here so that stitch_predictions can
        execute with zero per-tile arithmetic.

        _stitch_meta element layout
        ---------------------------
        (rs, re, cs, ce,
         t_row_s, t_row_e, t_col_s, t_col_e,
         chunk_data)

        chunk_data element
        ------------------
        (chunk_id,
         b_rs, b_re, b_cs, b_ce,       # slice within chunk accumulation buffer
         lt_rs, lt_re, lt_cs, lt_ce)    # slice within already-cropped tile logits
        """
        for coord in self.coords:
            row_start_orig, _, col_start_orig, _ = coord

            # Prediction-space crop window
            rs, re, cs, ce = self._cropped_pred_coords(row_start_orig, col_start_orig)

            # Slice into raw (uncropped) pred_logits tensor that survives crop
            raw_rs  = row_start_orig // self.tiling_downsample_factor
            raw_cs  = col_start_orig // self.tiling_downsample_factor
            t_row_s = rs - raw_rs     # 0 at image border, else crop_size
            t_row_e = re - raw_rs     # tile_size or tile_size - crop_size
            t_col_s = cs - raw_cs
            t_col_e = ce - raw_cs

            # Chunks overlapped by this tile's crop window
            cr_start = rs // self.chunk_size
            cr_end   = min((re - 1) // self.chunk_size, self._n_chunks_row - 1)
            cc_start = cs // self.chunk_size
            cc_end   = min((ce - 1) // self.chunk_size, self._n_chunks_col - 1)

            chunk_data: List[Tuple] = []
            for cr in range(cr_start, cr_end + 1):
                for cc in range(cc_start, cc_end + 1):
                    chunk_id            = (cr, cc)
                    c_rs, c_re, c_cs, c_ce = self._chunk_bounds(cr, cc)

                    # Intersection of tile crop window with chunk (prediction space)
                    i_rs = max(rs, c_rs);  i_re = min(re, c_re)
                    i_cs = max(cs, c_cs);  i_ce = min(ce, c_ce)

                    # Slice within chunk accumulation buffer
                    b_rs = i_rs - c_rs;  b_re = i_re - c_rs
                    b_cs = i_cs - c_cs;  b_ce = i_ce - c_cs

                    # Slice within already-cropped tile logits
                    lt_rs = i_rs - rs;  lt_re = i_re - rs
                    lt_cs = i_cs - cs;  lt_ce = i_ce - cs

                    chunk_data.append(
                        (chunk_id,
                         b_rs, b_re, b_cs, b_ce,
                         lt_rs, lt_re, lt_cs, lt_ce)
                    )
                    self._chunk_remaining[chunk_id] += 1

            self._stitch_meta.append(
                (rs, re, cs, ce,
                 t_row_s, t_row_e, t_col_s, t_col_e,
                 tuple(chunk_data))
            )

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
        """
        [B5] Lazily allocate accumulation buffers on self._stitch_device.

        FP16 on GPU: a 5-class 4096x4096 chunk requires 160 MB
                     (vs 320 MB FP32) — safe in 24 GB VRAM even with several
                     simultaneously active chunks.
        FP32 on CPU: avoids accumulated precision loss over long sums.
        """
        cr, cc = chunk_id
        rs, re, cs, ce = self._chunk_bounds(cr, cc)
        h, w = re - rs, ce - cs
        buf_dtype = (
            torch.float16 if self._stitch_device.type == "cuda" else torch.float32
        )
        self._chunk_buffers[chunk_id] = {
            "pred":  torch.zeros(
                (self._num_classes, h, w),
                dtype=buf_dtype, device=self._stitch_device,
            ),
            "count": torch.zeros(
                (h, w),
                dtype=buf_dtype, device=self._stitch_device,
            ),
        }

    def _finalize_chunk(self, chunk_id: Tuple[int, int]) -> None:
        """
        [B5] Average accumulated logits, compute argmax, write uint8 to output.

        The argmax is computed on _stitch_device (GPU when available).  Only the
        resulting uint8 map is moved to the CPU numpy output array — 8x less data
        than transferring the full FP logit volume per chunk.
        """
        buf = self._chunk_buffers.pop(chunk_id)
        cr, cc = chunk_id
        rs, re, cs, ce = self._chunk_bounds(cr, cc)

        count    = buf["count"].clamp(min=1.0).unsqueeze(0)   # (1, H, W)
        averaged = buf["pred"] / count                         # (C, H, W)

        # Argmax on device, then minimal uint8 D2H (noop when already CPU)
        self._final_prediction[rs:re, cs:ce] = (
            averaged.argmax(dim=0).byte().cpu().numpy()
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

        tile_predictions: (B, C, H, W) — may arrive on any device / dtype.
            Moved to self._stitch_device with the correct buffer dtype before
            accumulation.  When the caller already supplies a tensor on the
            correct device (e.g. inference.py with GPU stitch), no transfer
            occurs.

        [B6] Zero arithmetic per tile: all slice indices are read directly from
        self._stitch_meta (precomputed at init-time).
        """
        if self._num_classes is None:
            self._num_classes = tile_predictions.shape[1]

        buf_dtype = (
            torch.float16 if self._stitch_device.type == "cuda" else torch.float32
        )
        if (tile_predictions.device != self._stitch_device
                or tile_predictions.dtype != buf_dtype):
            tile_predictions = tile_predictions.to(
                device=self._stitch_device, dtype=buf_dtype, non_blocking=True
            )

        for pred_logits, augmentation in zip(tile_predictions, augmentations):

            # --- 1. Inverse TTA (no-op when apply_tta=False) ---
            if augmentation is not None:
                pred_logits = augmentation.reverse(pred_logits.unsqueeze(0)).squeeze(0)

            # --- 2. Unpack pre-computed metadata — zero arithmetic ---
            (rs, re, cs, ce,
             t_row_s, t_row_e, t_col_s, t_col_e,
             chunk_data) = self._stitch_meta[self._tile_idx]
            self._tile_idx += 1

            # Crop the raw logits to the surviving prediction window
            cropped = pred_logits[:, t_row_s:t_row_e, t_col_s:t_col_e]

            # --- 3. Accumulate into every overlapping chunk ---
            for (chunk_id,
                 b_rs, b_re, b_cs, b_ce,
                 lt_rs, lt_re, lt_cs, lt_ce) in chunk_data:

                if chunk_id not in self._chunk_buffers:
                    self._allocate_chunk(chunk_id)

                buf = self._chunk_buffers[chunk_id]
                buf["pred"][:, b_rs:b_re, b_cs:b_ce]  += cropped[:, lt_rs:lt_re, lt_cs:lt_ce]
                buf["count"][   b_rs:b_re, b_cs:b_ce] += 1

                # --- 4. Finalise chunk once all contributing tiles are done ---
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

        Under normal operation all chunks are finalised during stitch_predictions.
        Any residual chunks (e.g. interrupted inference) are resolved here as a
        safety net.
        """
        if return_probs:
            raise ValueError(
                "return_probs=True is not supported with chunked stitching: "
                "chunk buffers are freed incrementally during stitch_predictions. "
                "If you need full probability maps, set chunk_size to cover the "
                "entire WSI or accumulate probabilities externally."
            )
        for chunk_id in list(self._chunk_buffers.keys()):
            self._finalize_chunk(chunk_id)
        return self._final_prediction, None

    # ------------------------------------------------------------------
    # Augmentation helpers
    # ------------------------------------------------------------------

    def prepare_augmentation_series(self) -> None:
        """Build per-tile angle and flip lists for TTA."""
        existing_flips = [0] + self.flips_values
        angles    = [a for a in self.rotations for _ in existing_flips]
        flips     = existing_flips * len(self.rotations)
        n_repeats = len(self.coords) / len(angles)
        self.angles = angles * int(n_repeats)
        self.flips  = flips  * int(n_repeats)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Tuple[int, int, int, int], Optional[TestTimeAugmentation]]:
        coord = self.coords[idx]

        new_location = (
            int(int(self.read_origin[0]) + coord[2] * self.level_downsampling),
            int(int(self.read_origin[1]) + coord[0] * self.level_downsampling),
        )

        # [B8] Fix tuple-multiply bug.
        # Original: `tile_size_wh * self.tiling_downsample_factor` where
        # tile_size_wh was a 2-tuple, producing a 4-tuple for factor > 1.
        # (w, h) * 2  ==>  (w, h, w, h)  — wrong number of dimensions.
        read_w = (coord[3] - coord[2]) * self.tiling_downsample_factor
        read_h = (coord[1] - coord[0]) * self.tiling_downsample_factor

        # [B3] self.wsi is the lazy per-worker property — fork-safe
        tile = np.array(self.wsi.read_region(new_location, self.level, (read_w, read_h)))

        if tile.shape[2] == 4:
            tile[:, :, 3] = 255
            tile = cv2.cvtColor(tile, cv2.COLOR_RGBA2RGB)

        if tile.shape[0] != self.tile_size:
            pad  = self.tile_size - tile.shape[0]
            tile = np.pad(tile, ((0, pad), (0, 0), (0, 0)),
                          mode="constant", constant_values=255)
        if tile.shape[1] != self.tile_size:
            pad  = self.tile_size - tile.shape[1]
            tile = np.pad(tile, ((0, 0), (0, pad), (0, 0)),
                          mode="constant", constant_values=255)

        tile = Image.fromarray(tile)
        if self.apply_tta:
            augmentation = TestTimeAugmentation(
                self.angles[idx], self.flips[idx],
                self.color_jitter, self.noise, self.blur, self.gamma,
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

        self.list_of_masks      = list_of_masks
        self.wsi_root           = self._get_wsi_roots(wsi_root)
        self.resolution         = resolution
        self.tile_size          = tile_size
        self.step_size          = step_size
        self.classes            = list(label2id.values())
        self.categories_to_use  = list(label2id.keys())
        self.label2id           = label2id
        self.ignored_categories = []
        self.num_classes        = num_classes
        self.img_transform = transforms.Compose([
            transforms.Resize((self.tile_size, self.tile_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.tile_size, self.tile_size)),
        ])
        self.sampling_weights = None
        self.indices          = None
        self.current_process  = None
        self.level            = 0
        self.level_downsampling = 0
        self.working_resolution = 0
        self.slide_cache        = {}
        self.dataset_save_path  = dataset_save_path
        self.printf             = print_function
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
            any(len(glob(os.path.join(wsi, f"*.{ext}"))) > 0 for ext in ACCEPTED_WSI_TYPES)
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
        elif torch.cuda.device_count() > 1 and torch.distributed.get_rank() == 0:
            self.current_process = "main"
        else:
            self.current_process = "worker"

    def get_level_from_mpp(self):
        filename = os.path.splitext(os.path.basename(self.list_of_masks[0]))[0]
        slide    = self._get_wsi(filename)
        mpp      = np.array(slide.properties[openslide.PROPERTY_NAME_MPP_X], dtype=np.float32)
        downsampling = self.resolution / mpp
        self.level              = slide.get_best_level_for_downsample(downsampling)
        self.level_downsampling = int(slide.level_downsamples[self.level])
        self.working_resolution = round(mpp * self.level_downsampling)
        if self.working_resolution != self.resolution:
            self.printf(
                f"Warning: Input resolution of {self.resolution} mpp is not available for "
                f"selected WSI level ({self.level}). Using {self.working_resolution} mpp instead."
            )

    def _get_wsi(self, filename: str, return_openslide: bool = True):
        wsi_exists, wsi_path = check_wsi_exists_all_formats(filename, self.wsi_root)
        assert wsi_exists, f"WSI file {filename} not found in {self.wsi_root}. Please check the path."
        return openslide.open_slide(wsi_path) if return_openslide else wsi_path

    def _get_wsi_path(self, filename: str) -> str:
        for wsi_root_i in self.wsi_root:
            wsi_path = os.path.join(wsi_root_i, filename + ".mrxs")
            if os.path.exists(wsi_path):
                return wsi_path
        raise FileNotFoundError(f"WSI file not found for {filename}")

    def apply_transforms(self, image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.fromarray(image.astype(np.uint8))
        mask  = Image.fromarray(mask)
        if self.data_augs is not None:
            if self.data_augs["rotation"] is not None:
                degree = random.choice(self.data_augs["rotation"])
                image  = image.rotate(degree)
                mask   = mask.rotate(degree)
            if self.data_augs["flip"] is not None:
                flip = random.choice(self.data_augs["flip"])
                if flip == "h":
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    mask  = mask.transpose(Image.FLIP_LEFT_RIGHT)
                elif flip == "v":
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                    mask  = mask.transpose(Image.FLIP_TOP_BOTTOM)
            if self.data_augs["color"] is not None:
                image = self._apply_color_jitter(image)
            if self.data_augs["noise"]:
                image = apply_pil_additive_noise(image)
            if self.data_augs["blur"]:
                image = apply_pil_gaussian_blur(image)
            if self.data_augs["contrast"]:
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
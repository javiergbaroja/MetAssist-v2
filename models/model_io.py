"""
model_io.py  — optimised version
==================================

Changes vs original
--------------------

[B1]  Batched interpolation — no per-sample loop
      _process_mask2former_output previously iterated over each item in the batch
      and called F.interpolate individually.  For a homogeneous batch (all tiles the
      same size, which is always the case in infer_wsi), this has been replaced with
      a single batched call on the full (B, Q, H, W) tensor.  No torch.stack at the
      end — the result is already (B, C, H, W) from the einsum.

[B10] Fused double interpolation
      The original code applied two sequential F.interpolate calls:
        1. masks_queries_logits  →  384×384   (fixed intermediate size)
        2. segmentation (B,C,384,384)  →  target_size
      These are replaced by a single interpolation of masks_queries_logits directly
      to target_size, then the einsum produces the output at the final resolution
      with no second resize.  This eliminates the 384×384 intermediate allocation
      (≈ B × Q × 384 × 384 × dtype bytes per batch) and one kernel launch.
      The heterogeneous-size fallback path preserves the original 384 intermediate
      for safety and backward compatibility.

[B11] Removed unsafe squeeze/unsqueeze in post_process_output
      The original code called .squeeze() with no dimension argument, which
      collapses any size-1 dimension — including the class axis (C=1) or a spatial
      axis — not just the batch axis.  For the Mask2Former path this is now
      unnecessary because _process_mask2former_output guarantees the correct shape.
      For the UNet/SETR/UNETR path, the blind squeeze is replaced with a
      dimension-specific squeeze(0) that only removes the batch dim when B=1,
      never touching C, H, or W.

[API] target_sizes now accepts Union[Tuple[int,int], List[Tuple[int,int]]]
      infer_wsi passes a single (H, W) tuple (all tiles are the same size).
      evaluate_wsi_tiles passes a list of per-tile tuples.  Both are handled:
        • single (H, W) tuple  →  fast batched path (B1 + B10 applied)
        • list of same-size tuples  →  same fast path (shared size extracted)
        • list of different-size tuples  →  heterogeneous fallback (original loop)
"""

import importlib
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from torchvision.transforms import functional

from data.augmentations import TestTimeAugmentation

ACCEPTED_MODEL_CLASSES = [
    "UNet", "SETR", "UNETR", "Mask2FormerForUniversalSegmentation", "VISTAPATH"
]


# ---------------------------------------------------------------------------
# Model utility functions — unchanged
# ---------------------------------------------------------------------------

def get_model_class_from_checkpoint(checkpoint_path: str) -> str:
    for model_class in ACCEPTED_MODEL_CLASSES:
        if model_class in checkpoint_path:
            return model_class
    return "Mask2FormerForUniversalSegmentation"


def get_model_class_from_model(model) -> str:
    for model_class in ACCEPTED_MODEL_CLASSES:
        if model.__class__.__name__ == model_class:
            return model_class
    raise ValueError(
        f"Model class {model.__class__.__name__} not in accepted model classes: "
        f"{ACCEPTED_MODEL_CLASSES}"
    )


def get_model_funcs(model_class: str) -> Tuple:
    if not isinstance(model_class, str):
        model_class = get_model_class_from_model(model_class)
    elif model_class not in ACCEPTED_MODEL_CLASSES:
        raise ValueError(
            f"Model class {model_class} not in accepted model classes: {ACCEPTED_MODEL_CLASSES}"
        )
    module_name = (
        model_class.lower()
        if model_class != "Mask2FormerForUniversalSegmentation"
        else "mask2former"
    )
    module             = importlib.import_module(f"models.architectures.{module_name}")
    create_img_processor = getattr(module, "create_img_processor", None)
    create_model       = getattr(module, "create_model", None)
    train_collator     = getattr(module, "TrainCollator", None)
    return create_img_processor, create_model, train_collator


# ---------------------------------------------------------------------------
# Collators — unchanged
# ---------------------------------------------------------------------------

class InferCollator:
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]

    def __call__(
        self,
        batch: List[Tuple[torch.Tensor, Tuple[int, int, int, int], TestTimeAugmentation]],
    ):
        tiles, coords, augmentations = zip(*batch)
        tiles = torch.stack(tiles)
        if self.normalize:
            tiles = functional.normalize(tiles, mean=self.mean, std=self.std)
        return tiles, list(coords), list(augmentations)


def infer_collate_fn(
    batch: List[Tuple[torch.Tensor, Tuple[int, int, int, int], TestTimeAugmentation]],
) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]], List[TestTimeAugmentation]]:
    tiles, coords, augmentations = zip(*batch)
    tiles = torch.stack(tiles)
    tiles = functional.normalize(
        tiles, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    return tiles, list(coords), list(augmentations)


# ---------------------------------------------------------------------------
# Model loading — unchanged
# ---------------------------------------------------------------------------

def create_mask2former_from_checkpoint(
    checkpoint_path,
    label2id: dict,
    encoder_name: str,
    decoder_model: str,
    out_indices: list = [],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    if not isinstance(checkpoint_path, str):
        return checkpoint_path
    model_class  = get_model_class_from_checkpoint(checkpoint_path)
    _, create_model, _ = get_model_funcs(model_class)
    checkpoint   = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model        = create_model(
        encoder_model=encoder_name,
        decoder_model=decoder_model,
        label2id=label2id,
        id2label={v: k for k, v in label2id.items()},
        out_indices=out_indices,
        freeze_encoder=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch.device(device))
    return model


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _normalize_target_sizes(
    target_sizes: Union[Tuple[int, int], List[Tuple[int, int]]],
    batch_size: int,
) -> Tuple[Optional[Tuple[int, int]], Optional[List[Tuple[int, int]]]]:
    """
    Normalise the target_sizes argument to (shared_size, None) for the fast
    batched path, or (None, list_of_sizes) for the per-tile fallback.

    Accepted inputs
    ---------------
    • (H, W)                        — single shared size (from infer_wsi)
    • [(H, W), (H, W), ...]         — per-tile list (from evaluate_wsi_tiles)
      - all identical  →  shared_size extracted, fast path used
      - heterogeneous  →  fallback path

    Returns
    -------
    shared_size : (H, W) or None
    per_tile    : list of (H, W) or None   (exactly one of the two is not None)
    """
    if target_sizes is None:
        raise ValueError("target_sizes must be provided.")

    # Single (H, W) tuple — new fast-path API from infer_wsi
    if (
        isinstance(target_sizes, tuple)
        and len(target_sizes) == 2
        and isinstance(target_sizes[0], int)
    ):
        return target_sizes, None

    # List of per-tile tuples
    if not isinstance(target_sizes, (list, tuple)):
        raise TypeError(f"Unsupported target_sizes type: {type(target_sizes)}")
    if len(target_sizes) != batch_size:
        raise ValueError(
            f"len(target_sizes)={len(target_sizes)} != batch_size={batch_size}"
        )

    unique = set(target_sizes)
    if len(unique) == 1:
        return target_sizes[0], None       # all same — use fast path
    return None, list(target_sizes)        # heterogeneous — use fallback


def _process_mask2former_output(
    outputs,
    target_sizes: Union[Tuple[int, int], List[Tuple[int, int]]],
    return_logits: bool,
) -> torch.Tensor:
    """
    Post-process Hugging Face Mask2Former outputs.

    Fast path (homogeneous batch)
    ------------------------------
    [B10] masks_queries_logits is interpolated *directly* to target_size,
          skipping the 384×384 intermediate allocation.
    [B1]  A single batched einsum produces (B, C, H, W) at the target
          resolution — no per-sample loop, no torch.stack.

    Fallback path (heterogeneous tile sizes)
    ----------------------------------------
    Original behaviour preserved exactly for backward compatibility.

    Returns
    -------
    return_logits=True  : (B, C, H, W) float
    return_logits=False : (B, H, W)    long
    """
    class_queries_logits = outputs.class_queries_logits   # (B, Q, C+1)
    masks_queries_logits = outputs.masks_queries_logits   # (B, Q, h, w)
    batch_size           = class_queries_logits.shape[0]

    shared_size, per_tile = _normalize_target_sizes(target_sizes, batch_size)

    # Remove the null/void class, compute per-query class probabilities
    masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]  # (B, Q, C)

    # ------------------------------------------------------------------
    # Fast path — all tiles share the same output resolution
    # ------------------------------------------------------------------
    if shared_size is not None:
        # [B10] Single interpolation: query-mask resolution → target directly.
        # Replaces the old two-step: query-res → 384 → target.
        masks_probs = F.interpolate(
            masks_queries_logits,
            size=shared_size,
            mode="bilinear",
            align_corners=False,
        ).sigmoid()                                        # (B, Q, H, W)

        # [B1] Batched einsum — one kernel, no Python loop, no torch.stack.
        segmentation = torch.einsum(
            "bqc, bqhw -> bchw", masks_classes, masks_probs
        )                                                  # (B, C, H, W)

        if return_logits:
            return segmentation                            # (B, C, H, W)
        return segmentation.argmax(dim=1)                  # (B, H, W)

    # ------------------------------------------------------------------
    # Fallback path — heterogeneous tile sizes (backward compat)
    # ------------------------------------------------------------------
    # Preserve original two-step interpolation exactly.
    masks_queries_logits = F.interpolate(
        masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
    )
    masks_probs  = masks_queries_logits.sigmoid()          # (B, Q, 384, 384)
    segmentation = torch.einsum(
        "bqc, bqhw -> bchw", masks_classes, masks_probs
    )                                                      # (B, C, 384, 384)

    processed_outputs = []
    for idx, size in enumerate(per_tile):
        resized = F.interpolate(
            segmentation[idx].unsqueeze(0), size=size, mode="bilinear", align_corners=False
        )[0]
        processed_outputs.append(resized if return_logits else resized.argmax(dim=0))
    return torch.stack(processed_outputs)


def post_process_output(
    outputs,
    target_sizes: Union[Tuple[int, int], List[Tuple[int, int]], None] = None,
    return_logits: bool = False,
) -> torch.Tensor:
    """
    Unified post-processing for all models (Mask2Former, UNet, UNETR, SETR).

    Args
    ----
    outputs      : Raw model output.
    target_sizes : Resize target.  Either a single (H, W) tuple (shared by all
                   tiles in the batch) or a list of per-tile (H, W) tuples.
                   Both forms are accepted; see _normalize_target_sizes.
    return_logits: True → (B, C, H, W) float.  False → (B, H, W) long indices.

    Returns
    -------
    torch.Tensor of shape (B, C, H, W) or (B, H, W).
    """

    # --- Case A: Mask2Former -----------------------------------------------
    if hasattr(outputs, "class_queries_logits") and hasattr(outputs, "masks_queries_logits"):
        # [B11] _process_mask2former_output guarantees correct shape:
        #       return_logits=True  → (B, C, H, W)
        #       return_logits=False → (B, H, W)
        # No squeeze/unsqueeze needed or safe here.
        return _process_mask2former_output(outputs, target_sizes, return_logits).float()

    # --- Case B: UNet / UNETR / SETR (ModelOutput with .preds) ------------
    elif hasattr(outputs, "preds"):
        processed = outputs.preds if return_logits else outputs.y_pred

    # --- Case C: Raw tensor fallback ---------------------------------------
    else:
        processed = outputs

    # --- Shape normalisation for Cases B and C ----------------------------
    # [B11] Use dimension-specific squeeze(0) rather than blind .squeeze().
    # Blind .squeeze() silently collapses class or spatial axes when their
    # size happens to be 1 (e.g. binary segmentation with C=1).
    expected_dim = 4 if return_logits else 3

    if processed.ndim == expected_dim:
        pass                                   # already the right rank
    elif processed.ndim == expected_dim + 1 and processed.shape[0] == 1:
        processed = processed.squeeze(0)       # remove batch dim only
    elif processed.ndim == expected_dim - 1:
        processed = processed.unsqueeze(0)     # restore batch dim
    else:
        raise ValueError(
            f"Unexpected output shape {processed.shape}; "
            f"cannot normalise to {expected_dim}D."
        )

    assert processed.ndim == expected_dim, (
        f"Shape mismatch after normalisation: expected {expected_dim}D, got {processed.shape}"
    )
    return processed.float()
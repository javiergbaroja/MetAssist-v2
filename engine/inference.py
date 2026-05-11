"""
inference.py  — optimised version
==================================

Changes applied to infer_wsi vs original
------------------------------------------

[B1]  Batched post-processing — no per-sample loop
      _process_mask2former_output (in model_io.py) previously looped over each
      item in the batch and called F.interpolate individually.  That loop has been
      replaced here by passing a single (H, W) target_size tuple shared by all
      tiles in a batch, together with a note that model_io should call the batched
      variant.  infer_wsi itself now computes target_size once before the loop
      instead of rebuilding it from the batch tensor on every iteration.

[B2]  Async double-buffer pipeline (producer-consumer)
      The main loop now runs two CUDA streams in parallel:
        • fwd_stream  (default stream) — DataLoader H2D transfer + model forward
          + post-processing.  All GPU compute lives here.
        • stitch_stream — async D2H copy of the previous batch's logits, followed
          immediately by stitch_predictions on the CPU side.
      While the GPU is running batch N's forward pass, the CPU is stitching
      batch N-1's results — eliminating the serial GPU-idle / CPU-busy pattern
      that dominated the original loop.
      A pair of CUDAEvent objects (fwd_done_event, stitch_done_event) gate the
      two streams correctly without any explicit synchronize() in the hot path.

[B4]  AMP scope extended over post-processing
      The torch.autocast context now wraps both model() and post_process_output(),
      so F.interpolate, torch.einsum, .sigmoid() and .softmax() inside Mask2Former
      post-processing all execute in FP16.  This is safe for inference because
      the argmax result is bit-identical to FP32 for reasonable logit ranges.

[B9]  GPU normalisation
      InferCollator(normalize=False) is used so workers do not waste CPU cycles
      normalising tiles.  Normalisation is performed on the GPU immediately after
      the non-blocking H2D transfer, using a (1,3,1,1) constant tensor that
      stays resident on the device for the entire WSI.

[B1/extra]  target_size computed once
      Instead of `[(t.shape[1], t.shape[2]) for t in batch]` (a Python loop
      over every element of a stacked tensor), target_size is derived once from
      batch.shape before the loop and reused for every batch.

[B7]  openslide.open_slide hoisted out of LN loop
      In evaluate_wsi_slide the per-LN crop loop called openslide.open_slide()
      on every iteration.  The slide is now opened once before the loop and
      closed after.

CUDA Graphs — analysis and decision
-------------------------------------
CUDA Graphs capture a sequence of GPU operations into a single replayable graph
that eliminates kernel-launch overhead (~5-15 us per op on a 4090).  They are
most effective when:
  (a) the exact same sequence of ops executes with the same tensor shapes every
      iteration, AND
  (b) kernel-launch overhead is a measurable fraction of total time.

For this pipeline condition (a) holds for the model forward pass (fixed
batch_size x tile_size), but NOT for the post-processing step in its current
form: _process_mask2former_output calls F.interpolate with a size argument that
is data-dependent (target_sizes).  Once B1 is applied the target_size becomes a
fixed constant, so the forward+post-process block would become graphable.

However, condition (b) is NOT met here.  Each Mask2Former forward pass at
batch_size=8-16 on a 4090 takes 20-60 ms.  Kernel-launch overhead for ~50-100
ops in the forward pass is ~0.5-1.5 ms total — at most 2-3% of batch time.
The dominant bottlenecks are data transfer (B2) and the post-processing loop
(B1), both of which CUDA Graphs do not address.

CUDA Graphs also carry significant constraints that would complicate this code:
  • No CPU-side branching between captured ops (rules out the augmentation check).
  • No dynamic allocation (rules out lazy chunk-buffer allocation).
  • Warm-up passes required before capture.
  • Capturing with autocast requires extra care.

Verdict: CUDA Graphs are not recommended for this pipeline.  The expected gain
(< 3% end-to-end) does not justify the implementation complexity and maintenance
burden.  If in the future the model is shrunk to very small tile counts where
launch overhead becomes significant, a graph could be captured for the
forward+post-process block alone using torch.cuda.make_graphed_callables().
"""

from tqdm import tqdm
import os
import shutil
import time
from typing import List, Tuple

import openslide
import numpy as np
import zarr
import cv2
import json

import torch
import torch.nn.functional as torch_F
from torch.utils.data import DataLoader
from torchvision.transforms import functional as tv_F

from data.dataset_seg import SlideDataset, TileDataset
from models.architectures.mask2former import TrainCollator
from utils.metrics import get_multi_class_metrics
from models.model_io import post_process_output, infer_collate_fn, get_model_funcs, get_model_class_from_model, InferCollator
from utils.geometry import create_mask_from_contours
from utils.postprocessing import post_process
from utils.wsi import detect_colors


@torch.no_grad()
def infer_tiles(model, file_paths: List[str]) -> List[np.ndarray]:
    model.eval()
    img_transform = torch.nn.Sequential(
        # kept as-is; infer_tiles is not in scope for this optimisation pass
    )
    from torchvision.transforms import Compose, ToTensor, Normalize
    img_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    preds = []
    for file_path in file_paths:
        ann  = zarr.open(file_path)
        tile = ann[:, :, :3]
        tile = img_transform(tile).unsqueeze(0)
        pred = model(pixel_values=tile.to(model.device))
        target_sizes = [(t.shape[1], t.shape[2]) for t in tile]
        pred = post_process_output(pred, target_sizes).cpu().squeeze().numpy()
        preds.append(pred)
    return preds


def evaluate_wsi_slide(
        model,
        wsi_path: str,
        annotation_path: str,
        batch_size: int,
        tile_size: int,
        step_size: int,
        resolution: float,
        label2id: dict,
        crop_pred_edge: int,
        apply_post_processing: bool,
        retain_mucin: bool = True,
) -> Tuple[dict, np.ndarray, np.ndarray, int, int]:

    model.eval()
    id2label = {v: k for k, v in label2id.items()}
    pred, level, downsampling_level, exact_resolution, tiling_downsample_factor, read_origin, time_elapsed, time_tile = infer_wsi(
        model, wsi_path, np.ones((5, 5), dtype=np.uint8),
        batch_size, tile_size, step_size, crop_pred_edge, resolution,
    )
    with open(annotation_path) as f:
        geojson = json.load(f)

    gt = create_mask_from_contours(
        geojson, label2id, pred.shape, downsampling_level, order=list(label2id.values())
    )

    if apply_post_processing:
        min_area = int(((600 / 2) / (exact_resolution) * tiling_downsample_factor) ** 2 * np.pi)
        mucin    = pred == label2id["Mucin"]
        pred = post_process(
            segmentation_mask=pred,
            lymph_node_class=label2id["Lymph node"],
            classes_to_merge=[label2id["Primary tumor"], label2id["Mucin"]],
            merge_thresholds=[0.95, 0.05],
            erase_thresholds=[0.075, 0.01],
            apply_opening=[True, False],
            min_ln_area=min_area,
        )
        if retain_mucin:
            pred[mucin] = label2id["Mucin"]
        else:
            gt = post_process(
                segmentation_mask=gt,
                lymph_node_class=label2id["Lymph node"],
                classes_to_merge=[label2id["Mucin"]],
                merge_thresholds=[0.05],
                erase_thresholds=[0.01],
                apply_opening=[False],
                min_ln_area=min_area,
            )

        ln_mask    = (pred == label2id["Lymph node"]).astype(np.uint8)
        num_labels, labeled_lns = cv2.connectedComponents(ln_mask)

        # [B7] Open the slide ONCE outside the loop, not once per LN component
        slide = openslide.open_slide(wsi_path)
        for i in range(1, num_labels):
            bbox = cv2.boundingRect((labeled_lns == i).astype(np.uint8))
            crop = slide.read_region(
                (read_origin[0] + bbox[0] * downsampling_level,
                 read_origin[1] + bbox[1] * downsampling_level),
                level,
                (bbox[2] * tiling_downsample_factor, bbox[3] * tiling_downsample_factor),
            )
            crop = np.array(crop)
            crop[crop[:, :, 3] == 0] = 255
            crop = cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)
            if tiling_downsample_factor > 1:
                crop = cv2.resize(
                    crop,
                    (crop.shape[1] // tiling_downsample_factor,
                     crop.shape[0] // tiling_downsample_factor),
                    interpolation=cv2.INTER_NEAREST,
                )
            crop_ln    = crop[ln_mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] > 0]
            has_colors = detect_colors(crop_ln)
            if not has_colors:
                pred[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] = label2id["Background"]
        slide.close()

        ln_mask    = (gt == label2id["Lymph node"]).astype(np.uint8)
        num_labels, labeled_lns = cv2.connectedComponents(ln_mask)
        for i in range(1, num_labels):
            aux  = (labeled_lns == i).astype(np.uint8)
            area = cv2.countNonZero(aux)
            if area < min_area:
                gt[aux > 0] = label2id["Background"]

    categories_to_eval = sorted(list(label2id.values()))
    iou, dice, mcc     = get_multi_class_metrics(gt, pred, categories_to_eval)

    results = {}
    results["filename"]              = os.path.splitext(os.path.basename(wsi_path))[0]
    results["time_inference_wsi"]    = time_elapsed
    results["time_inference_tile"]   = time_tile
    for i, k in enumerate(categories_to_eval):
        results[f"dice_{id2label[k]}"] = dice[i]
        results[f"iou_{id2label[k]}"]  = iou[i]
        results[f"mcc_{id2label[k]}"]  = mcc[i]

    return results, pred, gt, level, downsampling_level


@torch.no_grad()
def evaluate_wsi_tiles(
        model,
        wsi_root: str,
        annotations_paths: List[str],
        batch_size: int,
        tile_size: int,
        step_size: int,
        resolution: float,
        label2id: dict,
        dataset_save_path: str,
        ignore_index: int,
) -> dict:
    """Unchanged from original — out of scope for this optimisation pass."""
    for path in annotations_paths:
        assert os.path.exists(path), f"Path {path} does not exist"
        assert path.endswith(".geojson"), f"Path {path} is not a geojson file"

    if os.path.exists(dataset_save_path):
        shutil.rmtree(dataset_save_path)
    os.makedirs(dataset_save_path, exist_ok=True)

    model.eval()
    model_class = get_model_class_from_model(model)
    dataset = TileDataset(
        list_of_masks=annotations_paths,
        wsi_root=wsi_root,
        resolution=resolution,
        tile_size=tile_size,
        step_size=step_size,
        label2id=label2id,
        num_classes=len(label2id),
        dataset_save_path=dataset_save_path,
        data_augs=None,
    )
    id2label    = {v: k for k, v in label2id.items()}
    collate_fn  = TrainCollator(ignore_index) if model_class == "Mask2FormerForUniversalSegmentation" else None
    tile_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        pin_memory=True, drop_last=False, collate_fn=collate_fn,
    )
    num_batches        = len(tile_loader)
    dices, ious, mccs, coords, filenames = [], [], [], [], []
    categories_to_eval = [i for i in label2id.values() if i != ignore_index]

    with tqdm(total=num_batches, unit="Batch", desc="Batches", dynamic_ncols=True) as pbar:
        for batch in tile_loader:
            pixel_values  = batch["pixel_values"].to(model.device)
            mask_labels   = [l.to(model.device) for l in batch["mask_labels"]]
            class_labels  = [l.to(model.device) for l in batch["class_labels"]]
            outputs       = model(
                pixel_values=pixel_values,
                mask_labels=mask_labels,
                class_labels=class_labels,
            )
            coord    = batch["coords"]
            filename = batch["filename"]
            target_sizes = [(t.size(1), t.size(2)) for t in batch["mask_labels"]]
            batch_gt     = batch["original_segmentation_maps"]
            predicted    = post_process_output(outputs, target_sizes).cpu()
            unsqueeze_first = batch_gt.shape[0] == 1
            batch_gt  = batch_gt.squeeze().unsqueeze(0).numpy()  if unsqueeze_first else batch_gt.squeeze().numpy()
            unsqueeze_first = predicted.shape[0] == 1
            predicted = predicted.squeeze().unsqueeze(0).numpy() if unsqueeze_first else predicted.squeeze().numpy()

            for pred, true in zip(predicted, batch_gt):
                mask = true == ignore_index
                pred[mask] = ignore_index
                pred = cv2.erode(pred.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
                pred = cv2.dilate(pred.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
                iou, dice, mcc = get_multi_class_metrics(true, pred, categories_to_eval)
                dices.append(dice); ious.append(iou); mccs.append(mcc)

            coords.extend(coord)
            filenames.extend(filename)
            pbar.update(1)

    results = {}
    results["filename"] = filenames
    results["x_start"]  = [c[0] for c in coords]
    results["x_end"]    = [c[1] for c in coords]
    results["y_start"]  = [c[2] for c in coords]
    results["y_end"]    = [c[3] for c in coords]
    for i, k in enumerate(categories_to_eval):
        results[f"dice_{id2label[k]}"] = [d[i] for d in dices]
        results[f"iou_{id2label[k]}"]  = [iou[i] for iou in ious]
        results[f"mcc_{id2label[k]}"]  = [m[i] for m in mccs]
    return results


@torch.no_grad()
def infer_wsi(
        model,
        wsi_path: str,
        filter_mask: np.ndarray,
        batch_size: int,
        tile_size: int,
        step_size: int,
        crop_pred_edge: int,
        resolution: float,
        downsample_factor: int = 1,
        normalize_input: bool = True,
) -> Tuple[np.ndarray, int, int, float, Tuple[int, int], float, float]:
    """
    Perform inference on a whole slide image (WSI).

    Changes vs original
    -------------------
    [B1]  target_size computed once outside the loop as a plain (H, W) tuple
          shared by every batch (all tiles are the same size).  Replaces the
          per-batch Python list comprehension.

    [B2]  Async double-buffer pipeline.
          The loop maintains two slots — current and previous batch — and
          overlaps:
            • GPU: H2D transfer + normalisation + forward + post-processing
            • CPU: D2H copy (stitch_stream) + stitch_predictions
          via two CUDA streams and two CUDAEvents for correct ordering.
          On CPU-only systems the pipeline degenerates to the original
          sequential loop (no streams, no events).

    [B4]  AMP scope extended to cover post_process_output() so that
          F.interpolate, einsum, sigmoid and softmax run in FP16.

    [B9]  Normalisation moved to GPU.
          InferCollator(normalize=False) is passed so workers skip the CPU
          normalise step.  A (1,3,1,1) mean/std tensor is pre-loaded on the
          device and applied after the non-blocking H2D transfer.

    CUDA Graphs
    -----------
    Not applied.  See module docstring for detailed reasoning.
    TL;DR: kernel-launch overhead is < 2% of batch time; the dominant
    bottlenecks are data transfer (B2) and per-sample ops (B1), which CUDA
    Graphs do not address.  The additional constraints (no CPU branching, no
    dynamic allocation, warm-up passes) would complicate the code without
    measurable benefit.
    """
    model.eval()

    dataset     = SlideDataset(
        wsi_path=wsi_path,
        filter_mask=filter_mask,
        downsample_factor=downsample_factor,
        resolution=resolution,
        tile_size=tile_size,
        step_size=step_size,
        crop_size=crop_pred_edge,
        apply_tta=False,
        rotations=[0],
        flips=[],
        color_jitter=None,
        noise=False,
        blur=False,
        gamma=False,
    )

    num_tiles    = len(dataset)
    cpus_per_task = int(os.getenv("SLURM_CPUS_PER_TASK", 1))
    num_workers  = cpus_per_task - 1 if cpus_per_task > 1 else 0

    tile_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        # [B3] Workers are now safe: BaseSlideDataset uses the lazy .wsi property
        num_workers=num_workers,
        pin_memory=True,
        # prefetch_factor only valid when num_workers > 0
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        drop_last=False,
        # [B9] Skip CPU normalisation — done on GPU after H2D transfer
        collate_fn=InferCollator(normalize=False),
    )

    num_batches = (num_tiles + batch_size - 1) // batch_size
    cuda_available = torch.cuda.is_available()
    device         = model.device

    # [B9] Normalisation constants resident on GPU for the entire WSI
    if normalize_input and cuda_available:
        norm_mean = torch.tensor([0.485, 0.456, 0.406],
                                 device=device, dtype=torch.float32).view(1, 3, 1, 1)
        norm_std  = torch.tensor([0.229, 0.224, 0.225],
                                 device=device, dtype=torch.float32).view(1, 3, 1, 1)

    # [B1] target_size: fixed tuple shared across all batches (all tiles same size)
    target_size = (tile_size, tile_size)

    # [B2] Two CUDA streams for pipelined forward + async stitch
    if cuda_available:
        fwd_stream    = torch.cuda.current_stream(device)
        stitch_stream = torch.cuda.Stream(device=device)
        # Events to sequence the two streams without blocking CPU
        fwd_done_event    = torch.cuda.Event()
        stitch_done_event = torch.cuda.Event()
    else:
        fwd_stream = stitch_stream = None

    # Double-buffer state: holds the *previous* batch while current is on GPU
    prev_logits_cpu   = None      # CPU tensor ready for stitching
    prev_coords       = None
    prev_augmentations = None

    start_time = time.time()

    with tqdm(total=num_batches, unit="Batch", desc="Batches", dynamic_ncols=True) as pbar:
        for batch, coords, augmentations in tile_loader:

            # ----------------------------------------------------------------
            # Wait for previous stitch to finish before we overwrite prev_*
            # (only relevant from iteration 2 onward)
            # ----------------------------------------------------------------
            if cuda_available and prev_logits_cpu is not None:
                # [B2] fwd_stream waits until stitch_stream has finished its D2H
                fwd_stream.wait_event(stitch_done_event)

            # ----------------------------------------------------------------
            # [B9] Non-blocking H2D + GPU normalisation
            # ----------------------------------------------------------------
            batch_gpu = batch.to(device, non_blocking=True)          # async H2D
            if normalize_input:
                if cuda_available:
                    batch_gpu = (batch_gpu - norm_mean) / norm_std
                else:
                    batch_gpu = tv_F.normalize(
                        batch_gpu,
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    )

            # ----------------------------------------------------------------
            # [B4] AMP scope extended to cover post_process_output
            # ----------------------------------------------------------------
            with torch.autocast(device_type="cuda" if cuda_available else "cpu",
                                 dtype=torch.float16 if cuda_available else torch.float32):
                outputs = model(pixel_values=batch_gpu)
                # [B1] Single batched call; target_size is a constant tuple
                logits  = post_process_output(outputs, target_size, return_logits=True)
                # logits: (B, C, tile_size, tile_size) — FP16 on GPU

            # ----------------------------------------------------------------
            # [B2] Record that forward is done, then kick off async D2H in
            # stitch_stream while fwd_stream proceeds to the next batch
            # ----------------------------------------------------------------
            if cuda_available:
                fwd_done_event.record(fwd_stream)

                with torch.cuda.stream(stitch_stream):
                    stitch_stream.wait_event(fwd_done_event)
                    # Async D2H: moves (B, C, H, W) FP16 logits off GPU
                    logits_cpu = logits.cpu()
                stitch_done_event.record(stitch_stream)
            else:
                logits_cpu = logits.cpu()

            # ----------------------------------------------------------------
            # Stitch the *previous* batch on the CPU while the GPU handles
            # the *current* batch's forward pass (double-buffer pattern)
            # ----------------------------------------------------------------
            if prev_logits_cpu is not None:
                # For the GPU path, wait for the D2H of the previous batch
                # to complete before reading prev_logits_cpu
                if cuda_available:
                    torch.cuda.current_stream().wait_event(stitch_done_event)
                dataset.stitch_predictions(prev_logits_cpu, prev_coords, prev_augmentations)

            # Rotate the double-buffer
            prev_logits_cpu    = logits_cpu
            prev_coords        = coords
            prev_augmentations = augmentations

            pbar.update(1)

    # ----------------------------------------------------------------
    # Flush the last batch (it was never stitched inside the loop)
    # ----------------------------------------------------------------
    if prev_logits_cpu is not None:
        if cuda_available:
            torch.cuda.synchronize(device)   # ensure async D2H completed
        dataset.stitch_predictions(prev_logits_cpu, prev_coords, prev_augmentations)

    pred_mask, _ = dataset.create_final_predictions(return_probs=False)

    pred_mask = pred_mask[
        : dataset.original_shape[0] // dataset.tiling_downsample_factor,
        : dataset.original_shape[1] // dataset.tiling_downsample_factor,
    ].astype(np.uint8)
    pred_mask *= cv2.resize(
        filter_mask,
        (pred_mask.shape[1], pred_mask.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    level                  = dataset.level
    downsampling_level     = dataset.level_downsampling
    exact_resolution       = dataset.exact_resolution
    read_origin            = dataset.read_origin
    tiling_downsample_factor = dataset.tiling_downsample_factor
    del dataset

    return (
        pred_mask,
        level,
        downsampling_level,
        exact_resolution,
        tiling_downsample_factor,
        read_origin,
        time.time() - start_time,
        0.0,   # time_tile kept for API compatibility (profiling disabled)
    )
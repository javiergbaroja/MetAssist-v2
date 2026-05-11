"""
MetAssist 2.0
========================================
Two-stage Mask2Former pipeline:
  Stage 1 — Lymph node segmentation
  Stage 2 — Metastasis / deposit detection within LN boundaries

All inputs arrive as environment variables exported by the PathoDB API
(via sbatch --export). Outputs written to PATHODB_RESULT_DIR:
  result.json    — served to the browser when the job is done
  <wsi>_ln.geojson          — LN boundary overlay (QuPath-compatible)
  <wsi>_metastasis.geojson  — metastasis/deposit overlay
  error.txt      — stack trace on failure (for cluster debugging)
"""
import time
import glob
import json
import os
import sys
from natsort import natsorted
import yaml
import argparse

import cv2
import numpy as np
import openslide
import torch
from scipy.ndimage import binary_fill_holes
import tifffile

# ── Third-party package paths ──────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = "/storage/research/igmp_slide_workspace/GRP Zlobec/Amjad/qupath/metassist-v1/MetAssist_expansion/crc-ugi/code/package_refactored"
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PACKAGE_DIR)

from models.model_io import create_mask2former_from_checkpoint
from engine.inference import infer_wsi
from utils.wsi import prepare_read_from_slide, detect_colors, ACCEPTED_WSI_TYPES
from utils.geometry import save_geojson_annotation
from utils.evaluation import get_slide_level_result
from utils.postprocessing import post_process
from utils.visualization import COLORMAP


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def discover_wsis(input_dir: str) -> list[str]:
    if os.path.isfile(input_dir):
        with open(input_dir, 'r') as f:
            wsi_files = [line.strip() for line in f if line.strip()]
        wsi_files = natsorted([f for f in wsi_files if os.path.splitext(f)[1].lower() in ACCEPTED_WSI_TYPES])
    else:
        wsi_files = []
        for ext in ACCEPTED_WSI_TYPES:
            wsi_files.extend(glob.glob(os.path.join(input_dir, f"**/*.{ext}"), recursive=True))
        wsi_files = natsorted(wsi_files)
    if not wsi_files:
        raise FileNotFoundError(f"No WSIs found in {input_dir}")
    return wsi_files


def write_progress(pct: int, message: str) -> None:
    """
    Write progress.json atomically so the API never reads a partial file.
    pct must be in [0, 100].
    """
    pct = max(0, min(100, int(pct)))
    print(f"[{pct:3d}%] {message}", flush=True)


def close_metastasis(pred_mask: np.ndarray, metastasis_class: int) -> np.ndarray:
    """Close small gaps inside metastasis regions."""
    kernel = np.ones((5, 5), np.uint8)
    met_mask = (pred_mask == metastasis_class).astype(np.uint8)
    met_mask = cv2.morphologyEx(met_mask, cv2.MORPH_CLOSE, kernel)
    pred_mask[met_mask == 1] = metastasis_class
    return pred_mask


def merge_mucin_and_ln(
    ln_seg: np.ndarray,
    ln_class: int,
    mucin_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge mucin regions into LN if they are spatially adjacent, then
    fill holes in the resulting LN mask.

    Returns (updated_seg, filled_ln_binary_mask).

    Bug fix vs original: uses an explicit `mucin_class < 0` sentinel check
    instead of relying on `-1 not in array` which is fragile with NumPy
    integer arrays (would silently pass if the array ever contained -1).
    """
    ln_mask = (ln_seg == ln_class).astype(np.uint8)

    if mucin_class < 0 or mucin_class not in np.unique(ln_seg):
        return ln_seg, binary_fill_holes(ln_mask).astype(np.uint8)

    ln_dilated   = cv2.dilate(ln_mask, np.ones((5, 5), np.uint8), iterations=1)
    mucin_mask   = (ln_seg == mucin_class).astype(np.uint8)
    num, labeled = cv2.connectedComponents(mucin_mask, connectivity=8)

    for label in range(1, num):
        component = (labeled == label)
        if np.any(component & ln_dilated):
            ln_mask |= component
        else:
            ln_seg[component] = 0

    return ln_seg, binary_fill_holes(ln_mask).astype(np.uint8)


def filter_ln_noise_regions(
    wsi_path: str,
    ln_pred_mask: np.ndarray,
    label2id: dict[str, int],
    read_origin: tuple[int, int],
    level: int,
    level_downsampling: int,
    tiling_downsample_factor: float,
    original_dim: tuple[int, int],
    color_threshold: float = 0.025,
) -> np.ndarray:
    """Remove LN components that lack sufficient tissue-like color."""
    ln_class_mask = (ln_pred_mask == label2id.get("lymph_node", 1)).astype(np.uint8)
    num_labels, label_map = cv2.connectedComponents(ln_class_mask)
    slide_handle = openslide.open_slide(wsi_path)

    # Read the full LN-resolution slide once; slice per component to avoid repeated I/O.
    full_crop = np.array(
        slide_handle.read_region(
            read_origin,
            level,
            (original_dim[1], original_dim[0]),
        )
    )
    full_crop[full_crop[:, :, 3] == 0] = 255
    full_crop = cv2.cvtColor(full_crop, cv2.COLOR_RGBA2RGB)
    if tiling_downsample_factor > 1:
        full_crop = cv2.resize(
            full_crop,
            (
                full_crop.shape[1] // tiling_downsample_factor,
                full_crop.shape[0] // tiling_downsample_factor,
            ),
        )

    for i in range(1, num_labels):
        bbox = cv2.boundingRect((label_map == i).astype(np.uint8))
        y1 = min(bbox[1] + bbox[3], full_crop.shape[0])
        x1 = min(bbox[0] + bbox[2], full_crop.shape[1])
        if y1 <= bbox[1] or x1 <= bbox[0]:
            continue

        crop = full_crop[bbox[1]:y1, bbox[0]:x1]
        roi_mask = ln_class_mask[bbox[1]:y1, bbox[0]:x1]
        if not detect_colors(crop[roi_mask > 0], color_threshold):
            ln_pred_mask[
                bbox[1]:y1,
                bbox[0]:x1,
            ] = label2id.get("background", 0)

    slide_handle.close()
    return ln_pred_mask


def write_ometiff_overlays(
    ome_tiff: str,
    cfg: dict,
    met_boundary_mask: np.ndarray,
    met_pred_mask: np.ndarray,
) -> None:
    
    # 1. Resize LN mask
    tiff_array = cv2.resize(met_boundary_mask, (met_pred_mask.shape[1], met_pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # 2. Overwrite LN pixels with Tumor pixels where Tumor exists
    tumor_class_id = cfg['met_task']['label2id'].get('tumor', 2)
    tiff_array = np.where(met_pred_mask == tumor_class_id, tumor_class_id, tiff_array)

    # 3. Use dynamic mapping rather than hardcoded 3
    lut_rgba = np.zeros((256, 4), dtype=np.uint8)
    lut_rgba[0] = [0, 0, 0, 0] 
    lut_rgba[cfg['ln_task']['label2id']['lymph_node']] = list(COLORMAP.get('lymph_node', (0,255,0))) + [150]
    lut_rgba[tumor_class_id] = list(COLORMAP.get('tumor', (255,0,0))) + [150]

    rgba_mask = lut_rgba[tiff_array]
    levels = [rgba_mask]
    current = rgba_mask
    while min(current.shape[:2]) > 512:
        current = cv2.resize(
            current,
            (current.shape[1] // 2, current.shape[0] // 2),
            interpolation=cv2.INTER_NEAREST,
        )
        levels.append(current)

    with tifffile.TiffWriter(ome_tiff, bigtiff=True) as tif:
        for i, level_img in enumerate(levels):
            tif.write(
                level_img,
                subfiletype=1 if i > 0 else 0,
                photometric='rgb',
                tile=(256, 256),
                compression='deflate',
            )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference script for MetAssist 2.0")
    parser.add_argument("--config", type=str, default="./configs/default.yaml", help="Path to YAML config file")
    parser.add_argument("--input_dir", type=str, nargs="?", help="Path to input WSI or text file listing WSIs (overrides config)")
    parser.add_argument("--output_dir", type=str, nargs="?", help="Path to output directory (overrides config)")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.input_dir:
        cfg['input_dir'] = args.input_dir
    if args.output_dir:
        cfg['output_dir'] = args.output_dir
    
    return cfg

def resolve_existing_results(wsi_paths: list[str], output_dir: str, overwrite: bool = False) -> list[str]:
    if overwrite:
        return wsi_paths
    existing_results = glob.glob(os.path.join(output_dir, "*_result.json"))
    existing_basenames = {os.path.splitext(os.path.basename(p))[0].replace("_result", "") for p in existing_results}
    return [p for p in wsi_paths if os.path.splitext(os.path.basename(p))[0] not in existing_basenames]

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = get_args()   

    downsample_factor = 1

    # ── Pre-flight checks ──────────────────────────────────────────────────────
    wsi_paths = discover_wsis(cfg["input_dir"])
    wsi_paths = resolve_existing_results(wsi_paths, cfg["output_dir"], overwrite=cfg.get("overwrite_existing_results", False))
    if not wsi_paths:
        print(f"No WSIs to process. Exiting.")
        sys.exit(0)
    if not os.path.exists(cfg['ln_task']['checkpoint_path']):
        raise FileNotFoundError(f"LN checkpoint not found: {cfg['ln_task']['checkpoint_path']}")
    if not os.path.exists(cfg['met_task']['checkpoint_path']):
        raise FileNotFoundError(f"Metastasis checkpoint not found: {cfg['met_task']['checkpoint_path']}")
    os.makedirs(cfg['output_dir'], exist_ok=True)

    print(f"=== MetAssist 2.0 Lymph Node Metastasis Detection ===")
    print(f"Timestamp   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of WSIs : {len(wsi_paths)}")
    print(f"Result dir : {cfg['output_dir']}")
    print(f"LN Resolution : {cfg['ln_task']['resolution']} µm/px")
    print(f"Metastasis Resolution : {cfg['met_task']['resolution']} µm/px", flush=True)

    # ── Load models ────────────────────────────────────────────────────────────
    write_progress(0, "Loading models into memory…")
    cpu_device = torch.device("cpu")
    gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ln_model = create_mask2former_from_checkpoint(checkpoint_path=cfg['ln_task']['checkpoint_path'], 
                                                  label2id=cfg['ln_task']['label2id'], 
                                                  encoder_name=cfg['ln_task']['encoder_model'], 
                                                  decoder_model=cfg['ln_task']['decoder_model'], 
                                                  out_indices=cfg['ln_task']['feature_layers'])
    
    met_model = create_mask2former_from_checkpoint(checkpoint_path=cfg['met_task']['checkpoint_path'], 
                                                   label2id=cfg['met_task']['label2id'], 
                                                   encoder_name=cfg['met_task']['encoder_model'], 
                                                   decoder_model=cfg['met_task']['decoder_model'], 
                                                   out_indices=cfg['met_task']['feature_layers'])
    
    for wsi_path in wsi_paths:
        wsi_name, wsi_type = os.path.splitext(os.path.basename(wsi_path))

        # ── Stage 1 — Lymph node segmentation ─────────────────────────────────────
        write_progress(10, "Running lymph node segmentation…")
        ln_model.to(gpu_device)
        met_model.to(cpu_device)
        tissue_mask = np.ones((5, 5), dtype=np.uint8)
        (level, level_downsampling, exact_resolution, tiling_downsample_factor, original_dim, read_origin) = prepare_read_from_slide(wsi_path, 
                                                                                                                                    resolution=cfg['ln_task']['resolution'], 
                                                                                                                                    file_type=wsi_type.lower())

        ln_pred_mask, __, __, __, __, __, ln_time, __ = infer_wsi(model=ln_model, 
                                                            wsi_path=wsi_path, 
                                                            filter_mask=tissue_mask,
                                                            batch_size=cfg['ln_task']['batch_size'], 
                                                            tile_size=cfg['ln_task']['tile_size'], 
                                                            step_size=cfg['ln_task']['step_size'], 
                                                            crop_pred_edge=cfg['ln_task']['crop_pred_edge'],
                                                            resolution=cfg['ln_task']['resolution'], 
                                                            downsample_factor=downsample_factor,
        )

        write_progress(40, "Applying LN post-processing…")

        if cfg['apply_post_processing']:
            min_ln_area = ((600 / 2) / (exact_resolution * tiling_downsample_factor)) ** 2 * np.pi
            ln_pred_mask = post_process(
                segmentation_mask    = ln_pred_mask,
                lymph_node_class     = cfg['ln_task']['label2id'].get("lymph_node", 1),
                classes_to_merge     = [cfg['ln_task']['label2id'].get("tumor_deposit_or_primary_tumor", 2), 
                                        cfg['ln_task']['label2id'].get("mucin", 6)],
                merge_thresholds     = [0.95, 0.05],
                erase_thresholds     = [0.01, 0.01],
                apply_opening        = [True, False],
                min_ln_area          = int(min_ln_area),
                complexity_threshold = 2.9,
            )

            # Filter detections in tissue-free / noise regions
            write_progress(47, "Filtering LN noise regions…")
            ln_pred_mask = filter_ln_noise_regions(
                wsi_path=wsi_path,
                ln_pred_mask=ln_pred_mask,
                label2id=cfg['ln_task']['label2id'],
                read_origin=read_origin,
                level=level,
                level_downsampling=level_downsampling,
                tiling_downsample_factor=tiling_downsample_factor,
                original_dim=original_dim,
                color_threshold=0.025,
            )

        # Build LN boundary mask; release raw Stage 1 output
        ln_seg_all, met_boundary_mask = merge_mucin_and_ln(
            ln_pred_mask.copy(),
            cfg['ln_task']['label2id'].get("lymph_node", 1),
            cfg['ln_task']['label2id'].get("mucin", 6),
        )
        del ln_pred_mask
        ds_ln = level_downsampling * tiling_downsample_factor
        torch.cuda.empty_cache()

        # ── Stage 2 — Metastasis segmentation ─────────────────────────────────────
        write_progress(55, "Running metastasis segmentation…")
        ln_model.to(cpu_device)
        met_model.to(gpu_device)
        (level, level_downsampling, exact_resolution, tiling_downsample_factor, original_dim, read_origin) = prepare_read_from_slide(wsi_path, 
                                                                                                                                    resolution=cfg['met_task']['resolution'], 
                                                                                                                                    file_type=os.path.splitext(wsi_path)[1].lower())

        met_pred_mask, _, _, _, _, _, met_time, __ = infer_wsi(model=met_model, 
                                                            wsi_path=wsi_path, 
                                                            filter_mask=met_boundary_mask,
                                                            batch_size=cfg['met_task']['batch_size'], 
                                                            tile_size=cfg['met_task']['tile_size'], 
                                                            step_size=cfg['met_task']['step_size'],
                                                            crop_pred_edge=cfg['met_task']['crop_pred_edge'],
                                                            resolution=cfg['met_task']['resolution'], 
                                                            downsample_factor=tiling_downsample_factor,
        )

        if cfg['apply_post_processing']:
            met_pred_mask = close_metastasis(met_pred_mask, cfg['met_task']['label2id']["tumor"])

        # ── Outputs ───────────────────────────────────────────────────────────────
        if cfg['save_qupath_geojson']:
            write_progress(82, "Saving GeoJSON overlays…")

            geojson_met = os.path.join(cfg['output_dir'], f"{wsi_name}_metastasis.geojson")
            geojson_ln  = os.path.join(cfg['output_dir'], f"{wsi_name}_ln.geojson")

            ds_met = level_downsampling * tiling_downsample_factor

            save_geojson_annotation(
                out_path     = geojson_met,
                mask         = met_pred_mask,
                level        = level,
                level_downsampling = ds_met,
                category_dict = {
                    k: v for k, v in cfg['met_task']['label2id'].items()
                    if k.lower() not in cfg['ignore_class_for_overlay']
                },
            )
            save_geojson_annotation(
                out_path     = geojson_ln,
                mask         = met_boundary_mask,
                level        = level,
                level_downsampling = ds_ln,
                category_dict = {"lymph_node": 1},
            )

        if cfg['save_ometiff_overlays']:
            write_progress(85, "Saving OME-TIFF overlays…")
            ome_tiff = os.path.join(cfg['output_dir'], f"{wsi_name}_overlays.ome.tiff")
            write_ometiff_overlays(
                ome_tiff=ome_tiff,
                cfg=cfg,
                met_boundary_mask=met_boundary_mask,
                met_pred_mask=met_pred_mask,
            )           

        # ── Slide-level clinical result ────────────────────────────────────────────
        if cfg['save_slide_level_result']:
            write_progress(92, "Computing slide-level result…")

            status, label, measurement = get_slide_level_result(
                mask             = met_pred_mask,
                ln_seg_mask      = ln_seg_all,
                metastasis_class = cfg['met_task']['label2id'].get("tumor", 1),
                ln_class         = cfg['ln_task']['label2id']["lymph_node"],
                deposit_class    = cfg['ln_task']['label2id'].get("tumor_deposit_or_primary_tumor", 2),
                fat_class        = cfg['ln_task']['label2id'].get("fat", 4),
                mucin_class      = cfg['ln_task']['label2id'].get("mucin", 6),
                resolution       = exact_resolution * tiling_downsample_factor,
            )

        # ── result.json — read by GET /analysis/jobs/{id}/result ─────────────────
        write_progress(95, "Writing result summary…")

        result = {
            "model_id":    "metassist_v2",
            "wsi_path":   wsi_path,
            "params": {
                "ln_resolution":            cfg['ln_task']['resolution'],
                "ln_batch_size":            cfg['ln_task']['batch_size'],
                "ln_tile_size":             cfg['ln_task']['tile_size'],
                "ln_step_size":             cfg['ln_task']['step_size'],
                "met_resolution":           cfg['met_task']['resolution'],
                "met_batch_size":           cfg['met_task']['batch_size'],
                "met_tile_size":            cfg['met_task']['tile_size'],
                "met_step_size":            cfg['met_task']['step_size'],
                "complexity_threshold":     2.9,
                "apply_post_processing":    cfg['apply_post_processing'],
            },
            "timing": {
                "ln_inference_s":  round(ln_time,  2),
                "met_inference_s": round(met_time, 2),
                "total_s":         round(ln_time + met_time, 2),
            },
            "outcome": {
                "status":         status,
                "label":          label,
                "measurement_um": round(float(measurement), 2),
            },
            "files": {
                "metastasis_geojson": geojson_met if cfg['save_qupath_geojson'] else None,
                "ln_geojson":         geojson_ln if cfg['save_qupath_geojson'] else None,
                "overlay_tiff":       ome_tiff if cfg['save_ometiff_overlays'] else None, 
            },
        }

        result_path = os.path.join(cfg['output_dir'], f"{wsi_name}_result.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

    write_progress(100, "Done")

    print(f"\n=== Complete ===")
    print(f"LN inference   : {ln_time:.2f}s")
    print(f"Met inference  : {met_time:.2f}s")
    print(f"Total          : {ln_time + met_time:.2f}s")
    print(f"Outcome        : {measurement:.2f} µm — {status} — {label}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()

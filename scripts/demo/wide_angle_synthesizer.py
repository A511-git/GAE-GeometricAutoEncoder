#!/usr/bin/env python3
"""Multi-Image Wide-Angle Synthesizer with Sequential VRAM Offloading.

Takes multiple unposed images (e.g. 5 faces of a room/hexagon), auto-estimates
their 3D camera angles using Depth Anything 3 (DA3), places a virtual wide-angle
camera in the 3D space, and uses GAE's flow model to synthesize a single, clean,
seamless photorealistic wide-angle photograph of the entire space.

Sequential model lifecycle:
  DA3 (pose + encode) -> Qwen3 -> Flow DiT -> GAE Codec
Peak VRAM is bounded to ~10-12 GB, making it fully compatible with Kaggle,
Google Colab, and consumer GPUs.

Usage Examples:
    # Option 1: Pass a folder containing images (3, 5, or any number of photos):
    python scripts/demo/wide_angle_synthesizer.py \\
        --images path/to/room_photos/ \\
        --prompt "a photorealistic wide-angle view of a serene room, sharp focus, natural lighting" \\
        --pull-back 1.2 \\
        --fov-scale 0.70 \\
        --output results/wide_angle_room

    # Option 2: Pass a list of individual image files:
    python scripts/demo/wide_angle_synthesizer.py \\
        --images face1.jpg face2.jpg face3.jpg face4.jpg face5.jpg \\
        --prompt "a photorealistic wide-angle view of a serene room, sharp focus, natural lighting" \\
        --pull-back 1.2 \\
        --fov-scale 0.70 \\
        --output results/wide_angle_room
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for sub in (ROOT / "src", ROOT / "scripts" / "eval", ROOT / "scripts" / "demo", ROOT / "scripts" / "data"):
    if str(sub) not in sys.path:
        sys.path.insert(0, str(sub))

try:
    import psutil
except ImportError:
    psutil = None

_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _log_memory(stage: str) -> None:
    """Print real-time system RAM and GPU VRAM usage."""
    ram_mb = psutil.Process().memory_info().rss / (1024 * 1024) if psutil else 0
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024 * 1024)
        res = torch.cuda.memory_reserved() / (1024 * 1024)
        max_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"[{stage}] RAM: {ram_mb:.0f}MB | GPU VRAM: {alloc:.0f}MB alloc, {res:.0f}MB res (peak {max_alloc:.0f}MB)", flush=True)
    else:
        print(f"[{stage}] System RAM: {ram_mb:.0f}MB", flush=True)


def _free(*modules, stage: str = "") -> None:
    """Drop references and reclaim VRAM immediately."""
    for m in modules:
        if m is not None:
            if hasattr(m, "to"):
                try:
                    m.to("cpu")
                except Exception:
                    pass
            del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if stage:
        _log_memory(f"After Offload: {stage}")


def _cast(module: torch.nn.Module, precision: str) -> torch.nn.Module:
    return module.to(dtype=_DTYPES[precision]) if module is not None else module


# ─────────────────────────── Camera & Pose Math ───────────────────────────

def estimate_poses_with_da3(
    images_pil: list[Image.Image],
    device: torch.device,
    model_id: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    process_res: int = 504,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Auto-estimate camera poses (c2w) and intrinsics (K) from raw images using DA3."""
    print(f"\n[Auto-Pose] Loading DA3 pose estimator ({model_id})...")
    _log_memory("Before Loading DA3 Pose Estimator")
    from depth_anything_3.api import DepthAnything3
    da3 = DepthAnything3.from_pretrained(model_id).to(device).eval()
    _log_memory("DA3 Pose Estimator Loaded")

    print(f"[Auto-Pose] Estimating 3D camera angles across {len(images_pil)} images...")
    with torch.inference_mode():
        pred = da3.inference(
            images_pil,
            extrinsics=None,
            intrinsics=None,
            align_to_input_ext_scale=True,
            process_res=process_res,
        )

    w2c = np.asarray(pred.extrinsics, dtype=np.float64)
    K_raw = np.asarray(pred.intrinsics, dtype=np.float64)

    # Invert w2c to c2w
    c2w_list: list[np.ndarray] = []
    for m in w2c:
        if m.shape == (3, 4):
            m4 = np.eye(4, dtype=np.float64)
            m4[:3, :] = m
            m = m4
        c2w = np.linalg.inv(m)
        c2w_list.append(c2w)

    K_list = [K_raw[i].copy() for i in range(len(K_raw))]
    _free(da3, stage="DA3 Pose Estimator")
    print(f"[Auto-Pose] Solved {len(c2w_list)} camera poses successfully!")
    return c2w_list, K_list


def synthesize_wide_angle_camera(
    ref_c2w: list[np.ndarray],
    ref_K: list[np.ndarray],
    pull_back: float = 1.2,
    fov_scale: float = 0.70,
    yaw_offset_deg: float = 0.0,
    pitch_offset_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize a virtual wide-angle camera looking at the scene centroid.

    - Centers the camera at the centroid of reference camera positions.
    - Pulls back by `pull_back` meters along the average viewing direction.
    - Widens the field of view by scaling focal length by `fov_scale` (< 1.0 = wider lens).
    """
    positions = np.array([c[:3, 3] for c in ref_c2w], dtype=np.float64)
    centroid = positions.mean(axis=0)

    # Compute average viewing forward direction (+Z in camera frame)
    forwards = np.array([c[:3, 2] for c in ref_c2w], dtype=np.float64)
    avg_fwd = forwards.mean(axis=0)
    norm = np.linalg.norm(avg_fwd)
    if norm > 1e-6:
        avg_fwd = avg_fwd / norm
    else:
        avg_fwd = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # Position virtual camera pulled back along -avg_fwd
    cam_pos = centroid - pull_back * avg_fwd

    # Build rotation looking towards the scene
    z_axis = avg_fwd
    up_hint = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    x_axis = np.cross(up_hint, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    R = np.column_stack([x_axis, y_axis, z_axis])

    # Apply manual yaw / pitch offsets if given
    if abs(yaw_offset_deg) > 1e-4 or abs(pitch_offset_deg) > 1e-4:
        rad_y = np.deg2rad(yaw_offset_deg)
        rad_p = np.deg2rad(pitch_offset_deg)
        Ry = np.array([
            [np.cos(rad_y), 0, np.sin(rad_y)],
            [0, 1, 0],
            [-np.sin(rad_y), 0, np.cos(rad_y)],
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rad_p), -np.sin(rad_p)],
            [0, np.sin(rad_p), np.cos(rad_p)],
        ])
        R = R @ Ry @ Rx

    c2w_target = np.eye(4, dtype=np.float64)
    c2w_target[:3, :3] = R
    c2w_target[:3, 3] = cam_pos

    # Widen focal length in intrinsics
    K_target = ref_K[0].copy()
    K_target[0, 0] *= float(fov_scale)
    K_target[1, 1] *= float(fov_scale)

    return c2w_target, K_target


def render_trajectory_frustums(
    pose_list: list[np.ndarray],
    cond_num: int,
    output_path: Path | str,
    title: str = "3D Multi-Camera Layout & Wide-Angle Target",
) -> None:
    """Draw 3D camera frustums (Blue=reference images, Red=new wide-angle camera)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    V = len(pose_list)
    pos = np.array([p[:3, 3] for p in pose_list])
    pos_plt = np.column_stack([pos[:, 0], pos[:, 2], pos[:, 1]])
    extent = max(float(pos_plt.ptp(axis=0).max()), 1e-6)
    fd = extent * 0.16
    fs = extent * 0.10
    fs_h = fs * 0.65

    src_c, tgt_c = "#4285f4", "#ea4335"

    def _p(v):
        return np.array([v[0], v[2], v[1]])

    fig = plt.figure(figsize=(10, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    for i in range(V):
        c2w = pose_list[i].astype(np.float64)
        o = c2w[:3, 3]
        r = c2w[:3, 0]
        u = c2w[:3, 1]
        fwd = c2w[:3, 2]
        is_target = (i >= cond_num)
        color = tgt_c if is_target else src_c

        # Increase frustum size slightly for the target wide camera
        scale = 1.3 if is_target else 1.0
        corners = [
            o + fwd * fd * scale + (-r * fs * scale + u * fs_h * scale),
            o + fwd * fd * scale + ( r * fs * scale + u * fs_h * scale),
            o + fwd * fd * scale + ( r * fs * scale - u * fs_h * scale),
            o + fwd * fd * scale + (-r * fs * scale - u * fs_h * scale),
        ]
        cp = [_p(c) for c in corners]
        op = _p(o)

        for c in cp:
            ax.plot3D([op[0], c[0]], [op[1], c[1]], [op[2], c[2]], color=color, lw=1.4 if is_target else 1.0, alpha=0.85)
        base = cp + [cp[0]]
        ax.plot3D([c[0] for c in base], [c[1] for c in base], [c[2] for c in base], color=color, lw=1.4 if is_target else 1.0, alpha=0.85)

        poly = Poly3DCollection([cp], alpha=0.35 if is_target else 0.20, facecolor=color, edgecolor="none")
        ax.add_collection3d(poly)
        label = " TARGET WIDE" if is_target else f" Ref {i}"
        ax.text(op[0], op[1], op[2], label, fontsize=9, fontweight="bold",
                color="red" if is_target else "#222222")

    ax.set_xlabel("World X", labelpad=8)
    ax.set_ylabel("World Z (Depth)", labelpad=8)
    ax.set_zlabel("World Y (Up)", labelpad=8)
    ax.set_title(title, fontsize=13, pad=12)
    ax.view_init(elev=24, azim=-55)
    ax.grid(True, alpha=0.25)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[Frustums] 3D layout plot saved: {output_path}")


def snap_to_multiple_of_14(val: int) -> int:
    """Auto-round dimension to the nearest multiple of 14 (minimum 14)."""
    return max(14, int(round(val / 14.0)) * 14)


def resolve_resolution(h: int, w: int) -> tuple[int, int]:
    """Snap (H, W) to the nearest valid patch-grid multiples of 14."""
    snapped_h = snap_to_multiple_of_14(h)
    snapped_w = snap_to_multiple_of_14(w)
    if (snapped_h, snapped_w) != (h, w):
        print(f"[Resolution] Auto-snapped ({h}x{w}) -> ({snapped_h}x{snapped_w}) to match 14px patch grid.")
    return snapped_h, snapped_w


# ─────────────────────────── CLI Arguments ───────────────────────────

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def _natural_sort_key(p: Path) -> list:
    """Sort filenames containing numbers naturally (e.g. img_2 before img_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", p.name)]


def _collect_images(paths: list[Path] | None, folder: Path | None = None) -> list[Path]:
    """Gather image paths from either a list of file paths, directory paths, or a folder."""
    collected: list[Path] = []

    def _scan_dir(d: Path) -> list[Path]:
        files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS]
        return sorted(files, key=_natural_sort_key)

    if paths:
        for p in paths:
            if p.is_dir():
                found = _scan_dir(p)
                print(f"[WideSynthesizer] Found {len(found)} image(s) in folder: {p}")
                collected.extend(found)
            elif p.is_file():
                if p.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    collected.append(p)
                else:
                    print(f"[WideSynthesizer] Skipping non-image file: {p}", file=sys.stderr)
            else:
                # Attempt glob if wildcard
                if any(ch in str(p) for ch in ("*", "?", "[")):
                    globbed = sorted([x for x in Path().glob(str(p)) if x.is_file() and x.suffix.lower() in VALID_IMAGE_EXTENSIONS], key=_natural_sort_key)
                    collected.extend(globbed)
                else:
                    print(f"[WideSynthesizer] Warning: Path not found: {p}", file=sys.stderr)

    if folder and folder.is_dir():
        found = _scan_dir(folder)
        print(f"[WideSynthesizer] Found {len(found)} image(s) in folder: {folder}")
        collected.extend(found)

    # Deduplicate while preserving order
    seen: set[Path] = set()
    deduped: list[Path] = []
    for p in collected:
        try:
            resolved = p.resolve()
        except Exception:
            resolved = p
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(p)

    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", type=Path, default=None,
                        help="List of input image paths or folder(s) containing images (e.g. --images my_folder/ or --images f1.jpg f2.jpg).")
    parser.add_argument("--image-dir", "--images-dir", "--folder", type=Path, default=None, dest="image_dir",
                        help="Directory containing the input images (e.g. 3, 5, or more images).")
    parser.add_argument("--poses", type=Path, default=None,
                        help="Optional pre-existing pose .npz file (if omitted, DA3 auto-estimates).")
    parser.add_argument("--pose-model", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                        help="Depth Anything 3 model id or path for pose estimation (default: depth-anything/DA3NESTED-GIANT-LARGE-1.1).")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt describing the scene.")
    parser.add_argument("--prompt-file", type=Path, default=None,
                        help="Text file containing the prompt.")
    parser.add_argument("--pull-back", type=float, default=1.2,
                        help="Distance in meters to pull back the virtual wide camera (default: 1.2m).")
    parser.add_argument("--fov-scale", type=float, default=0.70,
                        help="Focal length scale (< 1.0 widens the field of view, default: 0.70).")
    parser.add_argument("--yaw-offset", type=float, default=0.0,
                        help="Manual yaw angle offset (deg) for the wide-angle camera.")
    parser.add_argument("--pitch-offset", type=float, default=0.0,
                        help="Manual pitch angle offset (deg) for the wide-angle camera.")
    parser.add_argument("--resolution", type=int, nargs=2, default=(378, 672),
                        metavar=("HEIGHT", "WIDTH"), help="Render resolution (default: 378 672).")
    parser.add_argument("--sample-steps", type=int, default=50, help="DiT Flow sampling steps.")
    parser.add_argument("--cfg-scale", type=float, default=2.0, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save-pointcloud", action="store_true", default=True, help="Export .ply 3D point cloud.")
    parser.add_argument("--hf-repo", default=None, help="Hugging Face repo id.")
    parser.add_argument("--checkpoint-dir", "--ckpt-dir", type=Path, default=None, dest="checkpoint_dir",
                        help="Directory where model checkpoints are saved/cached.")
    parser.add_argument("--flow-ckpt", type=Path, default=None)
    parser.add_argument("--codec-ckpt", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/flow_gae64.yaml")
    parser.add_argument("--codec-config", type=Path, default=ROOT / "configs/gae_64.yaml")
    parser.add_argument("--output", type=Path, required=True, help="Output results directory.")
    parser.add_argument("--backbone-precision", choices=list(_DTYPES), default="bf16")
    parser.add_argument("--text-precision", choices=list(_DTYPES), default="bf16")
    parser.add_argument("--codec-precision", choices=list(_DTYPES), default="bf16")
    parser.add_argument("--flow-precision", choices=list(_DTYPES), default="bf16")

    args = parser.parse_args()

    # Collect images
    image_paths = _collect_images(args.images, args.image_dir)

    if len(image_paths) < 2:
        source_hint = args.images if args.images else args.image_dir
        raise SystemExit(
            f"Error: Wide-angle synthesis requires >= 2 images to estimate relative 3D space "
            f"(found {len(image_paths)} from {source_hint})."
        )
    args.collected_images = image_paths

    if args.prompt_file and args.prompt_file.is_file():
        args.prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    if not args.prompt:
        args.prompt = "a photorealistic wide-angle view of a room, highly detailed, sharp focus, natural lighting"

    # Auto-resolve resolution to nearest valid multiples of 14
    args.resolution = resolve_resolution(args.resolution[0], args.resolution[1])

    from gae.hub import resolve_checkpoint_dir, resolve_model_file
    args.checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_dir)
    args.ckpt_dir = args.checkpoint_dir
    if args.codec_ckpt is not None:
        args.codec_ckpt = resolve_model_file(args.codec_ckpt, args.checkpoint_dir)
    if args.flow_ckpt is not None:
        args.flow_ckpt = resolve_model_file(args.flow_ckpt, args.checkpoint_dir)

    return args


def _download_checkpoints(args: argparse.Namespace) -> None:
    if args.hf_repo or args.flow_ckpt is None or args.codec_ckpt is None:
        from gae.hub import DEFAULT_REPO, download_weights, extract_da3_stats
        repo = args.hf_repo or os.environ.get("GAE_HF_REPO") or DEFAULT_REPO
        print(f"[WideSynthesizer] Verifying weights from '{repo}'...")
        paths = download_weights(repo, size="64", out_dir=args.checkpoint_dir)
        tar = paths.get("da3_stats_giant_5ds.tar")
        if tar is not None:
            extract_da3_stats(tar, ROOT / "model_stats" / "da3_giant_5ds")
        if args.codec_ckpt is None:
            args.codec_ckpt = paths["gae_64.pt"]
        if args.flow_ckpt is None:
            args.flow_ckpt = paths["flow_gae64.pt"]


# ─────────────────────────── Main Pipeline ───────────────────────────

def run_pipeline(args: argparse.Namespace) -> int:
    _download_checkpoints(args)
    args.output.mkdir(parents=True, exist_ok=True)
    height, width = args.resolution
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log_memory("Pipeline Start")

    N = len(args.collected_images)
    print(f"\n=======================================================")
    print(f" [GAE Wide-Angle Synthesizer] Processing {N} Images")
    print(f" Output Resolution: {width}x{height} | Pullback: {args.pull_back}m | FOV Scale: {args.fov_scale}")
    print(f"=======================================================")
    print(f"Input image sequence ({N} views):")
    for i, p in enumerate(args.collected_images):
        print(f"  [{i+1}/{N}] {p.name}")

    # Load and resize images
    pil_images: list[Image.Image] = []
    cv_images: list[np.ndarray] = []
    for p in args.collected_images:
        raw = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if raw is None:
            raise SystemExit(f"Cannot read image: {p}")
        resized = cv2.resize(raw, (width, height), interpolation=cv2.INTER_AREA)
        cv_images.append(resized)
        pil_images.append(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))

    # ── [1/5] Camera Pose Estimation (Auto-detected or Loaded) ──
    _log_memory("Before Stage 1 (Camera Poses)")
    if args.poses and args.poses.is_file():
        print(f"\n[1/5] Loading pre-existing poses from: {args.poses}")
        data = np.load(args.poses)
        c2w_key = "input_c2w" if "input_c2w" in data else "c2w"
        K_key = "input_K" if "input_K" in data else "K"
        ref_c2w = [data[c2w_key][i] for i in range(min(N, len(data[c2w_key])))]
        ref_K = [data[K_key][i] for i in range(min(N, len(data[K_key])))]
    else:
        print(f"\n[1/5] Auto-detecting 3D camera angles using DA3 ({args.pose_model})...")
        ref_c2w, ref_K = estimate_poses_with_da3(
            pil_images, device, model_id=args.pose_model, process_res=min(height, width)
        )
    _log_memory("After Stage 1 (Camera Poses)")

    # Synthesize target wide-angle camera
    target_c2w, target_K = synthesize_wide_angle_camera(
        ref_c2w, ref_K,
        pull_back=args.pull_back,
        fov_scale=args.fov_scale,
        yaw_offset_deg=args.yaw_offset,
        pitch_offset_deg=args.pitch_offset,
    )

    all_c2w = [*ref_c2w, target_c2w]
    all_K = [*ref_K, target_K]
    total_views = len(all_c2w)
    cond_num = N

    # Save 3D Camera Diagram & Poses
    render_trajectory_frustums(
        all_c2w, cond_num, args.output / "000_trajectory.png",
        title=f"Multi-Camera Layout: {N} References (Blue) + Wide Target (Red)"
    )
    np.savez_compressed(
        args.output / "000_poses.npz",
        input_c2w=np.stack([np.asarray(p, dtype=np.float64) for p in all_c2w]),
        input_K=np.stack([np.asarray(k, dtype=np.float64) for k in all_K]),
        cond_num=np.int64(cond_num),
        image_size=np.array([height, width], dtype=np.int64),
    )
    print(f"      Camera poses saved: {args.output / '000_poses.npz'}")

    from omegaconf import OmegaConf
    from gae.pipeline import load_backbone, load_codec, load_flow, GAE, _read_latent_stats, _resolve_path

    # ── [2/5] Encode Reference Images into GAE Multi-View Latent ──
    _log_memory("Before Stage 2 (GAE Reference Encode)")
    print(f"\n[2/5] Loading DA3 backbone & encoding {N} reference images...")
    backbone = _cast(load_backbone(str(args.config), device), args.backbone_precision)
    bb_dtype = _DTYPES[args.backbone_precision]

    codec = _cast(load_codec(str(args.codec_config), str(args.codec_ckpt), device), args.codec_precision)

    flow_cfg = OmegaConf.load(str(args.config))
    stats_path = flow_cfg.get("latent_stats")
    latent_mean = latent_std = None
    if stats_path:
        resolved = _resolve_path(str(stats_path), args.checkpoint_dir)
        if resolved.is_file():
            latent_mean, latent_std = _read_latent_stats(str(resolved))

    model = GAE(backbone, codec, flow=None, latent_mean=latent_mean, latent_std=latent_std).to(device)

    # Tensor shape: (1, N, 3, H, W)
    imgs_rgb = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in cv_images])
    img_tensor = torch.from_numpy(imgs_rgb.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    img_tensor = img_tensor.unsqueeze(0).to(device=device, dtype=bb_dtype)

    with torch.inference_mode():
        z_ref = model.encode(img_tensor)[0]   # (N, C, h, w)
    print(f"      Encoded {N} views into latent array: {tuple(z_ref.shape)}")

    # Offload heavy DA3 ViT encoder from VRAM
    from gae.pipeline import _resolve_backbone_norm
    backbone_norm = _resolve_backbone_norm(backbone)
    heavy_vit = backbone.encoder
    del backbone.encoder
    _free(heavy_vit, stage="DA3 ViT Encoder")
    print("      DA3 ViT offloaded from VRAM (DPT head retained).")
    _log_memory("After Stage 2 (ViT Offloaded)")

    # ── [3/5] Compute Text Prompt Embeddings (Qwen3) ──
    _log_memory("Before Stage 3 (Text Encode)")
    print("\n[3/5] Loading Qwen3 text encoder...")
    from stage2.models.text_encoder import Qwen3TextEncoder
    text_cfg = flow_cfg.get("text_encoder", {})
    text_enc = Qwen3TextEncoder(
        model_name=str(text_cfg.get("model_name", "Qwen/Qwen3-0.6B")),
        max_length=int(text_cfg.get("max_length", 256)),
    ).to(device)
    text_enc = _cast(text_enc, args.text_precision).eval()
    text_enc.requires_grad_(False)
    with torch.inference_mode():
        ref_global = text_enc([args.prompt])["tokens"]
        cfg_null = text_enc([""])["tokens"]
    _free(text_enc, stage="Qwen3 Text Encoder")
    print("      Text encoder offloaded from VRAM.")
    _log_memory("After Stage 3 (Text Offloaded)")

    # ── [4/5] Construct Plücker Rays & Denoise Novel Wide View ──
    _log_memory("Before Stage 4 (Flow DiT Setup)")
    print("\n[4/5] Building Plücker rays & sampling wide-angle view with Flow DiT...")
    from eval_generation import build_plucker_from_cameras
    training_cfg = flow_cfg.get("training", {})
    pose_origin = str(training_cfg.get("pose_origin", "first"))
    pose_origin_idx = 0 if pose_origin == "first" else -1
    pose_translation_norm = str(training_cfg.get("pose_translation_norm", "none"))

    stage2_params = flow_cfg.get("stage_2", {}).get("params", {})
    s_patch = int(stage2_params.get("patch_size", 1))
    latent_h, latent_w = height // 14, width // 14
    plucker_h, plucker_w = latent_h // s_patch, latent_w // s_patch

    plucker_6d = build_plucker_from_cameras(
        all_K, all_c2w, height, width, patches_y=plucker_h, patches_x=plucker_w,
        device=device, origin_idx=pose_origin_idx,
        translation_norm=pose_translation_norm, pose_scale=None,
    ).to(dtype=_DTYPES[args.flow_precision])

    flow = _cast(load_flow(str(args.config), str(args.flow_ckpt), device), args.flow_precision)
    model.flow = flow
    model._configure_sampler(str(args.config))

    flow_dtype = _DTYPES[args.flow_precision]
    use_amp = device.type == "cuda" and (flow_dtype in (torch.float16, torch.bfloat16))

    _log_memory("Before Stage 4 Sampling (Flow DiT on GPU)")
    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=flow_dtype if use_amp else None):
            z_out = model.sample(
                z_ref=z_ref.to(flow_dtype),
                total_views=total_views,
                cond_num=cond_num,
                plucker_6d=plucker_6d,
                ref_global=ref_global.to(flow_dtype) if ref_global is not None else None,
                cfg_uncond_ref_global=cfg_null.to(flow_dtype) if cfg_null is not None else None,
                num_steps=args.sample_steps,
                cfg_scale=args.cfg_scale,
                seed=args.seed,
            )

    model.flow = None
    _free(flow, stage="Flow DiT Model")
    print("      Flow DiT offloaded from VRAM.")
    _log_memory("After Stage 4 (Flow DiT Offloaded)")

    # ── [5/5] Decode Single Combined Wide-Angle Image & 3D Geometry ──
    _log_memory("Before Stage 5 (VAE & DPT Geometry Decode)")
    print("\n[5/5] Decoding final wide-angle photorealistic image...")
    z_dec = z_out.to(_DTYPES[args.codec_precision])
    with torch.inference_mode():
        trunk_seq, h_p, w_p = codec._decode_trunk(z_dec)
        rgb = model._denormalize_rgb(codec.rgb_head(trunk_seq, h_p, w_p, num_views=total_views))

        c = trunk_seq.shape[-1]
        feat_recon = codec.dec_conv(trunk_seq.permute(0, 2, 1).reshape(-1, c, h_p, w_p))
        feats_split = codec.denormalize_and_split(feat_recon)
        from utils.dpt_helpers import _format_recon_for_dpt
        dpt_in = _format_recon_for_dpt(feats_split, backbone_norm, z_dec.shape[0])
        with torch.autocast(device_type=device.type, enabled=False):
            geo = backbone.rae_cl_decoder(dpt_in, height, width, patch_start_idx=0)

    # Extract target wide-angle image (the novel view at index N)
    wide_angle_tensor = rgb[cond_num].float().clamp(0, 1)
    wide_np = (wide_angle_tensor.permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    wide_bgr = cv2.cvtColor(wide_np, cv2.COLOR_RGB2BGR)

    out_image_path = args.output / "000_wide_angle_view.png"
    cv2.imwrite(str(out_image_path), wide_bgr)
    print(f"\n=======================================================")
    print(f" [SUCCESS] Wide-Angle Image Saved: {out_image_path}")
    print(f"=======================================================")

    # Also save 3D Point Cloud (.ply) if requested
    if args.save_pointcloud:
        out = {"rgb": rgb, **geo}
        _ = model.save_outputs(out, args.output, stem="000", pc_stride=4)
        print(f"      3D Scene Point Cloud saved: {args.output / '000_pred_pointcloud.ply'}")

    _free(codec, stage="Codec")
    del backbone_norm, backbone
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _log_memory("Pipeline Finished")
    print(f"All outputs ready in: {args.output}\n")
    return 0


def main() -> int:
    args = parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())


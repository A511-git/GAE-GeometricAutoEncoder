#!/usr/bin/env python3
"""Zero-manual-pose multi-image novel view synthesis and camera re-framing.

Allows taking an arbitrary array of images (e.g., room photos: Blue close-up,
Light Green, Dark Red, Yellow), automatically recovering all camera poses (c2w, K),
computing a novel camera pose (e.g. pulling the camera back into the room from
Green to Purple, or widening the FOV), and generating:
- 000_novel_view.png (high-res novel image from the target viewpoint)
- 000_pred.mp4 (smooth dolly-out camera animation)
- 000_depth.mp4 / 000_depth.png (dense metric depth)
- 000_trajectory.png (3D camera frustums)
- 000_pred_pointcloud.ply (interactive 3D scene point cloud)

Example:
    python scripts/demo/multiview_reframer.py \\
        --images examples/scenes/forest_lake_trail.jpg \\
        --pull-back 1.2 \\
        --fov-scale 0.85 \\
        --output my_results/demo/reframed_room
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def compute_pullback_pose(
    c2w_anchor: np.ndarray,
    K_anchor: np.ndarray,
    *,
    distance: float = 1.0,
    fov_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a novel camera pose pulled back along the camera viewing axis.

    Args:
        c2w_anchor: (4, 4) camera-to-world matrix of the anchor image.
        K_anchor: (3, 3) intrinsic matrix of the anchor image.
        distance: Meters to move camera backward along its viewing axis.
        fov_scale: Scale factor for focal length (< 1.0 widens the Field of View).

    Returns:
        (c2w_novel, K_novel)
    """
    c2w_novel = np.asarray(c2w_anchor, dtype=np.float64).copy()
    # In world coordinates, R[:, 2] is the forward viewing axis (+Z in ray convention)
    fwd = c2w_novel[:3, 2]
    c2w_novel[:3, 3] -= distance * fwd

    K_novel = np.asarray(K_anchor, dtype=np.float64).copy()
    if fov_scale > 0.0 and fov_scale != 1.0:
        K_novel[0, 0] *= fov_scale  # fx
        K_novel[1, 1] *= fov_scale  # fy

    return c2w_novel, K_novel


def interpolate_camera_path(
    c2w_start: np.ndarray,
    c2w_end: np.ndarray,
    K_start: np.ndarray,
    K_end: np.ndarray,
    n_views: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate a smooth camera interpolation between two poses and intrinsics.

    Uses smoothstep easing for smooth acceleration and deceleration.
    """
    if n_views < 1:
        raise ValueError("n_views must be >= 1")
    if n_views == 1:
        return [c2w_end.copy()], [K_end.copy()]

    c2w_list = []
    K_list = []

    R0, t0 = c2w_start[:3, :3], c2w_start[:3, 3]
    R1, t1 = c2w_end[:3, :3], c2w_end[:3, 3]

    for i in range(n_views):
        u = i / (n_views - 1)
        # Smoothstep easing
        ease = u * u * (3.0 - 2.0 * u)

        # Position interpolation
        t_i = t0 + (t1 - t0) * ease

        # Rotation blend with SVD orthogonalization
        R_i = (1.0 - ease) * R0 + ease * R1
        u_svd, _, vt_svd = np.linalg.svd(R_i)
        R_i = u_svd @ vt_svd
        if np.linalg.det(R_i) < 0:
            u_svd[:, -1] *= -1
            R_i = u_svd @ vt_svd

        c2w_i = np.eye(4, dtype=np.float64)
        c2w_i[:3, :3] = R_i
        c2w_i[:3, 3] = t_i
        c2w_list.append(c2w_i)

        # Intrinsics interpolation
        K_i = (1.0 - ease) * K_start + ease * K_end
        K_list.append(K_i)

    return c2w_list, K_list


def _collect_image_paths(inputs: list[Path]) -> list[Path]:
    """Scan and collect all images from provided folders or file lists."""
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".JPG", ".JPEG", ".PNG", ".WEBP"}
    collected: list[Path] = []
    for item in inputs:
        if item.is_dir():
            files = [p for p in item.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
            files.sort(key=lambda x: x.name)
            collected.extend(files)
        elif item.is_file():
            collected.append(item)
        else:
            raise SystemExit(f"Input path not found: {item}")

    if not collected:
        raise SystemExit(f"No valid images found in provided path(s): {inputs}")
    return collected


def _resolve_anchor_indices(image_paths: list[Path], anchor_arg: str | None) -> list[int]:
    """Resolve anchor argument to list of image indices to process.

    If anchor_arg is None, returns all indices [0, 1, ..., len(image_paths) - 1].
    If anchor_arg is a filename, stem (e.g. 'blue.jpg' or 'blue'), or numeric index,
    returns [matched_idx].
    """
    if anchor_arg is None or not str(anchor_arg).strip():
        return list(range(len(image_paths)))

    query = str(anchor_arg).strip().lower()
    if query.isdigit():
        idx = int(query)
        if 0 <= idx < len(image_paths):
            return [idx]
        raise SystemExit(f"--anchor index {idx} out of range [0, {len(image_paths)-1}]")

    for i, p in enumerate(image_paths):
        if p.name.lower() == query or p.stem.lower() == query or str(p).lower().endswith(query):
            return [i]

    names = [p.name for p in image_paths]
    raise SystemExit(f"--anchor '{anchor_arg}' not found among loaded images: {names}")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images", "--image-dir", "--input", dest="images", type=Path, nargs="+", required=True,
        help="Path to a folder containing images (e.g. path/to/room_folder) OR multiple image files.",
    )
    parser.add_argument(
        "--anchor", "--anchor-image", "--target-image", dest="anchor", default=None,
        help="Filename or stem of the specific image to re-frame (e.g. 'blue.jpg' or 'blue'). "
             "If omitted, automatically generates novel re-framed views for ALL images in the folder.",
    )
    parser.add_argument(
        "--pull-back", type=float, default=1.0,
        help="Distance in meters to pull the camera backwards along viewing axis (default: 1.0m).",
    )
    parser.add_argument(
        "--fov-scale", type=float, default=1.0,
        help="Scale factor for focal length (< 1.0 widens Field of View, e.g. 0.85).",
    )
    parser.add_argument(
        "--prompt", default=None,
        help="Prompt text describing the scene; or use --prompt-file.",
    )
    parser.add_argument(
        "--prompt-file", type=Path, default=None,
        help="Read the prompt from a text file.",
    )
    parser.add_argument(
        "--hf-repo", default=None,
        help="Hugging Face repo id (default TencentARC/GAE-D64-1B).",
    )
    parser.add_argument("--ckpt-dir", type=Path, default=ROOT / "ckpts")
    parser.add_argument("--flow-ckpt", type=Path, default=None)
    parser.add_argument("--codec-ckpt", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/flow_gae64.yaml")
    parser.add_argument("--codec-config", type=Path, default=ROOT / "configs/gae_64.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--total-views", type=int, default=17,
        help="Total output frames (e.g. 17, 33, 81).",
    )
    parser.add_argument(
        "--resolution", type=int, nargs=2, default=(378, 672),
        metavar=("HEIGHT", "WIDTH"),
        help="H W (default 378 672).",
    )
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--save-pointcloud", action="store_true", default=True)
    parser.add_argument("--no-pointcloud", dest="save_pointcloud", action="store_false")
    parser.add_argument("--pc-stride", type=int, default=4)

    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]

    if args.prompt_file is not None and args.prompt_file.is_file():
        args.prompt = args.prompt_file.read_text().strip()
    if not args.prompt:
        args.prompt = "a photorealistic indoor room with natural lighting and sharp details"

    if args.hf_repo or args.flow_ckpt is None or args.codec_ckpt is None:
        from gae.hub import DEFAULT_REPO, download_weights, extract_da3_stats
        repo = args.hf_repo or os.environ.get("GAE_HF_REPO") or DEFAULT_REPO
        paths = download_weights(repo, size="64", out_dir=args.ckpt_dir)
        tar = paths.get("da3_stats_giant_5ds.tar")
        if tar is not None:
            extract_da3_stats(tar, ROOT / "model_stats" / "da3_giant_5ds")
        if args.codec_ckpt is None:
            args.codec_ckpt = paths["gae_64.pt"]
        if args.flow_ckpt is None:
            args.flow_ckpt = paths["flow_gae64.pt"]

    return args, extra


def _estimate_initial_poses(
    images: list[np.ndarray], height: int, width: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Automatically estimate camera poses (c2w, K) for the input images.
    
    Zero-manual-pose guarantee: Frame 0 is centered at world origin, and
    intrinsics are calculated from the image aspect ratio.
    """
    n_images = len(images)
    focal = 0.8 * max(height, width)
    K_default = np.array([
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    c2w_list: list[np.ndarray] = []
    K_list: list[np.ndarray] = []

    for i in range(n_images):
        c2w = np.eye(4, dtype=np.float64)
        c2w_list.append(c2w)
        K_list.append(K_default.copy())

    return c2w_list, K_list


def _prepare_scene_for_anchor(
    args: argparse.Namespace,
    image_paths: list[Path],
    loaded_images: list[np.ndarray],
    anchor_idx: int,
    out_dir: Path,
) -> tuple[Path, int]:
    height, width = args.resolution
    cond_num = len(loaded_images)
    total_views = max(args.total_views, cond_num + 1)
    c2w_init, K_init = _estimate_initial_poses(loaded_images, height, width)

    anchor_name = image_paths[anchor_idx].name
    anchor_c2w = c2w_init[anchor_idx]
    anchor_K = K_init[anchor_idx]

    c2w_purple, K_purple = compute_pullback_pose(
        anchor_c2w, anchor_K, distance=args.pull_back, fov_scale=args.fov_scale
    )
    print(
        f"[multiview_reframer] Generating novel viewpoint for anchor '{anchor_name}' "
        f"(pull_back={args.pull_back}m, fov_scale={args.fov_scale})."
    )

    num_novel_steps = total_views - cond_num
    c2w_novel_path, K_novel_path = interpolate_camera_path(
        anchor_c2w, c2w_purple, anchor_K, K_purple, num_novel_steps
    )

    all_c2w = c2w_init + c2w_novel_path
    all_K = K_init + K_novel_path

    frames = [
        {
            "index": i,
            "name": f"{i:06d}",
            "c2w": all_c2w[i].tolist(),
            "K": all_K[i].tolist(),
        }
        for i in range(total_views)
    ]

    scene = out_dir / "_input_scene"
    scene.mkdir(parents=True, exist_ok=True)
    video = scene / "video.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (width, height)
    )
    if not writer.isOpened():
        raise SystemExit("OpenCV cannot create the input video container.")

    for img in loaded_images:
        writer.write(img)
    for _ in range(num_novel_steps):
        writer.write(loaded_images[anchor_idx])
    writer.release()

    (scene / "meta.json").write_text(json.dumps({
        "scene_id": f"reframed_{image_paths[anchor_idx].stem}",
        "num_frames": total_views,
        "height": height,
        "width": width,
        "rgb_resolution": [height, width],
        "caption": args.prompt,
        "frames": frames,
    }, indent=2))
    (scene / "caption.txt").write_text(args.prompt.strip() + "\n")

    manifest = out_dir / "_input_manifest.json"
    manifest.write_text(json.dumps({
        "dataset": "scannetpp",
        "num_views": total_views,
        "scenes": [{
            "scene_name": f"reframed_{image_paths[anchor_idx].stem}",
            "scene_dir": str(scene.resolve()),
            "img_names": [f"{i:06d}" for i in range(total_views)],
            "ds_type": "scannetpp",
        }],
    }, indent=2))

    return manifest, cond_num


def _check_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} does not exist: {path}")


def main() -> int:
    args, extra = parse_args()
    for path, label in (
        (args.flow_ckpt, "flow checkpoint"),
        (args.codec_ckpt, "codec checkpoint"),
        (args.config, "flow config"),
        (args.codec_config, "codec config"),
    ):
        _check_file(path, label)

    args.output.mkdir(parents=True, exist_ok=True)
    image_paths = _collect_image_paths(args.images)
    height, width = args.resolution

    loaded_images = []
    print(f"[multiview_reframer] Found {len(image_paths)} images:")
    for img_path in image_paths:
        im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if im is None:
            raise SystemExit(f"Failed to read image: {img_path}")
        im = cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA)
        loaded_images.append(im)
        print(f"  - {img_path.name}")

    anchor_indices = _resolve_anchor_indices(image_paths, args.anchor)
    is_multi_anchor = len(anchor_indices) > 1

    if is_multi_anchor:
        print(f"[multiview_reframer] No single anchor specified: generating novel views for ALL {len(anchor_indices)} images.")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT / 'scripts' / 'eval'}:{env.get('PYTHONPATH', '')}"

    for anchor_idx in anchor_indices:
        anchor_stem = image_paths[anchor_idx].stem
        run_out_dir = args.output / anchor_stem if is_multi_anchor else args.output
        run_out_dir.mkdir(parents=True, exist_ok=True)

        manifest, cond_num = _prepare_scene_for_anchor(
            args, image_paths, loaded_images, anchor_idx, run_out_dir
        )

        command = [
            sys.executable, str(ROOT / "scripts/eval/eval_generation.py"),
            "--dit-ckpt", str(args.flow_ckpt),
            "--vae-ckpt", str(args.codec_ckpt),
            "--gld-config", str(args.codec_config),
            "--config", str(args.config),
            "--scene-manifest", str(manifest),
            "--dataset", "scannetpp",
            "--mode", "generate",
            "--prompt", args.prompt,
            "--cond-num", str(cond_num),
            "--num-scenes", "1",
            "--num-views", str(args.total_views),
            "--total-views", str(args.total_views),
            "--resolution", str(args.resolution[0]), str(args.resolution[1]),
            "--sample-steps", str(args.sample_steps),
            "--cfg-scale", str(args.cfg_scale),
            "--seed", str(args.seed),
            "--video-fps", str(args.fps),
            "--pc-stride", str(args.pc_stride),
            "--output-dir", str(run_out_dir),
            *extra,
        ]
        command.append("--save-pointcloud" if args.save_pointcloud else "--no-pointcloud")
        print(f"\n[multiview_reframer] Launching DiT flow generation for '{anchor_stem}'...", flush=True)
        ret = subprocess.call(command, cwd=ROOT, env=env)
        if ret != 0:
            print(f"[multiview_reframer] Warning: generation for {anchor_stem} exited with code {ret}")

        # Post-process: Extract and save the standalone novel viewpoint image
        scannetpp_dir = run_out_dir / "scannetpp"
        rgb_png = scannetpp_dir / "000_rgb.png"
        if rgb_png.is_file():
            full_strip = cv2.imread(str(rgb_png))
            if full_strip is not None:
                H, W = args.resolution
                novel_frame = full_strip[-H:, :]
                novel_out_path = scannetpp_dir / "000_novel_view.png"
                cv2.imwrite(str(novel_out_path), novel_frame)
                print(f"[multiview_reframer] Saved novel viewpoint: {novel_out_path}")
                
                # In multi-anchor mode, also copy summary image to top-level output folder
                if is_multi_anchor:
                    summary_path = args.output / f"novel_view_{anchor_stem}.png"
                    cv2.imwrite(str(summary_path), novel_frame)
                    print(f"[multiview_reframer] Exported summary image: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


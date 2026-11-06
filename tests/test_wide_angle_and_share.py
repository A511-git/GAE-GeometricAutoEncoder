"""Unit tests for folder-based wide-angle synthesis and Gradio --share in Camera Studio."""
import ast
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_wide_angle_collect_images_from_folder(tmp_path):
    """Test that wide_angle_synthesizer collects all images inside a folder in natural numerical order."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts" / "demo"))
    from wide_angle_synthesizer import _collect_images, _natural_sort_key

    # Create dummy images with unordered names: img_10.jpg, img_1.jpg, img_2.png, sub.txt
    img_dir = tmp_path / "room_photos"
    img_dir.mkdir()
    (img_dir / "img_10.jpg").write_text("fake image")
    (img_dir / "img_1.jpg").write_text("fake image")
    (img_dir / "img_2.png").write_text("fake image")
    (img_dir / "img_3.webp").write_text("fake image")
    (img_dir / "notes.txt").write_text("not an image")

    # 1. Pass folder as a path in paths list (--images <folder>)
    collected = _collect_images([img_dir])
    assert len(collected) == 4
    names = [p.name for p in collected]
    assert names == ["img_1.jpg", "img_2.png", "img_3.webp", "img_10.jpg"]

    # 2. Pass folder via folder parameter (--image-dir <folder>)
    collected_dir = _collect_images(None, folder=img_dir)
    assert len(collected_dir) == 4
    assert [p.name for p in collected_dir] == ["img_1.jpg", "img_2.png", "img_3.webp", "img_10.jpg"]

    # 3. Pass individual images
    collected_individual = _collect_images([img_dir / "img_1.jpg", img_dir / "img_2.png"])
    assert len(collected_individual) == 2
    assert [p.name for p in collected_individual] == ["img_1.jpg", "img_2.png"]


def test_wide_angle_cli_folder_argument(monkeypatch, tmp_path):
    """Test parse_args accepts a folder for --images or --image-dir."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts" / "demo"))
    from wide_angle_synthesizer import parse_args

    img_dir = tmp_path / "scene_5_views"
    img_dir.mkdir()
    for i in range(5):
        (img_dir / f"face_{i}.jpg").write_text("dummy")

    out_dir = tmp_path / "out"

    # Test --images <folder>
    monkeypatch.setattr(
        sys, "argv",
        ["wide_angle_synthesizer.py", "--images", str(img_dir), "--output", str(out_dir)]
    )
    args = parse_args()
    assert len(args.collected_images) == 5
    assert [p.name for p in args.collected_images] == [f"face_{i}.jpg" for i in range(5)]

    # Test --image-dir <folder>
    monkeypatch.setattr(
        sys, "argv",
        ["wide_angle_synthesizer.py", "--image-dir", str(img_dir), "--output", str(out_dir)]
    )
    args2 = parse_args()
    assert len(args2.collected_images) == 5


def test_camera_studio_share_argument():
    """Verify camera_studio.py has CLI parser supporting --share."""
    code = (ROOT / "camera_studio.py").read_text(encoding="utf-8")
    tree = ast.parse(code)

    # Check that main() function exists and configures --share
    main_func = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"), None)
    assert main_func is not None, "main() not found in camera_studio.py"

    main_src = ast.get_source_segment(code, main_func)
    assert "--share" in main_src
    assert "share=should_share" in main_src


def test_distributed_engine_share_argument():
    """Verify distributed_engine.py supports --share argument."""
    code = (ROOT / "scripts" / "demo" / "distributed_engine.py").read_text(encoding="utf-8")
    assert "--share" in code
    assert "share=a.share" in code


def test_camera_studio_no_caption_argument():
    """Verify camera_studio.py supports --no-caption argument."""
    code = (ROOT / "camera_studio.py").read_text(encoding="utf-8")
    assert "--no-caption" in code
    assert 'os.environ["GAE_CAPTION"] = "0"' in code


def test_wide_angle_synthesizer_pose_model_and_memory_logging():
    """Verify wide_angle_synthesizer.py supports --pose-model and _log_memory."""
    code = (ROOT / "scripts" / "demo" / "wide_angle_synthesizer.py").read_text(encoding="utf-8")
    assert "--pose-model" in code
    assert "def _log_memory" in code
    assert "RAM:" in code


def test_gaussian_fourier_embedding_mixed_precision():
    """Verify GaussianFourierEmbedding handles Float timestep with BFloat16 weights without mat1/mat2 dtype error."""
    import ast
    import torch
    import torch.nn as nn

    code = (ROOT / "src" / "stage2" / "models" / "model_utils.py").read_text(encoding="utf-8")
    tree = ast.parse(code)
    cls_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GaussianFourierEmbedding"
    )
    cls_code = ast.get_source_segment(code, cls_node)
    local_ns = {"nn": nn, "torch": torch}
    exec(cls_code, local_ns)
    GaussianFourierEmbedding = local_ns["GaussianFourierEmbedding"]

    embedder = GaussianFourierEmbedding(hidden_size=64, embedding_size=32)
    embedder = embedder.to(dtype=torch.bfloat16)

    # 1. t is float32, embedder is bfloat16 (The exact failure case from Kaggle)
    t_float = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32)
    out = embedder(t_float)
    assert out.shape == (3, 64)
    assert out.dtype == torch.bfloat16

    # 2. t is bfloat16, embedder is bfloat16
    t_bf16 = torch.tensor([0.25, 0.5, 0.75], dtype=torch.bfloat16)
    out_bf16 = embedder(t_bf16)
    assert out_bf16.shape == (3, 64)
    assert out_bf16.dtype == torch.bfloat16


def test_sm75_bfloat16_precision_fallback_and_attention_fallback():
    """Verify precision fallback for SM < 8.0 and that attention code contains try/except fallback to SDPA."""
    code_plucker = (ROOT / "src" / "stage2" / "models" / "plucker_attention.py").read_text(encoding="utf-8")
    assert "except (NotImplementedError, RuntimeError, ValueError):" in code_plucker
    assert "x_attn = F.scaled_dot_product_attention(" in code_plucker
    assert "needs_fp16" in code_plucker

    code_model_utils = (ROOT / "src" / "stage2" / "models" / "model_utils.py").read_text(encoding="utf-8")
    assert "except (NotImplementedError, RuntimeError, ValueError):" in code_model_utils
    assert "x = F.scaled_dot_product_attention(" in code_model_utils
    assert "needs_fp16" in code_model_utils

    code_synth = (ROOT / "scripts" / "demo" / "wide_angle_synthesizer.py").read_text(encoding="utf-8")
    assert "Auto-fallback" in code_synth
    assert "major >= 8" in code_synth


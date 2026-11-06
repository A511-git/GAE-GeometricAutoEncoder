"""Tests for centralized --checkpoint-dir / --ckpt-dir resolution and flattening across GAE."""
import ast
import os
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _extract_function_from_file(file_path: Path, func_name: str, extra_globals: dict | None = None):
    """Parse a single function definition from a file and compile it into a callable."""
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    func_node = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in (func_name,)),
        None,
    )
    if func_node is None:
        raise ValueError(f"Function '{func_name}' not found in {file_path}")

    mod = ast.Module(body=[func_node], type_ignores=[])
    code = compile(mod, filename=str(file_path), mode="exec")
    ns = {"_ROOT": ROOT, "ROOT": ROOT, "Path": Path}
    if extra_globals:
        ns.update(extra_globals)
    exec(code, ns)
    return ns[func_name]


def test_resolve_checkpoint_dir_parent_nesting(tmp_path):
    from gae.hub import resolve_checkpoint_dir

    # 1. Passing a parent directory like /tmp or /kaggle/working appends 'checkpoints'
    parent = tmp_path / "kaggle_working"
    parent.mkdir()
    resolved = resolve_checkpoint_dir(parent)
    assert resolved == parent / "checkpoints"
    assert resolved.is_dir()

    # 2. Passing a path that already ends in 'checkpoints' does not double-nest
    explicit = tmp_path / "custom" / "checkpoints"
    resolved_explicit = resolve_checkpoint_dir(explicit)
    assert resolved_explicit == explicit
    assert resolved_explicit.is_dir()

    # 3. Passing a path ending in 'ckpts' uses it directly
    ckpts_path = tmp_path / "custom" / "ckpts"
    resolved_ckpts = resolve_checkpoint_dir(ckpts_path)
    assert resolved_ckpts == ckpts_path
    assert resolved_ckpts.is_dir()


def test_resolve_model_file_flattening(tmp_path):
    from gae.hub import resolve_model_file

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    # Simulate flattened files in checkpoint_dir
    fake_gae = ckpt_dir / "gae_64.pt"
    fake_gae.write_text("dummy gae")
    fake_stats = ckpt_dir / "latent_stats_gae_64.pt"
    fake_stats.write_text("dummy stats")

    # A yaml config specifying "ckpts/gae_64.pt" should resolve to the flattened file in ckpt_dir
    found_gae = resolve_model_file("ckpts/gae_64.pt", ckpt_dir)
    assert found_gae == fake_gae

    # A yaml config specifying "ckpts/latent_stats_gae_64.pt" should resolve to flattened stats
    found_stats = resolve_model_file("ckpts/latent_stats_gae_64.pt", ckpt_dir)
    assert found_stats == fake_stats


def test_eval_generation_checkpoint_dir_cli(monkeypatch, tmp_path):
    """Verify eval_generation.py parse_args accepts --checkpoint-dir and resolves dit/vae/latent_stats."""
    import argparse
    parse_args_eval = _extract_function_from_file(
        ROOT / "scripts" / "eval" / "eval_generation.py",
        "parse_args",
        extra_globals={"argparse": argparse, "os": os, "sys": sys},
    )

    ckpt_dir = tmp_path / "eval_ckpts" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    fake_flow = ckpt_dir / "flow_gae64.pt"
    fake_flow.write_text("dummy flow")

    monkeypatch.setattr(
        sys, "argv",
        ["eval_generation.py", "--checkpoint-dir", str(ckpt_dir), "--mode", "generate"]
    )
    args = parse_args_eval()
    assert args.checkpoint_dir == ckpt_dir
    assert Path(args.dit_ckpt) == fake_flow


def test_generate_py_checkpoint_dir_cli(monkeypatch, tmp_path):
    """Verify generate.py accepts --checkpoint-dir and propagates it."""
    from scripts.demo.generate import parse_args

    ckpt_dir = tmp_path / "gen_ckpts"
    fake_pt = tmp_path / "dummy.pt"
    fake_pt.write_text("dummy")

    monkeypatch.setattr(
        sys, "argv",
        [
            "generate.py", "--image", "dummy.jpg", "--prompt", "test",
            "--checkpoint-dir", str(ckpt_dir), "--flow-ckpt", str(fake_pt),
            "--codec-ckpt", str(fake_pt), "--output", str(tmp_path / "out")
        ]
    )
    args, _ = parse_args()
    assert args.checkpoint_dir == ckpt_dir / "checkpoints"
    assert args.ckpt_dir == args.checkpoint_dir


def test_wide_angle_synthesizer_checkpoint_dir_cli(monkeypatch, tmp_path):
    """Verify wide_angle_synthesizer.py accepts --checkpoint-dir and --ckpt-dir."""
    from scripts.demo.wide_angle_synthesizer import parse_args

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    (img_dir / "1.jpg").write_text("dummy")
    (img_dir / "2.jpg").write_text("dummy")

    ckpt_dir = tmp_path / "wide_ckpts"
    monkeypatch.setattr(
        sys, "argv",
        ["wide_angle_synthesizer.py", "--images", str(img_dir), "--checkpoint-dir", str(ckpt_dir), "--output", str(tmp_path / "out")]
    )
    args = parse_args()
    assert args.checkpoint_dir == ckpt_dir / "checkpoints"


def test_camera_studio_checkpoint_dir_cli():
    """Verify camera_studio.py CLI supports --checkpoint-dir."""
    code = (ROOT / "camera_studio.py").read_text(encoding="utf-8")
    tree = ast.parse(code)
    main_func = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
    assert main_func is not None
    src = ast.get_source_segment(code, main_func)
    assert "--checkpoint-dir" in src
    assert "CKPT_DIR" in src

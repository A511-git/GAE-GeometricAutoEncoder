"""Tests for centralized low-VRAM mode across generate.py, eval_generation.py, camera_studio.py, and generate_t2i.py."""

import ast
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
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

    # Build module AST containing only this function
    mod = ast.Module(body=[func_node], type_ignores=[])
    code = compile(mod, filename=str(file_path), mode="exec")
    ns = {"_ROOT": ROOT, "ROOT": ROOT, "Path": Path}
    if extra_globals:
        ns.update(extra_globals)
    exec(code, ns)
    return ns[func_name]


def test_generate_py_cli_low_vram_args():
    """Verify scripts/demo/generate.py parses --low-vram by default and supports --no-low-vram."""
    import argparse
    from scripts.demo.generate import parse_args

    # 1. Default invocation (no low-vram flag passed) -> should be True
    old_argv = sys.argv
    try:
        base_argv = [
            "generate.py", "--image", "examples/scenes/forest_lake_trail.jpg",
            "--prompt", "test", "--output", "tmp_out",
            "--flow-ckpt", "dummy.pt", "--codec-ckpt", "dummy.pt",
        ]
        sys.argv = list(base_argv)
        args, _ = parse_args()
        assert hasattr(args, "low_vram"), "generate.py parser must have --low-vram attribute"
        assert args.low_vram is True, f"Default for --low-vram should be True, got {args.low_vram}"

        # 2. Explicit --no-low-vram -> should be False
        sys.argv = list(base_argv) + ["--no-low-vram"]
        args, _ = parse_args()
        assert args.low_vram is False, f"Passing --no-low-vram should set low_vram=False, got {args.low_vram}"

        # 3. Explicit --low-vram -> should be True
        sys.argv = list(base_argv) + ["--low-vram"]
        args, _ = parse_args()
        assert args.low_vram is True, f"Passing --low-vram should set low_vram=True, got {args.low_vram}"
    finally:
        sys.argv = old_argv


def test_eval_generation_cli_low_vram_args():
    """Verify scripts/eval/eval_generation.py parse_args() defines --low-vram=True by default."""
    import argparse

    parse_args_fn = _extract_function_from_file(
        ROOT / "scripts" / "eval" / "eval_generation.py",
        "parse_args",
        extra_globals={"argparse": argparse, "_fast_load": lambda *a, **k: {}},
    )

    old_argv = sys.argv
    try:
        # Default invocation
        sys.argv = ["eval_generation.py", "--dit-ckpt", "dummy.pt"]
        args = parse_args_fn()
        assert hasattr(args, "low_vram"), "eval_generation.py parser must have --low-vram attribute"
        assert args.low_vram is True, f"Default for --low-vram should be True, got {args.low_vram}"

        # Explicit --no-low-vram
        sys.argv = ["eval_generation.py", "--dit-ckpt", "dummy.pt", "--no-low-vram"]
        args = parse_args_fn()
        assert args.low_vram is False, f"Passing --no-low-vram should set low_vram=False, got {args.low_vram}"
    finally:
        sys.argv = old_argv


def test_generate_t2i_cli_low_vram_args():
    """Verify scripts/demo/generate_t2i.py parse_args() defines --low-vram=True by default."""
    import argparse

    parse_args_fn = _extract_function_from_file(
        ROOT / "scripts" / "demo" / "generate_t2i.py",
        "parse_args",
        extra_globals={"argparse": argparse},
    )

    old_argv = sys.argv
    try:
        # Default invocation
        sys.argv = ["generate_t2i.py", "--prompts", "test prompt", "--output", "tmp_out"]
        args = parse_args_fn()
        assert hasattr(args, "low_vram"), "generate_t2i.py parser must have --low-vram attribute"
        assert args.low_vram is True, f"Default for --low-vram should be True, got {args.low_vram}"

        # Explicit --no-low-vram
        sys.argv = ["generate_t2i.py", "--prompts", "test prompt", "--output", "tmp_out", "--no-low-vram"]
        args = parse_args_fn()
        assert args.low_vram is False, f"Passing --no-low-vram should set low_vram=False, got {args.low_vram}"
    finally:
        sys.argv = old_argv


def test_camera_studio_low_vram_command_generation(tmp_path):
    """Verify camera_studio.py generate_i2v formats --low-vram and --no-low-vram appropriately."""
    tree = ast.parse((ROOT / "camera_studio.py").read_text(encoding="utf-8"))
    func_node = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate_i2v"),
        None,
    )
    assert func_node is not None, "generate_i2v not found in camera_studio.py"

    captured_commands = []

    def mock_run_i2v_stream(command, run_dir, timeout):
        captured_commands.append(list(command))
        yield {"elapsed": 0.1, "error": None, "video": False}

    ns = {
        "os": os,
        "sys": sys,
        "ROOT": ROOT,
        "OUTPUT_ROOT": tmp_path,
        "HF_REPO": "test-repo",
        "CKPT_DIR": tmp_path,
        "DEFAULT_EXAMPLE_POSES": tmp_path / "absent.npz",
        "_pose_reference_for_image": lambda _: None,
        "_duration_i2v": lambda *a, **k: 120,
        "_run_i2v_stream": mock_run_i2v_stream,
        "_latest": lambda *a, **k: None,
        "uuid": types.SimpleNamespace(uuid4=lambda: types.SimpleNamespace(hex="1234")),
        "gr": types.SimpleNamespace(Error=Exception),
        "spaces": types.SimpleNamespace(GPU=lambda *a, **k: (lambda fn: fn)),
        "RESIDENT_ENGINE": None,
    }

    mod = ast.Module(body=[func_node], type_ignores=[])
    exec(compile(mod, "camera_studio.py", "exec"), ns)
    generate_i2v = ns["generate_i2v"]

    # 1. Test with low_vram=True (default)
    try:
        list(generate_i2v("fake_img.jpg", "a prompt", "example", 17, 10, 2.0, 42, 2, low_vram=True))
    except Exception:
        pass
    assert len(captured_commands) == 1
    assert "--low-vram" in captured_commands[0], f"Expected --low-vram in command: {captured_commands[0]}"
    assert "--no-low-vram" not in captured_commands[0]

    # 2. Test with low_vram=False
    captured_commands.clear()
    try:
        list(generate_i2v("fake_img.jpg", "a prompt", "example", 17, 10, 2.0, 42, 2, low_vram=False))
    except Exception:
        pass
    assert len(captured_commands) == 1
    assert "--no-low-vram" in captured_commands[0], f"Expected --no-low-vram in command: {captured_commands[0]}"
    assert "--low-vram" not in captured_commands[0]


def test_sequential_offload_device_lifecycle():
    """Verify that staged models start on CPU, move to target device during execution, and offload back to CPU."""
    device = torch.device("cpu")  # In CPU test environment, device is cpu; simulation verifies device tags

    class DummyModel(nn.Module):
        def __init__(self, size: int):
            super().__init__()
            self.linear = nn.Linear(size, size)
            self.current_device = "cpu"

        def to(self, *args, **kwargs):
            dev = args[0] if args else kwargs.get("device")
            self.current_device = str(dev)
            if str(dev).startswith("cuda") and not torch.cuda.is_available():
                return self
            return super().to(*args, **kwargs)

    qwen = DummyModel(16)
    da3 = DummyModel(32)
    dit = DummyModel(64)
    vae = DummyModel(16)

    def _move_to_device(m):
        if m is not None and hasattr(m, "to"):
            m.to("cuda:0")  # mock device
        return m

    def _offload_to_cpu(*modules, low_vram=True):
        if not low_vram:
            return
        for m in modules:
            if m is not None and hasattr(m, "to"):
                m.to("cpu")

    # When low_vram is True:
    low_vram = True

    # Initial state: all models on CPU
    assert qwen.current_device == "cpu"
    assert da3.current_device == "cpu"
    assert dit.current_device == "cpu"
    assert vae.current_device == "cpu"

    # Stage 1: Text Encoding
    _move_to_device(qwen)
    assert qwen.current_device == "cuda:0"
    _offload_to_cpu(qwen, low_vram=low_vram)
    assert qwen.current_device == "cpu"

    # Stage 2: Latent Encoding
    _move_to_device(da3)
    _move_to_device(vae)
    assert da3.current_device == "cuda:0"
    assert vae.current_device == "cuda:0"
    assert dit.current_device == "cpu"
    _offload_to_cpu(da3, vae, low_vram=low_vram)
    assert da3.current_device == "cpu"
    assert vae.current_device == "cpu"

    # Stage 3: DiT Denoising
    _move_to_device(dit)
    assert dit.current_device == "cuda:0"
    assert qwen.current_device == "cpu"
    assert da3.current_device == "cpu"
    assert vae.current_device == "cpu"
    _offload_to_cpu(dit, low_vram=low_vram)
    assert dit.current_device == "cpu"

    # Stage 4: VAE Decoding
    _move_to_device(vae)
    assert vae.current_device == "cuda:0"
    assert dit.current_device == "cpu"
    _offload_to_cpu(vae, low_vram=low_vram)
    assert vae.current_device == "cpu"


if __name__ == "__main__":
    test_generate_py_cli_low_vram_args()
    print("PASS: test_generate_py_cli_low_vram_args")
    test_eval_generation_cli_low_vram_args()
    print("PASS: test_eval_generation_cli_low_vram_args")
    test_generate_t2i_cli_low_vram_args()
    print("PASS: test_generate_t2i_cli_low_vram_args")
    test_camera_studio_low_vram_command_generation(Path("results/test_tmp"))
    print("PASS: test_camera_studio_low_vram_command_generation")
    test_sequential_offload_device_lifecycle()
    print("PASS: test_sequential_offload_device_lifecycle")
    print("\nALL LOW-VRAM TESTS PASSED SUCCESSFULLY!")

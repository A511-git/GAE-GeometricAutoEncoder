"""Download / resolve GAE weights from the Hugging Face Hub."""

from __future__ import annotations

import os
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = os.environ.get("GAE_HF_REPO", "TencentARC/GAE-D64-1B")


def resolve_checkpoint_dir(path: Path | str | None = None) -> Path:
    """Normalize and resolve the checkpoint directory.

    Rules:
    - If path is None or empty:
      - Checks env GAE_CHECKPOINT_DIR or GAE_CKPT_DIR.
      - If none set, defaults to `_REPO_ROOT / "checkpoints"` (or fallback `_REPO_ROOT / "ckpts"` if it already exists).
    - If user passes a parent folder (e.g. `/kaggle/working`, `/kaggle/working/`, `/tmp`, `.`, or a directory without 'checkpoints'/'ckpts'):
      - Appends `checkpoints` subfolder (e.g. `/kaggle/working/checkpoints`).
    - If path already ends with `checkpoints` or `ckpts` (e.g. `/kaggle/working/checkpoints`):
      - Uses that exact directory.
    - Creates the directory if it does not exist.
    - Automatically sets `HF_HOME` and `HUGGINGFACE_HUB_CACHE` to `<checkpoint_dir>/hub`
      unless already customized by the user.
    """
    if path is None or str(path).strip() == "":
        env_val = os.environ.get("GAE_CHECKPOINT_DIR") or os.environ.get("GAE_CKPT_DIR")
        if env_val:
            p = Path(env_val).expanduser().resolve()
        else:
            existing_ckpts = _REPO_ROOT / "ckpts"
            if existing_ckpts.is_dir() and not (_REPO_ROOT / "checkpoints").is_dir():
                p = existing_ckpts.resolve()
            else:
                p = (_REPO_ROOT / "checkpoints").resolve()
    else:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = (_REPO_ROOT / p).resolve()
        else:
            p = p.resolve()

        leaf = p.name.lower()
        if leaf not in ("checkpoints", "ckpts"):
            p = p / "checkpoints"

    p.mkdir(parents=True, exist_ok=True)

    if "HF_HOME" not in os.environ and "HUGGINGFACE_HUB_CACHE" not in os.environ:
        hf_cache = p / "hub"
        hf_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_cache)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache)

    return p


def resolve_model_file(
    filename_or_path: str | Path | None,
    checkpoint_dir: Path | str | None = None,
) -> Path | None:
    """Resolve a model or stats file path with flattening support.

    Lookup precedence:
    1. Direct file path if it exists as-is.
    2. Flattened filename inside `checkpoint_dir / Path(name).name`.
    3. Nested path inside `checkpoint_dir / filename_or_path`.
    4. Relative to `_REPO_ROOT / filename_or_path`.
    5. Fallback to `checkpoint_dir / Path(name).name` even if not yet created.
    """
    if not filename_or_path:
        return None
    raw = Path(filename_or_path)
    if raw.is_file():
        return raw

    ckpt_dir = resolve_checkpoint_dir(checkpoint_dir)
    flat = ckpt_dir / raw.name
    if flat.is_file():
        return flat

    nested = ckpt_dir / raw
    if nested.is_file():
        return nested

    repo_rel = _REPO_ROOT / raw
    if repo_rel.is_file():
        return repo_rel

    return flat


# filename -> (subset, size|"shared", description)
MANIFEST: Dict[str, tuple] = {
    "gae_64.pt": ("codec", "64", "GAE-64 codec."),
    "gae_128.pt": ("codec", "128", "GAE-128 codec."),
    "latent_stats_gae_64.pt": ("codec", "64", "Per-channel latent mean/std for GAE-64."),
    "latent_stats_gae_128.pt": ("codec", "128", "Per-channel latent mean/std for GAE-128."),
    "da3_stats_giant_5ds.tar": (
        "codec",
        "shared",
        "DA3-GIANT feature-level stats. Extract to model_stats/da3_giant_5ds/.",
    ),
    "flow_gae64.pt": ("flow", "64", "GAE-64 flow model."),
    "flow_gae128.pt": ("flow", "128", "GAE-128 flow model."),
}


def files_for(size: str = "64", subset: str = "all") -> List[str]:
    out: List[str] = []
    for name, (sub, sz, _desc) in MANIFEST.items():
        if subset not in ("all", sub):
            continue
        if size == "all" or sz == "shared" or sz == str(size):
            out.append(name)
    return out


def download_weights(
    repo_id: Optional[str] = None,
    *,
    size: str = "64",
    subset: str = "all",
    out_dir: str | Path | None = None,
    filenames: Optional[Iterable[str]] = None,
) -> Dict[str, Path]:
    """Fetch Hub files into ``out_dir``. Skips files that already exist."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required (pip install huggingface-hub)") from exc

    repo = repo_id or DEFAULT_REPO
    dest = resolve_checkpoint_dir(out_dir)
    names = list(filenames) if filenames is not None else files_for(str(size), subset)
    paths: Dict[str, Path] = {}
    for name in names:
        local = dest / name
        if local.is_file() and local.stat().st_size > 0:
            paths[name] = local
            continue
        paths[name] = Path(
            hf_hub_download(repo_id=repo, filename=name, local_dir=str(dest))
        )
    return paths


def extract_da3_stats(tar_path: str | Path, dest_dir: str | Path | None = None) -> Path:
    if dest_dir is None:
        dest_dir = _REPO_ROOT / "model_stats" / "da3_giant_5ds"
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / "normalization_stats_level0.pt"
    if marker.is_file():
        return dest
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest)
    return dest

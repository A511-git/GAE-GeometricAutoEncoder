"""GAE — Learning a Geometry-Native Latent Space for 3D-Consistent World Generation.

A compact geometry representation autoencoder and flow-matching model. GAE
compresses frozen geometry-foundation features into a small latent whose
generated states decode jointly into RGB and geometry.

Two stages (paper Sections 3.1 and 3.2-3.3):

* **Stage 1** trains the codec that turns the four-level frozen DA3 feature
  hierarchy into one compact per-view latent and rebuilds every level, so the
  frozen DA3 head still reads depth, rays and point maps out of it.
* **Stage 2** freezes that codec and trains a conditional flow model over the
  standardized latent, conditioned on clean reference evidence, metric Plucker
  ray maps, and text.

Quick start::

    from gae import GAE

    gae_model = GAE.from_pretrained("TencentARC/GAE-D64-1B")
    # or, with local files:
    # gae_model = GAE.from_configs(
    #     codec_cfg="configs/gae_64.yaml",
    #     codec_ckpt="ckpts/gae_64.pt",
    # )
    z = gae_model.encode(images)          # [B, V, C, h, w]
    out = gae_model.reconstruct(images)   # {'rgb': ..., 'depth': ...}
    gae_model.save_outputs(out, "results/api", stem="recon")  # png + depth + ply

The underlying modules stay importable directly for training and research:

* ``src.stage1``  — codec, frozen DA3 backbone, representation losses
* ``src.stage2``  — flow model, transport, sampler, camera / text conditioners
* ``src.metrics`` — latent and image diagnostics
"""

from .hub import DEFAULT_REPO, download_weights, resolve_checkpoint_dir, resolve_model_file

__version__ = "0.1.0"
__all__ = [
    "GAE",
    "DEFAULT_REPO",
    "download_weights",
    "resolve_checkpoint_dir",
    "resolve_model_file",
    "load_codec",
    "load_backbone",
    "load_flow",
    "__version__",
]


def __getattr__(name: str):
    if name in ("GAE", "load_backbone", "load_codec", "load_flow"):
        from .pipeline import GAE, load_backbone, load_codec, load_flow

        globals()["GAE"] = GAE
        globals()["load_backbone"] = load_backbone
        globals()["load_codec"] = load_codec
        globals()["load_flow"] = load_flow
        return globals()[name]
    raise AttributeError(f"module 'gae' has no attribute {name!r}")

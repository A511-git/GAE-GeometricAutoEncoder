# GAE Resolution & Aspect Ratio Reference Notes

## 1. Training Dataset Resolutions

### Stage 1: Codec (VAE) Training
The Stage-1 Codec was trained on a **7-resolution mixture** across datasets (RealEstate10K, DL3DV, ScanNet++, MVSSynth, OpenVid, VidGen, OSP):

| Aspect Ratio | Resolution ($W \times H$) | Format |
| :--- | :--- | :--- |
| **16:9 (Primary Max)** | **`896 × 504`** | **Max Landscape** |
| **9:16 (Portrait)** | **`378 × 672`** | **Portrait** |
| 16:9 | `672 × 378` | Mid-Wide Landscape |
| 1:1 | `504 × 504` | Square |
| 7:4 | `588 × 336` | Landscape |
| 1:1 | `448 × 448` | Square |
| 1:1 | `336 × 336` | Small Square |

* **Max Training Resolution**: **`896 × 504`** (Landscape) / **`504 × 896`** (Portrait equivalent).

---

### Stage 2: Flow Matching (Generative Model)
* **Trained Resolution**: **`672 × 378`** (at 81 views).
* Stage-2 flow generation uses `672 × 378` as the canonical resolution to manage the GPU memory footprint of 81 temporal frames and 3D Plücker ray attention.

---

## 2. Architectural Constraints for Custom Resolutions

1. **Divisibility by 14**:
   The backbone (DA3-GIANT / ViT) uses **$14 \times 14$ patches**. Both Width and Height **must be divisible by 14**.
2. **Positional Encoding Flexibility**:
   The codec uses **2D axial RoPE (Rotary Position Embeddings)**, which allows extrapolation to resolutions outside the training set. Staying near the trained token budget ($W \times H \le 896 \times 504$) ensures the best geometric fidelity and avoids VRAM out-of-memory errors.

---

## 3. Recommended Portrait Resolutions for Inference

When running `scripts/demo/reconstruct_vae.py` on portrait images (e.g. 9:16 aspect ratio), pass `--resolution H W`:

* **Training-Matched 9:16 Portrait:**
  ```bash
  --resolution 672 378
  ```
  *(Height: 672, Width: 378)*

* **Higher Resolution 9:16 Portrait (Divisible by 14):**
  ```bash
  --resolution 896 504
  ```
  *(Height: 896, Width: 504)*

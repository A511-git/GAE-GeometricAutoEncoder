# The Plain-English Master Guide to GAE (Geometric AutoEncoder)

> **What is GAE in one sentence?**  
> GAE is an AI system that generates realistic camera-controlled videos where every single pixel has real-world 3D depth, meaning you can pause the video and fly around the scene as a true 3D point cloud.

---

## Table of Contents
1. [The Big Problem GAE Solves](#1-the-big-problem-gae-solves)
2. [The 3-Step Bird’s Eye View](#2-the-3-step-birds-eye-view)
3. [Stage 1: The Senses & The Codec (Compression & Decoders)](#3-stage-1-the-senses--the-codec)
   - [The Eyes: Depth Anything 3 (DA3) & The 4 Levels](#a-the-eyes-da3-and-the-4-levels)
   - [The Compressor: GAE Encoder](#b-the-compressor-gae-encoder)
   - [The Brain State: The 64-Number Latent](#c-the-brain-state-the-64-number-latent)
   - [The 2D Painter: RGB Head](#d-the-2d-painter-rgbhead)
   - [The 3D Sculptor: DPT Geometry Head](#e-the-3d-sculptor-dpt-head)
   - [The 3D Polisher: Temporal Geometry Adapter](#f-the-3d-polisher-temporal-geometry-adapter)
   - [The Teachers: C-RADIO, DINOv2, & LPIPS](#g-the-teachers-training-losses)
4. [Stage 2: The Imagination Engine (Flow Matching DiT)](#4-stage-2-the-imagination-engine)
   - [The Director: Qwen3 Text Encoder](#a-the-director-qwen3-text-encoder)
   - [The Camera Rig: Plücker Ray Controller](#b-the-camera-rig-plücker-rays)
   - [The Anchor: Clean Reference Latent](#c-the-anchor-clean-reference-views)
   - [The Generator: GAEFlowTemporal Transformer](#d-the-generator-gaeflowtemporal)
5. [How to Directly Convert Latents into Images and 3D Point Clouds](#5-direct-conversion-to-images-and-3d)
6. [Every Term Translated to Everyday English (Glossary)](#6-plain-english-glossary)

---

## 1. The Big Problem GAE Solves

Normal video AI (like Sora, Runway, or Pika) only thinks in **flat 2D pictures**. 
- When the camera pans around a table, a normal video AI often "hallucinates": table legs bend, background doors disappear, and objects morph. 
- It has no concept of physical space—it's just drawing pretty pictures that look plausible frame-by-frame.

**GAE's Solution:**
Instead of teaching the AI to draw flat 2D frames, GAE forces the AI's internal thoughts to be **3D geometric blueprints**.
When GAE dreams up a video, it is actually generating **both the visual colors AND the exact physical 3D coordinates** of every surface in the room.

---

## 2. The 3-Step Bird’s Eye View

```
[Real Camera Footage]
         │
         ▼
[STAGE 1: Codec (Compression)]
1. The Eyes (DA3) take 6,144 detailed notes across 4 zoom levels.
2. The GAE Encoder squashes those 6,144 notes into a 64-number "ZIP file" (The Latent).
         │
         ▼
[STAGE 2: Imagination (Generation)]
3. From a text prompt + a camera flight path, GAEFlow invents a sequence of 64-number thoughts.
         │
         ▼
[THE READOUTS]
├───> GAE.decode_rgb()      ──> Paints the photorealistic 2D Video frames (.mp4)
└───> GAE.decode_geometry() ──> Calculates real meters and drops a 3D Point Cloud (.ply)
```

---

## 3. Stage 1: The Senses & The Codec

### A. The Eyes: DA3 and the 4 Levels
* **Code file:** [`src/stage1/da3.py`](file:///C:/Users/A-511/OneDrive/Desktop/projects/GAE-GeometricAutoEncoder/src/stage1/da3.py) and [`src/stage1/encoders/da3.py`](file:///C:/Users/A-511/OneDrive/Desktop/projects/GAE-GeometricAutoEncoder/src/stage1/encoders/da3.py)
* **Status:** Completely Frozen (trained once by researchers, never modified).
* **What is it:** Pretrained **Depth Anything 3 (DA3-GIANT)** built on a 24-layer DINOv2 vision network.

#### What are the 4 Levels?
Imagine an assembly line of 24 workers analyzing an image:
1. **Level 0 (Microscopic Outlines - Worker 6):** Looks at tiny pixel boundaries. It detects razor-sharp dark-to-light edges, fine wood grain, wallpaper stripes, and hair strands. *(1,536 numbers)*
2. **Level 1 (Shapes & Parts - Worker 12):** Connects the lines into corners, rounded edges, chair legs, and window frames. *(1,536 numbers)*
3. **Level 2 (Whole Objects - Worker 18):** Identifies full items: a chair, a table, a lamp. *(1,536 numbers)*
4. **Level 3 (3D Room Layout - Worker 24):** Looks at the big picture: how far the floor extends, the orientation of the walls, and the distance from the camera in meters. *(1,536 numbers)*

**Total: $4 \times 1,536 = 6,144$ numbers per patch.**
> **Why do we need all 4?**  
> If you only keep Level 3, objects have 3D positions but their edges become blurry blobs. If you only keep Level 0, you have sharp lines but zero 3D depth. Keeping all 4 gives razor-sharp lines AND true 3D depth.

---

### B. The Compressor: GAE Encoder
* **Code file:** [`src/stage1/gae_codec.py`](file:///C:/Users/A-511/OneDrive/Desktop/projects/GAE-GeometricAutoEncoder/src/stage1/gae_codec.py#L718-L760)
* **Status:** Trained.
* **What it does:** 
  6,144 numbers per patch is too massive for a computer to generate quickly.
  The GAE Encoder takes those 6,144 numbers and funnels them through convolutional layers and self-attention blocks down to **just 64 numbers** (or 128 in the large version).

---

### C. The Brain State: The 64-Number Latent
* **What it is:** The internal mental model of the scene.
* Think of it as an ultra-smart **ZIP file**. 
* Even though 99% of the bulk was thrown away, it preserves both:
  1. The **appearance** (what colors and textures belong where).
  2. The **geometry** (how far away each point is in 3D space).

---

### D. The 2D Painter: RGB Head (`RGBHead`)
* **Code file:** [`src/stage1/gae_codec.py`](file:///C:/Users/A-511/OneDrive/Desktop/projects/GAE-GeometricAutoEncoder/src/stage1/gae_codec.py#L257-L491)
* **Status:** Trained.
* **Architecture:** 12-layer Vision Transformer with **Divided Space-Time attention**.
* **What it does:** 
  - It reads the 64 numbers in the latent.
  - Across time, it compares neighboring video frames to make sure colors don't flicker.
  - It paints the final red, green, and blue pixels onto your screen.

---

### E. The 3D Sculptor: DPT Geometry Head (`H_DPT`)
* **Code file:** [`src/stage1/da3.py`](file:///C:/Users/A-511/OneDrive/Desktop/projects/GAE-GeometricAutoEncoder/src/stage1/da3.py#L117-L131)
* **Status:** Frozen.
* **What it does:**
  1. The GAE Feature Decoder (`dec_conv`) unzips the 64 numbers back up to the original 6,144 notes.
  2. The frozen DA3 DPT head reads those notes.
  3. It outputs:
     - **Metric Depth:** Exactly how many meters away each pixel is.
     - **3D Point Coordinates $(X, Y, Z)$:** The exact physical location of every surface point.
     - Saves directly into a **`.ply` 3D point cloud file**.

---

### F. The 3D Polisher: Temporal Geometry Adapter (`TemporalGeoAdapter`)
* **Code file:** [`src/stage1/gae_codec.py`](file:///C:/Users/A-511/OneDrive/Desktop/projects/GAE-GeometricAutoEncoder/src/stage1/gae_codec.py#L542-L594)
* **Status:** Fine-tuned post-hoc via [`scripts/train/finetune_geo_decoder.py`](file:///C:/Users/A-511/OneDrive/Desktop/projects/GAE-GeometricAutoEncoder/scripts/train/finetune_geo_decoder.py).
* **What it does:** 
  When watching a video, tiny imperfections in 3D depth can make surfaces look like they're slightly "jittering" in 3D.
  This adapter sits exclusively on the 3D geometry branch and locks multi-frame 3D points together so walls and objects stay rock-solid, **without touching or altering the RGB colors**.

---

### G. The Teachers: Training Losses
During Stage 1 training, the model's brain is kept organized by three external "teachers":
1. **C-RADIO Teacher (`CRadioTarget`):** A frozen vision foundation model from NVIDIA that ensures nearby objects have smooth, understandable semantics ($L_{\text{tok}}$).
2. **DINOv2 Teacher (`DINOv2Target`):** A frozen Meta AI model that compares patch-to-patch relationships, preventing the model from collapsing or scrambling object layouts ($L_{\text{struct}}$).
3. **LPIPS Perceptual Loss (`disc.lpips.LPIPS`):** Measures whether the generated images look realistic to human eyes rather than just matching raw mathematical pixel values.

---

## 4. Stage 2: The Imagination Engine

Once the Codec knows how to compress and decompress 3D worlds, **Stage 1 is locked and frozen**.
Now comes Stage 2: **teaching the AI to dream up new 3D worlds from scratch**.

```
[Prompt: "Modern living room, sunny"] ──> Qwen3 Text Encoder ──┐
[Camera Path: Drone flying forward]    ──> Plücker Embedder     ──┼──> [GAEFlow Transformer] ──> New 3D Latent
[First Frame / Reference View]         ──> Clean Token Anchor  ──┘
```

### A. The Director: Qwen3 Text Encoder
* **Code file:** [`src/stage2/models/text_encoder.py`](file:///C:/Users/A-511/OneDrive/Desktop/projects/GAE-GeometricAutoEncoder/src/stage2/models/text_encoder.py)
* **Model:** `Qwen/Qwen3-0.6B` (Frozen).
* **What it does:** Reads your English prompt (e.g. *"A drone flying over a rocky coastline at sunset"*) and translates it into mathematical concept vectors that guide the generator.

---

### B. The Camera Rig: Plücker Ray Controller
* **Code file:** [`src/stage2/models/camera.py`](file:///C:/Users/A-511/OneDrive/Desktop/projects/GAE-GeometricAutoEncoder/src/stage2/models/camera.py)
* **Module:** `PluckerFlipPE_V13`.
* **What it does:** 
  Instead of saying "turn left", it calculates the exact light rays coming out of every single camera lens pixel in 3D space:
  - Ray direction ($d$)
  - Moment / rotation anchor ($\hat{m}$)
  - Metric scale in meters ($s = \log \|m\|$)
  This allows you to steer the camera with millimeter accuracy.

---

### C. The Anchor: Clean Reference Views (`z_ref_clean`)
* **Code file:** [`src/stage2/models/token_concat_ddt.py`](file:///C:/Users/A-511/OneDrive/Desktop/projects/GAE-GeometricAutoEncoder/src/stage2/models/token_concat_ddt.py)
* **What it does:** 
  If you provide an existing photo and say *"make a video continuing from this photo"*, the model encodes that photo and pins it at the beginning ($t=0$). 
  The generator looks at this photo for guidance, but never alters it—it only invents the new future frames.

---

### D. The Generator: `GAEFlowTemporal`
* **Code file:** [`src/stage2/models/dit_temporal.py`](file:///C:/Users/A-511/OneDrive/Desktop/projects/GAE-GeometricAutoEncoder/src/stage2/models/dit_temporal.py)
* **Architecture:** 34-layer Spatiotemporal Dual Diffusion Transformer (DDT) with 16 attention heads.
* **How it creates worlds (Flow Matching):**
  - Starts with pure random static (white noise).
  - Over 25–50 steps, it pushes that noise along a straight mathematical velocity line until it becomes a clean, structured 64-number 3D latent state.
  - Because it uses **temporal RoPE**, it tracks objects continuously across time without temporal jumping or warping.

---

## 5. Direct Conversion to Images and 3D

You never need to perform manual math to extract results. The repository includes prebuilt pipelines:

### 1. In Python Code:
```python
from gae import GAE

# 1. Load the model
model = GAE.from_pretrained(size=64).cuda()

# 2. Encode any image into the 64-number brain state
z = model.encode(image_tensor)

# 3. DIRECTLY get the RGB image/video:
rgb_video = model.decode_rgb(z)

# 4. DIRECTLY get the 3D depth and Point Cloud:
geom_data = model.decode_geometry(z, height=504, width=672)
# geom_data contains:
# - 'depth': Metric distances in meters
# - 'points': Exact 3D (X, Y, Z) coordinates of every pixel
```

### 2. From the Command Line:
* **Generate Video + 3D Pointcloud from an image:**
  ```bash
  bash scripts/demo/run_demo.sh --task i2v
  ```
  Outputs an `.mp4` video and a `.ply` 3D point cloud file in `results/demo/i2v/`.
* **Export Interactive 3D Point Clouds:**
  ```bash
  python scripts/demo/render_progressive_ply.py --config configs/gae_64.yaml --out scene.ply
  ```

---

## 6. Plain-English Glossary

| Term | What it actually means |
| :--- | :--- |
| **Latent ($z$)** | The compressed "thought" inside the model's brain (64 numbers per grid cell). |
| **Codec** | The translator pair: Encoder squashes images into thoughts; Decoder expands thoughts back into images & 3D. |
| **DINOv2** | Meta's computer vision model that acts as the model's "eyes" to recognize objects and outlines. |
| **DA3 (Depth Anything 3)** | The frozen foundation model that estimates depth and 3D camera rays. |
| **DPT Head** | The specific readout module that turns neural features into metric depth (meters). |
| **Plücker Rays** | A mathematical way to describe a camera ray in 3D so the AI knows where you are looking. |
| **Flow Matching / DiT** | The generation technique that gradually turns random static into a crisp 3D video. |
| **`.ply` file** | A standard 3D file format (Polygon/Point Cloud) that you can open in Blender to view the scene in 3D. |
| **Divided Space-Time** | A transformer design that looks across space (within a frame) and across time (between frames) separately. |

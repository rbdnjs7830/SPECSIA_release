---
license: cc-by-nc-4.0
task_categories:
  - image-to-image
language:
  - en
tags:
  - drawing
  - animation
  - 3d-reconstruction
  - stylization
  - novel-view-synthesis
  - artifact-correction
size_categories:
  - 10K<n<100K
---

# SPECSIA-15K

**SPECSIA: Stylization Dataset for Novel-View Enhancement in Drawing-based 3D Animation**  
Kyuwon Kim, Sunjae Yoon, Chang D. Yoo (KAIST / Chung-Ang University)  
*European Conference on Computer Vision (ECCV) 2026*

[[Project Page]](https://rbdnjs7830.github.io/SPECSIA/) · [[Paper]](#) · [[Code]](https://github.com/rbdnjs7830/SPECSIA_release)

---

## Dataset Description

SPECSIA-15K is a large-scale paired dataset for learning to correct projection artifacts in
drawing-based 3D character animation.

Drawing-based 3D pipelines (e.g. [DrawingSpinUp](https://github.com/LordLiang/DrawingSpinUp))
lift a single 2D cartoon into an animatable 3D character. When the 3D mesh is re-projected
to 2D from unseen viewpoints, the rendered image carries visible artifacts — texture
degradation, missing contours, mesh noise. SPECSIA-15K provides matched
(artifact-prone input, artifact-free GT) pairs so a refinement network can be
pre-trained to correct these artifacts in a view-generalizable way.

### How the dataset was built

1. **Source**: 1,498 characters from [3DBiCar](https://gaplab.cuhk.edu.cn/projects/RaBit/dataset.html) (bicartoon 3D characters).
2. **GT renders (Y\*)**: Each character's T-pose OBJ is rendered from 10 uniformly-spaced
   yaw angles (0°–360°) using Blender Cycles with an orthographic camera and Freestyle
   outline (external contour only; thickness uniformly sampled from {1, 2, 3, 4} pixels,
   grayscale line color uniformly sampled from [0, 255]).  
   → `{char}_{view}/char/texture_with_bg.png`
3. **Reconstructed input (Z)**: The front-view render is passed through
   [Wonder3D](https://github.com/xxlong0/Wonder3D) → [instant-nsr](https://github.com/zxhuang1698/instant-nsr-pl)
   to obtain a coarse 3D mesh. This mesh is re-rendered from the same 10 yaw angles
   using Blender Eevee with orthographic projection.  
   → `{char}_{view}/mesh/blender_render/rest_pose/color/0001.png`
4. **Positional hint (Z_pos)**: Normalized world-space XYZ vertex colors rendered from
   the same viewpoint. Encodes spatial location for location-aware correction.  
   → `{char}_{view}/mesh/blender_render/rest_pose/pos/0001.png`
5. **Edge map**: Sobel gradient on Z_pos, highlighting geometry boundaries.  
   → `{char}_{view}/mesh/blender_render/rest_pose/edge/0001.png`

---

## Dataset Statistics

| Split      | Characters | Samples |
|------------|-----------|---------|
| train      | 1,298     | ~12,980 |
| validation | 100       | ~1,000  |
| test       | 100       | ~1,000  |
| **total**  | **1,498** | **~14,980** |

- Resolution: **512 × 512** pixels
- Format: **RGB PNG** (color, pos, edge, gt), **grayscale PNG** (mask)
- Random seed for split: **42** (character-level, no view leakage across splits)

---

## Fields

| Field   | Type    | Description |
|---------|---------|-------------|
| `uid`   | string  | `"{char_id}_{view_id}"` — unique sample identifier |
| `color` | Image   | Reconstructed 3D re-projection — artifact-prone input **Z** |
| `pos`   | Image   | Positional hint (world-space XYZ → RGB) — **Z_pos** |
| `edge`  | Image   | Sobel edge map derived from `pos` |
| `mask`  | Image   | Foreground silhouette mask (grayscale) |
| `gt`    | Image   | GT Blender Cycles render — artifact-free target **Y\*** |

---

## Usage

```python
from datasets import load_dataset

ds = load_dataset("Kyu0528/SPECSIA-15K")

sample = ds["train"][0]
sample["color"].show()  # artifact-prone reconstructed render (Z)
sample["gt"].show()     # artifact-free GT render (Y*)
```

### Pre-training DraViE on SPECSIA-15K

```bash
git clone https://github.com/rbdnjs7830/SPECSIA_release
cd SPECSIA_release
# Edit configs/pretrain_3dbicar.yaml: set root_dir to your dataset path
python train_pretrain.py --config configs/pretrain_3dbicar.yaml
```

---

## License

SPECSIA-15K is released under **CC BY-NC 4.0**.

This dataset contains only rendered image pairs derived from 3DBiCar character models.
**Raw 3DBiCar 3D assets are NOT included** and must be obtained separately from the
[3DBiCar repository](https://gaplab.cuhk.edu.cn/projects/RaBit/dataset.html) under their original license.

---

## Citation

```bibtex
TBA
```

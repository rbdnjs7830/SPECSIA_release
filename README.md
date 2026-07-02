# SPECSIA: Stylization Dataset for Novel-View Enhancement in Drawing-based 3D Animation

[Kyuwon Kim](https://rbdnjs7830.github.io/)<sup>1</sup> · [Sunjae Yoon](https://dbstjswo505.github.io/)<sup>2</sup> · [Chang D. Yoo](https://sanctusfactory.com/family.php)<sup>1</sup>

<sup>1</sup>School of Electrical Engineering, KAIST · <sup>2</sup>Department of AI, Chung-Ang University

*European Conference on Computer Vision (ECCV) 2026*

<a href='https://arxiv.org/abs/2607.00525'><img src='https://img.shields.io/badge/arXiv-2607.00525-b31b1b?logo=arxiv&logoColor=white'></a>
<a href='https://rbdnjs7830.github.io/SPECSIA/'><img src='https://img.shields.io/badge/Project-Page-green?logo=googlechrome&logoColor=white'></a>
<a href='https://huggingface.co/datasets/Kyu0528/SPECSIA-15K'><img src='https://img.shields.io/badge/HuggingFace-Dataset-yellow?logo=huggingface&logoColor=white'></a>

---

## Overview

Drawing-based 3D animation lifts a single 2D drawing into an animatable character
([DrawingSpinUp](https://github.com/LordLiang/DrawingSpinUp), SIGGRAPH Asia 2024).
A key remaining challenge is **stylization quality under novel views**: 2D projections of 3D geometry
introduce artifacts (missing contours, speckle noise, texture degradation) that compound across views,
especially under self-occlusion.

**SPECSIA-15K** is a large-scale paired dataset of 14,980 samples (1,498 3DBiCar characters × 10 viewpoints)
that enables pre-training a general projection-artifact corrector.

> **Code & Models:** DraViE model, training scripts, inference pipeline, and pretrained checkpoints — **coming soon.**

---

## Dataset

### Download SPECSIA-15K from HuggingFace

```python
from datasets import load_dataset

ds = load_dataset("Kyu0528/SPECSIA-15K")
train_ds = ds["train"]
sample = train_ds[0]
sample["color"].show()  # artifact-prone reconstructed render (Z)
sample["gt"].show()     # artifact-free GT render (Y*)
```

### Dataset Fields

Each sample is a 5-tuple `(Z, Z_mask, Z_pos, Z_edge, Y*)`:

| Field | Type | Meaning |
|---|---|---|
| `uid` | string | `{char_id}_{view_id}` |
| `color` | RGB image | Reconstructed render with projection artifacts (Z) |
| `mask` | L image | Foreground silhouette mask |
| `pos` | RGB image | Canonical world-space XYZ color map (Z_pos) |
| `edge` | RGB image | Sobel edge map derived from `pos` |
| `gt` | RGB image | Artifact-free GT Blender Cycles render (Y\*) |

### Splits

| Split | Characters | Samples |
|---|---|---|
| train | 1,298 | 12,980 |
| validation | 100 | 1,000 |
| test | 100 | 1,000 |
| **Total** | **1,498** | **14,980** |

Splits are character-disjoint (seed=42).

---

## Build SPECSIA-15K from Scratch

You can reproduce the full dataset from the raw [3DBiCar](https://gaplab.cuhk.edu.cn/projects/RaBit/dataset.html) release.

### 0. Setup

Clone this repo and run the one-time setup (downloads Blender 3.6.14, clones DrawingSpinUp, installs Python deps):

```bash
git clone https://github.com/rbdnjs7830/SPECSIA_release
cd SPECSIA_release
bash setup.sh        # creates conda env 'specsia', installs deps
conda activate specsia
```

### 1. Download 3DBiCar

Download from [https://gaplab.cuhk.edu.cn/projects/RaBit/dataset.html](https://gaplab.cuhk.edu.cn/projects/RaBit/dataset.html) and extract so each character follows:

```
{dl_extracted}/{uid}/
├── tpose/
│   ├── m.obj          ← body mesh (required)
│   ├── m.bmp          ← body texture
│   ├── e.obj          ← accessory mesh (required)
│   └── e.bmp          ← accessory texture
├── image/
│   ├── image_reshape512.jpeg  ← front-view 2D drawing
│   └── mask512.png
└── metadata.json
```

### 2. Configure paths and run the pipeline

Edit `dataset_prep/paths.sh` to set your data directories, then run:

```bash
source dataset_prep/paths.sh
bash dataset_prep/run_pipeline_3dbicar.sh --resume
```

The pipeline runs all stages automatically:
- **Stage 0** — Blender Cycles renders 10 yaw views of each T-pose OBJ (GT targets)
- **Stage 1** — Wonder3D generates 6-view priors from the front-view drawing
- **Stage 2** — instant-nsr reconstructs a 3D mesh from the 6 views
- **Stage 3** — Blender Eevee renders the reconstructed mesh (color, pos, edge) from 10 angles
- **Stage 4** — Creates character-disjoint train/val/test splits

Individual stages can be skipped with `--skip-gt-render`, `--skip-mv`, `--skip-recon`, `--skip-render`.

---

## Related Work

| Paper | Venue | Relation |
|---|---|---|
| [DrawingSpinUp](https://github.com/LordLiang/DrawingSpinUp) — Zhou et al. | SIGGRAPH Asia 2024 | Mesh reconstruction + Blender projection pipeline |
| [OSF](https://dbstjswo505.github.io/Drawing-based-3D-Animation-page/) — Yoon et al. | ICCV 2025 | Occlusion-aware stylization baseline |

---

## Citation

```bibtex
TBA
```

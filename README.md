# SPECSIA: Stylization Dataset for Novel-View Enhancement in Drawing-based 3D Animation

[Paper](#) | [Project Page](https://rbdnjs7830.github.io/SPECSIA/) | [Dataset (HuggingFace)](https://huggingface.co/datasets/Kyu0528/SPECSIA-15K)

> **Kyuwon Kim, Sunjae Yoon, Chang D. Yoo**  
> KAIST School of Electrical Engineering
>
> *European Conference on Computer Vision (ECCV) 2026*

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

If you prefer to build the dataset locally from [3DBiCar](https://gaplab.cuhk.edu.cn/projects/RaBit/dataset.html):

**1. Download 3DBiCar**

Download from [https://gaplab.cuhk.edu.cn/projects/RaBit/dataset.html](https://gaplab.cuhk.edu.cn/projects/RaBit/dataset.html)
and organize each character as:

```
{data_dir}/{uid}/
└── char/
    ├── texture.png      ← front-view character drawing
    └── mask.png         ← foreground mask
```

**2. Multi-view generation (Wonder3D)**

```bash
python dataset_prep/run_mv_parallel.py \
  --dsu_dir  /path/to/DrawingSpinUp \
  --data_dir /path/to/3dbicar/preprocessed \
  --gpus 0 1 2 3 --resume
# Output → {data_dir}/{uid}/mv/
```

**3. 3D reconstruction (instant-nsr)**

```bash
python dataset_prep/run_recon_parallel.py \
  --dsu_dir  /path/to/DrawingSpinUp \
  --data_dir /path/to/3dbicar/preprocessed \
  --gpus 0 1 2 3 --resume
# Output → {data_dir}/{uid}/mesh/*.obj
```

**4. Multi-view Blender rendering**

```bash
python dataset_prep/run_render_obj_parallel.py \
  --data_dir    /path/to/3dbicar/preprocessed \
  --output_root /path/to/SPECSIA-15k \
  --uid_list    /path/to/3dbicar/preprocessed/base_uids.json \
  --blender     ./blender-3.6.14-linux-x64/blender \
  --script      dataset_prep/blender_render_obj.py \
  --num_workers 4 --align
# Output → {output_root}/{uid}/
```

**5. Create splits and upload**

```bash
python dataset_prep/create_splits.py --specsia_root /path/to/SPECSIA-15k
python dataset_prep/upload_hf.py \
  --specsia_root /path/to/SPECSIA-15k \
  --repo_id      YOUR_HF_USERNAME/SPECSIA-15K
```

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

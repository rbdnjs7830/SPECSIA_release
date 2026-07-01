#!/usr/bin/env bash
# SPECSIA-15K — Path Configuration
#
# 1. Run `bash setup.sh` once from the repo root (downloads Blender, inits submodule).
# 2. Edit the data paths below (only DL_EXTRACTED, BICAR_RENDER, BICAR_PREPROCESSED,
#    SPECSIA_OUTPUT, and GPUS typically need changing).
# 3. Then:
#      source dataset_prep/paths.sh
#      bash dataset_prep/run_pipeline_3dbicar.sh --resume

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Raw 3DBiCar download root ─────────────────────────────────────────────────
# Structure: ${DL_EXTRACTED}/{char_id}/{tpose/m.obj, image/mask512.png, ...}
export DL_EXTRACTED="${HOME}/3dbicar/dl_extracted"

# ── Intermediate GT renders (created by Stage 0a) ────────────────────────────
export BICAR_RENDER="${HOME}/3dbicar/render"

# ── 3DBiCar preprocessed directory (created by Stage 0b) ────────────────────
# Structure: ${BICAR_PREPROCESSED}/{char_id}_{view}/char/{texture.png,mask.png}
export BICAR_PREPROCESSED="${HOME}/3dbicar/preprocessed"

# ── SPECSIA-15k output directory (created by Stage 3) ────────────────────────
export SPECSIA_OUTPUT="${HOME}/SPECSIA-15k"

# ── DrawingSpinUp repo root (set up by `bash setup.sh`) ──────────────────────
export DSU_DIR="${_REPO_ROOT}/external/DrawingSpinUp"

# ── Blender 3.6.x executable (downloaded by `bash setup.sh`) ─────────────────
export BLENDER="${_REPO_ROOT}/blender-3.6.14-linux-x64/blender"

# ── CUDA device indices ───────────────────────────────────────────────────────
# Space-separated list. Tip: CUDA indices ≠ nvidia-smi indices if GPU 0 is absent.
# Verify with: python -c "import torch; print(torch.cuda.device_count())"
export GPUS="0 1 2 3"

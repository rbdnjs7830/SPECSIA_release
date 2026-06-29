#!/usr/bin/env bash
# SPECSIA-15K — Path Configuration
# Fill in the paths below, then run:
#   source dataset_prep/paths.sh
#   bash dataset_prep/run_pipeline_3dbicar.sh --resume

# ── Raw 3DBiCar download root ─────────────────────────────────────────────────
# Structure: ${DL_EXTRACTED}/{char_id}/{tpose/m.obj, image/mask512.png, ...}
export DL_EXTRACTED="${HOME}/3dbicar/dl_extracted"

# ── Intermediate GT renders (created by Stage 0a) ────────────────────────────
# Will contain: ${BICAR_RENDER}/{char_id}/view_XX_rgba_outline.png
export BICAR_RENDER="${HOME}/3dbicar/render"

# ── 3DBiCar preprocessed directory (created by Stage 0b) ────────────────────
# Structure: ${BICAR_PREPROCESSED}/{char_id}_{view}/char/{texture.png,mask.png}
export BICAR_PREPROCESSED="${HOME}/3dbicar/preprocessed"

# ── SPECSIA-15k output directory (created by Stage 3) ────────────────────────
export SPECSIA_OUTPUT="${HOME}/SPECSIA-15k"

# ── DrawingSpinUp repo root ───────────────────────────────────────────────────
# Clone from: https://github.com/LordLiang/DrawingSpinUp
export DSU_DIR="${HOME}/DrawingSpinUp"

# ── Blender 3.6.x executable ─────────────────────────────────────────────────
# Download from: https://www.blender.org/download/release/Blender3.6/
export BLENDER="${HOME}/blender-3.6.14-linux-x64/blender"

# ── CUDA device indices ───────────────────────────────────────────────────────
# Space-separated CUDA_VISIBLE_DEVICES values for mv + recon stages.
# Tip: CUDA indices ≠ nvidia-smi indices if GPU 0 is broken.
# Verify count with: python -c "import torch; print(torch.cuda.device_count())"
export GPUS="0 1 2 3"

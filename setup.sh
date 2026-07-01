#!/usr/bin/env bash
# One-time environment setup for SPECSIA_release.
# Run from the repository root: bash setup.sh
set -e

ENV_NAME="specsia"
PYTHON_VERSION="3.10"

BLENDER_VERSION="3.6.14"
BLENDER_TARBALL="blender-${BLENDER_VERSION}-linux-x64.tar.xz"
BLENDER_URL="https://download.blender.org/release/Blender3.6/${BLENDER_TARBALL}"
BLENDER_DIR="blender-${BLENDER_VERSION}-linux-x64"
BLENDER_BIN="./${BLENDER_DIR}/blender"
BLENDER_PY="./${BLENDER_DIR}/3.6/python/bin/python3.10"

# ── 0. Conda environment ──────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found. Install Miniconda or Anaconda first."
  exit 1
fi

if conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo "Conda env '${ENV_NAME}' already exists, skipping creation."
else
  echo "Creating conda env '${ENV_NAME}' (Python ${PYTHON_VERSION})…"
  conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

# ── 1. Blender ────────────────────────────────────────────────────────────────
if [ ! -d "${BLENDER_DIR}" ]; then
  echo "Downloading Blender ${BLENDER_VERSION}…"
  wget -q --show-progress "${BLENDER_URL}"
  tar -xf "${BLENDER_TARBALL}"
  rm "${BLENDER_TARBALL}"
else
  echo "Blender already present, skipping."
fi

# Install trimesh into Blender's bundled Python (needed by blender_render_obj.py)
if [ -f "${BLENDER_BIN}" ] && [ -f "${BLENDER_PY}" ]; then
  if ! "${BLENDER_PY}" -c "import trimesh" 2>/dev/null; then
    echo "Installing trimesh into Blender's Python…"
    wget -q -O /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
    "${BLENDER_PY}" /tmp/get-pip.py --quiet || true
    "${BLENDER_PY}" -m pip install trimesh -q
    rm -f /tmp/get-pip.py
  else
    echo "trimesh already installed in Blender Python, skipping."
  fi
fi

# ── 2. DrawingSpinUp ──────────────────────────────────────────────────────────
# Provides: Wonder3D (mv.py), instant-nsr (recon.py), and config_ortho.blend
if [ ! -d "external/DrawingSpinUp" ]; then
  echo "Cloning DrawingSpinUp…"
  mkdir -p external
  git clone https://github.com/LordLiang/DrawingSpinUp external/DrawingSpinUp
else
  echo "DrawingSpinUp already present, skipping."
fi

# ── 3. DrawingSpinUp Python dependencies ──────────────────────────────────────
# Wonder3D and instant-nsr require heavy ML packages (diffusers, xformers, etc.)
echo "Installing DrawingSpinUp dependencies into '${ENV_NAME}'…"
conda run -n "${ENV_NAME}" pip install -r external/DrawingSpinUp/requirements.txt

# ── 4. SPECSIA / DraViE Python dependencies ───────────────────────────────────
echo "Installing SPECSIA dependencies into '${ENV_NAME}'…"
conda run -n "${ENV_NAME}" pip install -r requirements.txt

# ── 5. Pretrained models ──────────────────────────────────────────────────────
# [Required] DIS background-removal model (used by Wonder3D)
DIS_DIR="external/DrawingSpinUp/2_charactor_reconstructor/dis_pretrained"
DIS_MODEL="${DIS_DIR}/isnet_dis.onnx"
if [ ! -f "${DIS_MODEL}" ]; then
  echo "Downloading DIS model for background removal…"
  mkdir -p "${DIS_DIR}"
  wget -q --show-progress \
    "https://huggingface.co/stoned0651/isnet_dis.onnx/resolve/main/isnet_dis.onnx" \
    -O "${DIS_MODEL}"
else
  echo "DIS model already present, skipping."
fi
#
# [Optional] LaMa contour-removal model (improves Wonder3D quality)
#   Download experiments.zip from https://github.com/LordLiang/DrawingSpinUp
#   and extract into: external/DrawingSpinUp/1_lama_contour_remover/experiments/
#
# [Auto-downloaded] Wonder3D weights (flamehaze1115/wonder3d-v1.0) are fetched
#   from HuggingFace automatically on the first run of Stage 1.

echo ""
echo "Setup complete."
echo "  Conda env : ${ENV_NAME}  (Python ${PYTHON_VERSION})"
echo "  Blender   : ${BLENDER_BIN}"
echo "  DSU       : external/DrawingSpinUp"
echo ""
echo "Next steps:"
echo "  1. conda activate ${ENV_NAME}"
echo "  2. Edit dataset_prep/paths.sh (set DL_EXTRACTED, BICAR_RENDER, etc.)"
echo "  3. source dataset_prep/paths.sh"
echo "  4. bash dataset_prep/run_pipeline_3dbicar.sh --resume"

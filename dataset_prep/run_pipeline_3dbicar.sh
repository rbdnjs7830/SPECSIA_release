#!/usr/bin/env bash
# SPECSIA-15K — Full Dataset Construction Pipeline for 3DBiCar
#
# Prerequisites
# -------------
#   1. Fill in dataset_prep/paths.sh and run: source dataset_prep/paths.sh
#   2. conda activate specsia
#   3. DrawingSpinUp repo cloned at $DSU_DIR with its dependencies installed
#      (see https://github.com/LordLiang/DrawingSpinUp for setup)
#
# Pipeline stages
# ---------------
#   0a. GT render   — Blender (Cycles) renders 10 yaw views of each character
#                     from the raw 3DBiCar T-pose OBJ.
#                     Input:  $DL_EXTRACTED/{char_id}/tpose/{m,e}.obj
#                     Output: $BICAR_RENDER/{char_id}/view_XX_rgba_outline.png
#
#   0b. GT pack     — Converts rendered PNGs to AnimatedDrawings preprocessed format.
#                     Input:  $BICAR_RENDER/
#                     Output: $BICAR_PREPROCESSED/{char_id}_{view}/char/{texture,mask}.png
#
#   1.  Wonder3D    — Predicts 6 multi-view images from the front-view image.
#                     Input:  $BICAR_PREPROCESSED/{id}_00/char/texture.png
#                     Output: $BICAR_PREPROCESSED/{id}_00/mv/{color,normal,mask}/
#
#   2.  instant-nsr — Reconstructs a 3D mesh from the 6 predicted views.
#                     Input:  $BICAR_PREPROCESSED/{id}_00/mv/
#                     Output: $BICAR_PREPROCESSED/{id}_00/mesh/*.obj
#
#   3.  10-view render — Renders the reconstructed OBJ from 10 yaw angles with
#                        color, positional hint, and edge maps.
#                     Input:  $BICAR_PREPROCESSED/{id}_00/mesh/*.obj
#                     Output: $SPECSIA_OUTPUT/{id}_{view}/mesh/blender_render/rest_pose/
#                             $SPECSIA_OUTPUT/{id}_{view}/char → (symlink to GT)
#
# Usage
# -----
#   source dataset_prep/paths.sh
#   bash dataset_prep/run_pipeline_3dbicar.sh [options]
#
# Options:
#   --skip-gt-render   Skip stages 0a + 0b (GT Blender render + pack)
#   --skip-mv          Skip Wonder3D stage
#   --skip-recon       Skip instant-nsr stage
#   --skip-render      Skip 10-view reconstructed render stage
#   --resume           Skip already-completed items in each stage

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load path config if present
if [ -f "${SCRIPT_DIR}/paths.sh" ]; then
    # shellcheck source=dataset_prep/paths.sh
    source "${SCRIPT_DIR}/paths.sh"
fi

# Validate required paths
: "${DL_EXTRACTED:?'Set DL_EXTRACTED or source dataset_prep/paths.sh'}"
: "${BICAR_RENDER:?'Set BICAR_RENDER or source dataset_prep/paths.sh'}"
: "${BICAR_PREPROCESSED:?'Set BICAR_PREPROCESSED or source dataset_prep/paths.sh'}"
: "${SPECSIA_OUTPUT:?'Set SPECSIA_OUTPUT or source dataset_prep/paths.sh'}"
: "${DSU_DIR:?'Set DSU_DIR or source dataset_prep/paths.sh'}"
: "${BLENDER:?'Set BLENDER or source dataset_prep/paths.sh'}"
GPUS="${GPUS:-0 1 2 3}"

CONFIG_BLEND="${DSU_DIR}/3_style_translator/configs/blender/config_ortho.blend"
BICAR_UIDS="${DL_EXTRACTED}/bicar_uids.json"
UID_LIST="${BICAR_PREPROCESSED}/base_uids_00.json"
BASE_UID_LIST="${BICAR_PREPROCESSED}/base_uids.json"
RENDER_SCRIPT="${SCRIPT_DIR}/blender_render_obj.py"

SKIP_GT=0; SKIP_MV=0; SKIP_RECON=0; SKIP_RENDER=0; RESUME=""
for arg in "$@"; do
    case $arg in
        --skip-gt-render) SKIP_GT=1    ;;
        --skip-mv)        SKIP_MV=1    ;;
        --skip-recon)     SKIP_RECON=1 ;;
        --skip-render)    SKIP_RENDER=1;;
        --resume)         RESUME="--resume" ;;
    esac
done

echo "========================================================"
echo " SPECSIA-15K Pipeline (3DBiCar)"
echo " Raw data:   $DL_EXTRACTED"
echo " GT render:  $BICAR_RENDER"
echo " Preproc:    $BICAR_PREPROCESSED"
echo " Output:     $SPECSIA_OUTPUT"
echo " DSU:        $DSU_DIR"
echo " GPUs:       $GPUS"
echo "========================================================"

# ── Stage 0: GT render ─────────────────────────────────────────────────────
if [ "$SKIP_GT" -eq 0 ]; then

    # 0a. Scan dl_extracted → bicar_uids.json
    if [ ! -f "$BICAR_UIDS" ]; then
        echo ""
        echo "[0a/3] Scanning raw download for character IDs ..."
        python "${SCRIPT_DIR}/prep_3dbicar_uids.py" \
            --dl_extracted "$DL_EXTRACTED" \
            --out_dir "$DL_EXTRACTED"
        echo "[0a/3] Done."
    else
        echo "[0a/3] bicar_uids.json exists, skipping scan."
    fi

    # 0b. Render 10 GT views per character (Blender Cycles)
    echo ""
    echo "[0b/3] Rendering GT multi-view images (Blender Cycles) ..."
    python "${SCRIPT_DIR}/bicar_multiview_render.py" \
        --input_models_path "$BICAR_UIDS" \
        --bicar_root        "$DL_EXTRACTED" \
        --save_folder       "$BICAR_RENDER" \
        --blender_install_path "$BLENDER" \
        --num_views 10 --yaw_start 0 --yaw_end 360 \
        --resolution 512 --ortho_scale 1.35
    echo "[0b/3] Done."

    # 0c. Pack rendered PNGs → preprocessed/{id}_{view}/char/
    echo ""
    echo "[0c/3] Packing GT renders to preprocessed format ..."
    python "${SCRIPT_DIR}/pack_multiview_to_animateddrawings.py" \
        --render_root             "$BICAR_RENDER" \
        --out_preprocessed_root   "$BICAR_PREPROCESSED" \
        --uids_json_out           "$BICAR_PREPROCESSED/uids.json" \
        --char_id_mode            folder \
        --bg_rgb                  240,240,240
    echo "[0c/3] Done."

    # Generate base_uids.json + base_uids_00.json from preprocessed/
    echo ""
    echo "[0d/3] Generating pipeline UID lists ..."
    python "${SCRIPT_DIR}/prep_3dbicar_uids.py" \
        --preprocessed "$BICAR_PREPROCESSED"
    echo "[0d/3] Done."

else
    echo "[0/3] SKIPPED (--skip-gt-render)"
    # Still ensure UID lists exist for later stages
    if [ ! -f "$UID_LIST" ]; then
        echo "  Generating UID lists from existing preprocessed/ ..."
        python "${SCRIPT_DIR}/prep_3dbicar_uids.py" \
            --preprocessed "$BICAR_PREPROCESSED"
    fi
fi

# ── Stage 1: Wonder3D multi-view generation ────────────────────────────────
if [ "$SKIP_MV" -eq 0 ]; then
    echo ""
    echo "[1/3] Wonder3D mv.py (GPUs: $GPUS) ..."
    python "${SCRIPT_DIR}/run_mv_parallel.py" \
        --dsu_dir  "$DSU_DIR" \
        --data_dir "$BICAR_PREPROCESSED" \
        --gpus $GPUS $RESUME
    echo "[1/3] Done."
else
    echo "[1/3] SKIPPED (--skip-mv)"
fi

# ── Stage 2: instant-nsr 3D reconstruction ────────────────────────────────
if [ "$SKIP_RECON" -eq 0 ]; then
    echo ""
    echo "[2/3] instant-nsr recon.py (GPUs: $GPUS) ..."
    python "${SCRIPT_DIR}/run_recon_parallel.py" \
        --dsu_dir  "$DSU_DIR" \
        --data_dir "$BICAR_PREPROCESSED" \
        --gpus $GPUS $RESUME
    echo "[2/3] Done."
else
    echo "[2/3] SKIPPED (--skip-recon)"
fi

# ── Stage 3: 10-view reconstructed OBJ render → SPECSIA-15k ───────────────
if [ "$SKIP_RENDER" -eq 0 ]; then
    echo ""
    echo "[3/3] 10-view OBJ render → $SPECSIA_OUTPUT ..."
    python "${SCRIPT_DIR}/run_render_obj_parallel.py" \
        --data_dir     "$BICAR_PREPROCESSED" \
        --output_root  "$SPECSIA_OUTPUT" \
        --uid_list     "$BASE_UID_LIST" \
        --blender      "$BLENDER" \
        --config_blend "$CONFIG_BLEND" \
        --script       "$RENDER_SCRIPT" \
        --num_workers  4 --align $RESUME
    echo "[3/3] Done."
else
    echo "[3/3] SKIPPED (--skip-render)"
fi

echo ""
echo "Pipeline complete. Dataset at: $SPECSIA_OUTPUT"

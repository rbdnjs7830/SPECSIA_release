"""
Parallel renderer: Wonder3D-reconstructed OBJ → 10-view color/pos/edge renders.

Alignment strategy (--align):
  Stage A — 3D coarse (front view only, once per character):
    Grid search over loc_x/z (49 candidates, Blender 256px mask renders)
    → best_locx, best_locz applied to ALL 10 yaw renders

  Stage B — 10-view render with best loc_x, loc_z

  Stage C — 2D fine (PIL only, no Blender):
    1. Estimate SCALE once from front view (consistent across views)
       full coarse+fine search: ~17k PIL ops
    2. Per view: find best tx/ty with fixed scale
       translate-only coarse+fine: ~800 PIL ops × 10 views = ~8k ops
    3. Apply (scale, tx_vi, ty_vi) to color, pos; re-derive edge from aligned pos

    GT mask per view: prefers char/mask.png → char/texture.png alpha
                      → background subtraction from char/texture_with_bg.png

Total with --align: ~65s/char  (Stage A ~25s + B ~30s + C ~10s)
Total without     : ~30s/char

Output layout (in --output_root = SPECSIA-15k/):
    {uid}/
        mesh/blender_render/rest_pose/
            color/0001.png
            pos/0001.png
            edge/0001.png
        char -> symlink to preprocessed/{uid}/char/   (GT target)

Usage:
    python run_render_obj_parallel.py \\
        --data_dir  /mnt/hdd/kyuwon/3DBiCar_test/preprocessed \\
        --output_root /mnt/hdd/kyuwon/SPECSIA-15k \\
        --uid_list  /mnt/hdd/kyuwon/3DBiCar_test/preprocessed/base_uids.json \\
        --blender   /path/to/blender \\
        --config_blend /path/to/config_ortho.blend \\
        --script    /path/to/blender_render_obj.py \\
        --num_workers 4 --align
"""

import os
import sys
import glob
import json
import argparse
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _find_blender() -> str:
    from_env = os.environ.get("BLENDER")
    if from_env and os.path.isfile(from_env) and os.access(from_env, os.X_OK):
        return from_env
    bundled = os.path.join(_repo_root(), "blender-3.6.14-linux-x64", "blender")
    if os.path.isfile(bundled) and os.access(bundled, os.X_OK):
        return bundled
    which = shutil.which("blender")
    if which:
        return which
    raise RuntimeError(
        "Blender not found. Run `source dataset_prep/setup.sh` to download it, "
        "or set the $BLENDER environment variable."
    )


def _find_config_blend() -> str:
    from_env = os.environ.get("CONFIG_BLEND")
    if from_env and os.path.isfile(from_env):
        return from_env
    default = os.path.join(
        _repo_root(), "external", "DrawingSpinUp",
        "3_style_translator", "configs", "blender", "config_ortho.blend"
    )
    if os.path.isfile(default):
        return default
    raise RuntimeError(
        "config_ortho.blend not found. Run `source dataset_prep/setup.sh` first, "
        "or set $CONFIG_BLEND."
    )
from align_rest_pose import (
    estimate_transform,
    transform_image,
    estimate_background_rgba,
    bbox_from_mask,
    compute_iou,
)


# ---------------------------------------------------------------------------
# Edge from pos
# ---------------------------------------------------------------------------

def pos2edge(pos_path: str):
    pos = cv2.imread(pos_path, cv2.IMREAD_UNCHANGED)
    if pos is None:
        return None
    pos_f = pos.astype(np.float32) / 255.0
    b, g, r, a = cv2.split(pos_f)
    b[a < 1] = 2; g[a < 1] = 2; r[a < 1] = 2

    def sobel(ch):
        gx = cv2.Sobel(ch, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy)

    edges = np.maximum(np.maximum(sobel(b), sobel(g)), sobel(r))
    return ((edges > 0.3) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Mask IoU (cv2-based, for Stage A 3D grid search)
# ---------------------------------------------------------------------------

def read_mask_cv2(path: str):
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[:, :, 3] if m.shape[2] == 4 else cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return (m > 127).astype(np.uint8)


def iou_np(a, b):
    return float(np.logical_and(a, b).sum()) / float(np.logical_or(a, b).sum() + 1e-8)


# ---------------------------------------------------------------------------
# GT mask helpers (Stage C)
# ---------------------------------------------------------------------------

def _derive_gt_mask(data_dir: str, base_uid: str, vi: int, res: int):
    """
    Load GT silhouette mask for view vi.  Priority:
      1. char/mask.png
      2. char/texture.png  (alpha channel)
      3. char/texture_with_bg.png  (background subtraction, bg=240,240,240)
    Returns PIL Image mode="L", size=(res, res), or None if not found.
    """
    uid_view = f"{base_uid}_{vi:02d}"
    char_dir = os.path.join(data_dir, uid_view, "char")

    def _resize(m):
        return m.resize((res, res), Image.NEAREST) if m.size != (res, res) else m

    mask_path = os.path.join(char_dir, "mask.png")
    if os.path.exists(mask_path):
        return _resize(Image.open(mask_path).convert("L"))

    tex_path = os.path.join(char_dir, "texture.png")
    if os.path.exists(tex_path):
        alpha = np.array(Image.open(tex_path).convert("RGBA"))[:, :, 3]
        return _resize(Image.fromarray((alpha > 5).astype(np.uint8) * 255).convert("L"))

    tex_bg_path = os.path.join(char_dir, "texture_with_bg.png")
    if os.path.exists(tex_bg_path):
        arr = np.array(Image.open(tex_bg_path).convert("RGB")).astype(int)
        diff = np.abs(arr - np.array([240, 240, 240])).max(axis=2)
        return _resize(Image.fromarray((diff > 20).astype(np.uint8) * 255).convert("L"))

    return None


def _src_mask_from_rendered(color_png: str, res: int):
    """Extract alpha-based silhouette mask from a rendered RGBA color PNG."""
    img = Image.open(color_png).convert("RGBA")
    if img.size != (res, res):
        img = img.resize((res, res), Image.BICUBIC)
    alpha = np.array(img)[:, :, 3]
    return Image.fromarray((alpha > 5).astype(np.uint8) * 255).convert("L")


# ---------------------------------------------------------------------------
# Translate-only search (scale is fixed) — Stage C per-view step
# ---------------------------------------------------------------------------

def _find_translate(src_mask, tgt_mask, scale, out_size,
                    coarse_offsets=range(-48, 49, 4), fine_range=6):
    """
    Find best (tx, ty) for a fixed scale using IoU grid search.
    ~800 PIL ops vs ~17k for full estimate_transform.
    """
    src_np = np.array(src_mask)
    tgt_np = np.array(tgt_mask)
    src_bbox = bbox_from_mask(src_np)
    tgt_bbox = bbox_from_mask(tgt_np)

    if src_bbox is None or tgt_bbox is None:
        return 0.0, 0.0

    scx = (src_bbox[0] + src_bbox[2]) / 2.0
    scy = (src_bbox[1] + src_bbox[3]) / 2.0
    tcx = (tgt_bbox[0] + tgt_bbox[2]) / 2.0
    tcy = (tgt_bbox[1] + tgt_bbox[3]) / 2.0

    base_tx = tcx - scx * scale
    base_ty = tcy - scy * scale

    best_iou = -1.0
    best_tx = base_tx
    best_ty = base_ty

    # coarse
    for dx in coarse_offsets:
        for dy in coarse_offsets:
            tx = base_tx + dx
            ty = base_ty + dy
            moved = transform_image(src_mask, scale, tx, ty, out_size, Image.NEAREST, fill=0)
            v = compute_iou(moved, tgt_mask)
            if v > best_iou:
                best_iou, best_tx, best_ty = v, float(tx), float(ty)

    # fine
    for ddx in range(-fine_range, fine_range + 1):
        for ddy in range(-fine_range, fine_range + 1):
            tx = best_tx + ddx
            ty = best_ty + ddy
            moved = transform_image(src_mask, scale, tx, ty, out_size, Image.NEAREST, fill=0)
            v = compute_iou(moved, tgt_mask)
            if v > best_iou:
                best_iou, best_tx, best_ty = v, float(tx), float(ty)

    return best_tx, best_ty


# ---------------------------------------------------------------------------
# Blender subprocess helpers
# ---------------------------------------------------------------------------

def _blender_mask(blender, config_blend, engine, script, obj_file, out_dir,
                  yaw_deg, loc_x, loc_y, loc_z, pitch_deg, res):
    os.makedirs(out_dir, exist_ok=True)
    cmd = (
        f'"{blender}" -b "{config_blend}" -E {engine} '
        f'--python "{script}" -- '
        f'--obj_file "{obj_file}" '
        f'--output_dir "{out_dir}" '
        f'--yaw_deg {yaw_deg} '
        f'--loc_x {loc_x} --loc_y {loc_y} --loc_z {loc_z} '
        f'--pitch_deg {pitch_deg} '
        f'--render_mask_only --res {res}'
    )
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pngs = sorted(glob.glob(os.path.join(out_dir, "mask_render", "*.png")))
    return pngs[-1] if pngs else None


def _blender_colorpos(blender, config_blend, engine, script, obj_file, out_dir,
                      yaw_deg, loc_x, loc_z, pitch_deg, res):
    cmd = (
        f'"{blender}" -b "{config_blend}" -E {engine} '
        f'--python "{script}" -- '
        f'--obj_file "{obj_file}" '
        f'--output_dir "{out_dir}" '
        f'--yaw_deg {yaw_deg} '
        f'--loc_x {loc_x} --loc_z {loc_z} '
        f'--pitch_deg {pitch_deg} '
        f'--res {res}'
    )
    subprocess.run(cmd, shell=True, check=False)


# ---------------------------------------------------------------------------
# Per-character worker
# ---------------------------------------------------------------------------

def render_character(base_uid: str, args_dict: dict) -> str:
    data_dir    = args_dict["data_dir"]
    output_root = args_dict["output_root"]
    blender     = args_dict["blender"]
    config      = args_dict["config_blend"]
    script      = args_dict["script"]
    engine      = args_dict["engine"]
    num_views   = args_dict["num_views"]
    do_align    = args_dict["align"]
    loc_range   = args_dict["loc_range"]
    loc_step    = args_dict["loc_step"]
    mask_res    = args_dict["mask_res"]
    render_res  = args_dict["render_res"]
    yaw_start   = args_dict["yaw_start"]
    yaw_end     = args_dict["yaw_end"]

    front_dir = os.path.join(data_dir, f"{base_uid}_00")
    obj_files = glob.glob(os.path.join(front_dir, "mesh", "*.obj"))
    if not obj_files:
        return f"[SKIP] {base_uid}: no .obj in {os.path.join(front_dir, 'mesh')}"
    obj_file = obj_files[0]

    span     = yaw_end - yaw_start
    loc_list = np.arange(-loc_range, loc_range + 1e-6, loc_step).tolist()
    results  = []
    t0       = time.time()

    # ── Stage A: 3D coarse align — front view only ────────────────────────────
    best_locx  = 0.0
    best_locz  = 0.0
    best_pitch = 0.0

    if do_align:
        gt_mask_path = os.path.join(data_dir, f"{base_uid}_00", "char", "mask.png")
        gt = read_mask_cv2(gt_mask_path)
        if gt is None:
            results.append(f"[WARN] {base_uid}: no GT mask for 3D align")
        else:
            tmp_dir = os.path.join(output_root, f"{base_uid}_00", "_tmp_align")
            try:
                # A1: XZ position search (pitch=0)
                best_3d = -1.0
                for x in loc_list:
                    for z in loc_list:
                        cand = os.path.join(tmp_dir, f"x{x:+.3f}_z{z:+.3f}")
                        pred_path = _blender_mask(
                            blender, config, engine, script, obj_file, cand,
                            yaw_start, x, 0.0, z, 0.0, mask_res
                        )
                        if pred_path is None:
                            continue
                        pred = read_mask_cv2(pred_path)
                        if pred is None:
                            continue
                        gt_rs = gt
                        if pred.shape != gt.shape:
                            gt_rs = cv2.resize(
                                gt.astype(np.uint8),
                                (pred.shape[1], pred.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        score = iou_np(pred, gt_rs)
                        if score > best_3d:
                            best_3d = score
                            best_locx, best_locz = float(x), float(z)

                # A2: pitch search at best XZ location
                pitch_list = args_dict.get("pitch_list",
                                           list(range(-20, 21, 5)))  # -20..+20 step 5
                best_pitch_score = best_3d  # baseline: pitch=0 already evaluated
                for p in pitch_list:
                    if p == 0:
                        continue  # already done above
                    cand = os.path.join(tmp_dir, f"pitch{p:+d}")
                    pred_path = _blender_mask(
                        blender, config, engine, script, obj_file, cand,
                        yaw_start, best_locx, 0.0, best_locz, float(p), mask_res
                    )
                    if pred_path is None:
                        continue
                    pred = read_mask_cv2(pred_path)
                    if pred is None:
                        continue
                    gt_rs = gt
                    if pred.shape != gt.shape:
                        gt_rs = cv2.resize(
                            gt.astype(np.uint8),
                            (pred.shape[1], pred.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    score = iou_np(pred, gt_rs)
                    if score > best_pitch_score:
                        best_pitch_score = score
                        best_pitch = float(p)

            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            results.append(
                f"[3D] {base_uid}: iou={best_pitch_score:.3f} "
                f"loc=({best_locx:.3f},{best_locz:.3f}) pitch={best_pitch:.0f}°"
            )

    # ── Stage B: render all 10 views ──────────────────────────────────────────
    for vi in range(num_views):
        yaw_deg  = yaw_start + span * (vi / num_views)
        uid_view = f"{base_uid}_{vi:02d}"
        out_dir  = os.path.join(
            output_root, uid_view, "mesh", "blender_render", "rest_pose"
        )
        os.makedirs(out_dir, exist_ok=True)

        char_link = os.path.join(output_root, uid_view, "char")
        char_src  = os.path.join(data_dir, uid_view, "char")
        if not os.path.exists(char_link) and os.path.isdir(char_src):
            os.symlink(char_src, char_link)

        _blender_colorpos(
            blender, config, engine, script, obj_file, out_dir,
            yaw_deg, best_locx, best_locz, best_pitch, render_res
        )

        # preliminary edge (overwritten after 2D align if do_align)
        pos_dir  = os.path.join(out_dir, "pos")
        edge_dir = os.path.join(out_dir, "edge")
        os.makedirs(edge_dir, exist_ok=True)
        for pos_fn in sorted(glob.glob(os.path.join(pos_dir, "*.png"))):
            edge = pos2edge(pos_fn)
            if edge is not None:
                cv2.imwrite(
                    os.path.join(edge_dir, os.path.basename(pos_fn)), 255 - edge
                )

    # ── Stage C: 2D fine align — scale once, tx/ty per view ──────────────────
    if do_align:
        try:
            out_size = (render_res, render_res)

            # Step 1: estimate scale from front view
            src_front_color = sorted(glob.glob(os.path.join(
                output_root, f"{base_uid}_00",
                "mesh", "blender_render", "rest_pose", "color", "*.png"
            )))
            if not src_front_color:
                raise RuntimeError("no front-view color render found")

            src_mask_front = _src_mask_from_rendered(src_front_color[0], render_res)
            gt_mask_front  = _derive_gt_mask(data_dir, base_uid, 0, render_res)
            if gt_mask_front is None:
                raise RuntimeError("no GT mask for front view")

            tfm_front = estimate_transform(src_mask_front, gt_mask_front)
            scale = tfm_front["scale"]

            results.append(
                f"[2D-scale] {base_uid}: scale={scale:.4f} "
                f"front_iou={tfm_front['iou']:.4f}"
            )

            # Step 2: per-view translate search + apply
            for vi in range(num_views):
                uid_view = f"{base_uid}_{vi:02d}"
                rest_dir = os.path.join(
                    output_root, uid_view, "mesh", "blender_render", "rest_pose"
                )

                # rendered src mask for this view
                color_pngs = sorted(glob.glob(os.path.join(rest_dir, "color", "*.png")))
                if not color_pngs:
                    continue
                src_mask_vi = _src_mask_from_rendered(color_pngs[0], render_res)

                # GT mask for this view (view-specific silhouette)
                gt_mask_vi = _derive_gt_mask(data_dir, base_uid, vi, render_res)
                if gt_mask_vi is None:
                    gt_mask_vi = gt_mask_front  # fallback

                tx_vi, ty_vi = _find_translate(src_mask_vi, gt_mask_vi, scale, out_size)

                # apply scale + view-specific translate to color and pos
                for modality, resample in [("color", Image.BICUBIC), ("pos", Image.BILINEAR)]:
                    mod_dir = os.path.join(rest_dir, modality)
                    for png in sorted(glob.glob(os.path.join(mod_dir, "*.png"))):
                        img  = Image.open(png).convert("RGBA")
                        fill = estimate_background_rgba(img)
                        aligned = transform_image(
                            img, scale, tx_vi, ty_vi, out_size, resample, fill=fill
                        )
                        aligned.save(png)

            # Step 3: re-derive edge from aligned pos
            for vi in range(num_views):
                uid_view = f"{base_uid}_{vi:02d}"
                rest_dir = os.path.join(
                    output_root, uid_view, "mesh", "blender_render", "rest_pose"
                )
                pos_dir  = os.path.join(rest_dir, "pos")
                edge_dir = os.path.join(rest_dir, "edge")
                for pos_fn in sorted(glob.glob(os.path.join(pos_dir, "*.png"))):
                    edge = pos2edge(pos_fn)
                    if edge is not None:
                        cv2.imwrite(
                            os.path.join(edge_dir, os.path.basename(pos_fn)), 255 - edge
                        )

        except Exception as e:
            results.append(f"[ERROR 2D-align] {base_uid}: {e}")

    elapsed = time.time() - t0
    n_color = len(glob.glob(os.path.join(
        output_root, f"{base_uid}_00",
        "mesh", "blender_render", "rest_pose", "color", "*.png"
    )))
    results.append(
        f"[OK] {base_uid}: {num_views} views, {n_color} color frames, t={elapsed:.1f}s"
    )
    return "\n".join(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("OBJ-only 10-view renderer → SPECSIA-15k dataset")
    parser.add_argument("--data_dir",     required=True,
                        help="preprocessed/ root (source OBJ + GT char/)")
    parser.add_argument("--output_root",  required=True,
                        help="SPECSIA-15k/ output root")
    parser.add_argument("--uid_list",     required=True,
                        help="JSON list of BASE char ids (e.g. ['5','6',...])")
    parser.add_argument("--blender",      default=None,
                        help="path to blender executable (default: auto-detect via $BLENDER or repo bundle)")
    parser.add_argument("--config_blend", default=None,
                        help="path to config_ortho.blend (default: external/DrawingSpinUp/...)")
    parser.add_argument("--script",       default=None,
                        help="path to blender_render_obj.py (default: dataset_prep/blender_render_obj.py)")
    parser.add_argument("--engine",       default="BLENDER_EEVEE",
                        choices=["BLENDER_EEVEE", "CYCLES"])
    parser.add_argument("--num_views",    type=int,   default=10)
    parser.add_argument("--yaw_start",    type=float, default=0.0)
    parser.add_argument("--yaw_end",      type=float, default=360.0)
    parser.add_argument("--render_res",   type=int,   default=512)
    parser.add_argument("--mask_res",     type=int,   default=256)
    parser.add_argument("--align",        action="store_true",
                        help="3D coarse (front-view XZ + pitch) + 2D fine (per-view scale+translate)")
    parser.add_argument("--loc_range",    type=float, default=0.15,
                        help="3D XZ search range (±)")
    parser.add_argument("--loc_step",     type=float, default=0.05,
                        help="3D XZ search step")
    parser.add_argument("--pitch_range",  type=int,   default=20,
                        help="pitch search range ±degrees (default ±20°)")
    parser.add_argument("--pitch_step",   type=int,   default=5,
                        help="pitch search step in degrees (default 5°)")
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--resume",       action="store_true",
                        help="Skip characters whose color renders already exist")
    args = parser.parse_args()

    blender     = args.blender     or _find_blender()
    config_blend = args.config_blend or _find_config_blend()
    script      = args.script      or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "blender_render_obj.py"
    )

    with open(args.uid_list) as f:
        base_uids = json.load(f)

    if args.resume:
        def _render_done(uid):
            color_dir = os.path.join(
                args.output_root, f"{uid}_00",
                "mesh", "blender_render", "rest_pose", "color"
            )
            return os.path.isdir(color_dir) and len(os.listdir(color_dir)) > 0
        pending = [u for u in base_uids if not _render_done(u)]
        print(f"Resume: {len(base_uids) - len(pending)} done, {len(pending)} remaining")
        base_uids = pending

    os.makedirs(args.output_root, exist_ok=True)
    print(
        f"Processing {len(base_uids)} characters × {args.num_views} views "
        f"→ {args.output_root}  (workers={args.num_workers}, align={args.align})"
    )

    pitch_list = list(range(-args.pitch_range, args.pitch_range + 1, args.pitch_step))

    args_dict = {
        "data_dir":     args.data_dir,
        "output_root":  args.output_root,
        "blender":      blender,
        "config_blend": config_blend,
        "script":       script,
        "engine":       args.engine,
        "num_views":    args.num_views,
        "yaw_start":    args.yaw_start,
        "yaw_end":      args.yaw_end,
        "render_res":   args.render_res,
        "mask_res":     args.mask_res,
        "align":        args.align,
        "loc_range":    args.loc_range,
        "loc_step":     args.loc_step,
        "pitch_list":   pitch_list,
    }

    if args.num_workers == 1:
        for uid in base_uids:
            print(render_character(uid, args_dict), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as exe:
            futures = {exe.submit(render_character, uid, args_dict): uid
                       for uid in base_uids}
            for fut in as_completed(futures):
                uid = futures[fut]
                try:
                    print(fut.result(), flush=True)
                except Exception as e:
                    print(f"[ERROR] {uid}: {e}", flush=True)


if __name__ == "__main__":
    main()

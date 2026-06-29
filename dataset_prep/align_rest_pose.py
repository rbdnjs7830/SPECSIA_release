#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Align rendered rest-pose frames to GT character silhouette.

Example:
python align_rest_pose.py \
    --ids_json /mnt/hdd/kyuwon/DrawingSpinUp_kyu/dataset/AnimatedDrawings/drawings_uids_rebut.json \
    --root /mnt/hdd/kyuwon/DrawingSpinUp_kyu/dataset/AnimatedDrawings/output_crm \
    --src_rel mesh/blender_render/rest_pose \
    --dst_rel mesh/blender_render/rest_pose_aligned \
    --modalities color edge pos \
    --num_views 10 \
    --target_size 512

If your Trainer hardcodes "rest_pose", use:
    --overwrite --backup
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        ids = json.load(f)
    if not isinstance(ids, list):
        raise ValueError("ids_json must be a JSON list.")
    return [str(x) for x in ids]


def read_rgba(path):
    return Image.open(path).convert("RGBA")


def read_l(path):
    return Image.open(path).convert("L")


def mask_from_rgba(img, thr=5):
    arr = np.array(img.convert("RGBA"))
    return (arr[:, :, 3] > thr).astype(np.uint8) * 255


def mask_from_texture_or_mask(uid_root, target_size=None):
    """
    Prefer char/mask.png if it exists.
    Otherwise use alpha channel of char/texture.png.
    Resize mask to texture size or target_size.
    """
    texture_path = os.path.join(uid_root, "char", "texture.png")
    mask_path = os.path.join(uid_root, "char", "mask.png")

    if not os.path.exists(texture_path):
        raise FileNotFoundError(f"Missing texture: {texture_path}")

    texture = read_rgba(texture_path)

    if target_size is not None:
        canvas_size = (target_size, target_size)
    else:
        canvas_size = texture.size

    if os.path.exists(mask_path):
        mask = read_l(mask_path)
    else:
        mask = Image.fromarray(mask_from_rgba(texture))

    if mask.size != canvas_size:
        mask = mask.resize(canvas_size, Image.NEAREST)

    return mask


def bbox_from_mask(mask_np):
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def bbox_center_size(bbox):
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w = x1 - x0
    h = y1 - y0
    return cx, cy, w, h


def estimate_background_rgba(img):
    """
    Estimate background RGB from border pixels.
    Returns transparent RGBA with estimated RGB.
    """
    arr = np.array(img.convert("RGBA"))
    h, w = arr.shape[:2]

    border = np.concatenate(
        [arr[0, :, :], arr[h - 1, :, :], arr[:, 0, :], arr[:, w - 1, :]], axis=0
    )

    # Prefer low-alpha border pixels
    low_alpha = border[border[:, 3] < 8]

    if len(low_alpha) > 0:
        rgb = np.median(low_alpha[:, :3], axis=0).astype(np.uint8)
    else:
        rgb = np.median(border[:, :3], axis=0).astype(np.uint8)

    return (int(rgb[0]), int(rgb[1]), int(rgb[2]), 0)


def transform_image(img, scale, tx, ty, out_size, resample, fill=None):
    """
    Scale whole image, then paste to canvas.
    tx, ty are top-left paste positions in output canvas.
    """
    src_w, src_h = img.size
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    img_resized = img.resize((new_w, new_h), resample)

    if fill is None:
        if img.mode == "RGBA":
            fill = (0, 0, 0, 0)
        elif img.mode == "L":
            fill = 0
        else:
            fill = (255, 255, 255)

    canvas = Image.new(img.mode, out_size, fill)

    x = int(round(tx))
    y = int(round(ty))

    # crop if pasted outside canvas
    src_x0 = max(0, -x)
    src_y0 = max(0, -y)
    dst_x0 = max(0, x)
    dst_y0 = max(0, y)

    src_x1 = min(new_w, out_size[0] - x)
    src_y1 = min(new_h, out_size[1] - y)

    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return canvas

    crop = img_resized.crop((src_x0, src_y0, src_x1, src_y1))
    canvas.paste(crop, (dst_x0, dst_y0))
    return canvas


def compute_iou(mask_a, mask_b):
    a = np.array(mask_a) > 0
    b = np.array(mask_b) > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def estimate_transform(src_mask, tgt_mask):
    """
    Estimate scale + translation by bbox init + IoU grid search.
    Transform maps source image to target canvas.
    """
    src_np = np.array(src_mask)
    tgt_np = np.array(tgt_mask)

    src_bbox = bbox_from_mask(src_np)
    tgt_bbox = bbox_from_mask(tgt_np)

    if src_bbox is None:
        raise RuntimeError("Source mask is empty.")
    if tgt_bbox is None:
        raise RuntimeError("Target mask is empty.")

    scx, scy, sw, sh = bbox_center_size(src_bbox)
    tcx, tcy, tw, th = bbox_center_size(tgt_bbox)

    # Initial scale based on bbox size.
    base_scale = 0.5 * ((tw / max(sw, 1)) + (th / max(sh, 1)))

    best = {
        "iou": -1.0,
        "scale": base_scale,
        "tx": 0.0,
        "ty": 0.0,
    }

    out_size = tgt_mask.size

    def try_grid(scale_list, offset_list):
        nonlocal best

        for scale in scale_list:
            # Align scaled source bbox center to target bbox center, then add residual dx/dy.
            base_tx = tcx - scx * scale
            base_ty = tcy - scy * scale

            for dx in offset_list:
                for dy in offset_list:
                    tx = base_tx + dx
                    ty = base_ty + dy

                    moved = transform_image(
                        src_mask,
                        scale,
                        tx,
                        ty,
                        out_size,
                        Image.NEAREST,
                        fill=0,
                    )
                    iou = compute_iou(moved, tgt_mask)

                    if iou > best["iou"]:
                        best = {
                            "iou": iou,
                            "scale": float(scale),
                            "tx": float(tx),
                            "ty": float(ty),
                        }

    # Coarse search
    scales = base_scale * np.linspace(0.75, 1.35, 25)
    offsets = range(-48, 49, 4)
    try_grid(scales, offsets)

    # Fine search around best
    fine_scales = best["scale"] * np.linspace(0.94, 1.06, 13)
    fine_offsets = range(-6, 7, 1)

    # Recenter fine search around current best tx/ty by converting to residual offset
    # Simpler: directly search local tx/ty around best.
    current_best = dict(best)

    for scale in fine_scales:
        for ddx in fine_offsets:
            for ddy in fine_offsets:
                tx = current_best["tx"] + ddx
                ty = current_best["ty"] + ddy

                moved = transform_image(
                    src_mask,
                    scale,
                    tx,
                    ty,
                    out_size,
                    Image.NEAREST,
                    fill=0,
                )
                iou = compute_iou(moved, tgt_mask)

                if iou > best["iou"]:
                    best = {
                        "iou": iou,
                        "scale": float(scale),
                        "tx": float(tx),
                        "ty": float(ty),
                    }

    return best


def align_uid(
    uid,
    root,
    src_rel,
    dst_rel,
    modalities,
    num_views,
    target_size,
    overwrite=False,
    backup=False,
):
    uid_root = os.path.join(root, uid)
    src_root = os.path.join(uid_root, src_rel)

    if overwrite:
        dst_root = src_root
    else:
        dst_root = os.path.join(uid_root, dst_rel)

    color_0001 = os.path.join(src_root, "color", "0001.png")
    if not os.path.exists(color_0001):
        raise FileNotFoundError(f"Missing source color: {color_0001}")

    tgt_mask = mask_from_texture_or_mask(uid_root, target_size=target_size)

    src_color = read_rgba(color_0001)
    if target_size is not None and src_color.size != (target_size, target_size):
        src_color = src_color.resize((target_size, target_size), Image.BICUBIC)

    src_mask = Image.fromarray(mask_from_rgba(src_color)).convert("L")

    tfm = estimate_transform(src_mask, tgt_mask)

    print(
        f"[{uid}] IoU={tfm['iou']:.4f}, "
        f"scale={tfm['scale']:.4f}, tx={tfm['tx']:.2f}, ty={tfm['ty']:.2f}"
    )

    os.makedirs(dst_root, exist_ok=True)

    # Save debug masks
    debug_dir = os.path.join(dst_root, "_align_debug")
    os.makedirs(debug_dir, exist_ok=True)
    tgt_mask.save(os.path.join(debug_dir, "target_mask.png"))
    src_mask.save(os.path.join(debug_dir, "source_mask_before.png"))
    moved_mask = transform_image(
        src_mask,
        tfm["scale"],
        tfm["tx"],
        tfm["ty"],
        tgt_mask.size,
        Image.NEAREST,
        fill=0,
    )
    moved_mask.save(os.path.join(debug_dir, "source_mask_after.png"))

    # Save transform json
    with open(os.path.join(debug_dir, "transform.json"), "w") as f:
        json.dump(tfm, f, indent=2)

    for modality in modalities:
        src_dir = os.path.join(src_root, modality)
        dst_dir = os.path.join(dst_root, modality)

        if not os.path.isdir(src_dir):
            print(f"[WARN] Missing modality dir: {src_dir}")
            continue

        os.makedirs(dst_dir, exist_ok=True)

        for i in range(1, num_views + 1):
            name = f"{i:04d}.png"
            src_path = os.path.join(src_dir, name)
            dst_path = os.path.join(dst_dir, name)

            if not os.path.exists(src_path):
                print(f"[WARN] Missing frame: {src_path}")
                continue

            if overwrite and backup:
                bak_path = src_path + ".bak"
                if not os.path.exists(bak_path):
                    Image.open(src_path).save(bak_path)

            img = Image.open(src_path).convert("RGBA")

            if target_size is not None and img.size != (target_size, target_size):
                if modality == "edge":
                    img = img.resize((target_size, target_size), Image.NEAREST)
                elif modality == "pos":
                    img = img.resize((target_size, target_size), Image.BILINEAR)
                else:
                    img = img.resize((target_size, target_size), Image.BICUBIC)

            # modality별 배경/보간 방식
            if modality == "edge":
                # edge는 grayscale로 저장하고, 빈 영역은 흰색으로 채움
                img = img.convert("L")
                fill_rgba = 255
                resample = Image.NEAREST
            elif modality == "pos":
                fill_rgba = estimate_background_rgba(img)
                resample = Image.BILINEAR
            else:  # color
                fill_rgba = estimate_background_rgba(img)
                resample = Image.BICUBIC

            aligned = transform_image(
                img,
                tfm["scale"],
                tfm["tx"],
                tfm["ty"],
                tgt_mask.size,
                resample,
                fill=fill_rgba,
            )
            aligned.save(dst_path)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ids_json", type=str, required=True)
    ap.add_argument("--root", type=str, required=True)

    ap.add_argument("--src_rel", type=str, default="mesh/blender_render/rest_pose")
    ap.add_argument(
        "--dst_rel", type=str, default="mesh/blender_render/rest_pose_aligned"
    )

    ap.add_argument(
        "--modalities", type=str, nargs="+", default=["color", "edge", "pos"]
    )
    ap.add_argument("--num_views", type=int, default=10)
    ap.add_argument("--target_size", type=int, default=512)

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--backup", action="store_true")

    args = ap.parse_args()

    ids = load_ids(args.ids_json)

    for uid in ids:
        try:
            align_uid(
                uid=uid,
                root=args.root,
                src_rel=args.src_rel,
                dst_rel=args.dst_rel,
                modalities=args.modalities,
                num_views=args.num_views,
                target_size=args.target_size,
                overwrite=args.overwrite,
                backup=args.backup,
            )
        except Exception as e:
            print(f"[ERROR] {uid}: {e}")


if __name__ == "__main__":
    main()

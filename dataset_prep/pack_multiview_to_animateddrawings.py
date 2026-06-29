#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


"""
python pack_multiview_to_animateddrawings.py \
    --render_root /mnt/hdd/kyuwon/3DBiCar_test/render \
    --out_preprocessed_root /mnt/hdd/kyuwon/3DBiCar_test/formed \
    --char_id_mode folder \
    --bg_rgb 240,240,240 \
    --uids_json_out /mnt/hdd/kyuwon/3DBiCar_test/formed/uids.json
"""

VIEW_RE_DEFAULT = r"view_(\d+)_rgba_outline\.png$"


def parse_rgb(s: str):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError("--bg_rgb must be like '240,240,240'")
    rgb = tuple(int(x) for x in parts)
    for v in rgb:
        if v < 0 or v > 255:
            raise ValueError("bg_rgb values must be 0..255")
    return rgb


def alpha_to_mask(alpha: np.ndarray, thr: int = 1):
    return (alpha >= thr).astype(np.uint8) * 255


def composite_on_solid_bg(rgba: np.ndarray, bg_rgb=(240, 240, 240)):
    rgb = rgba[..., :3].astype(np.float32)
    a = (rgba[..., 3:4].astype(np.float32) / 255.0)
    bg = np.array(bg_rgb, dtype=np.float32).reshape(1, 1, 3)
    out = rgb * a + bg * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Pack multiview RGBA(+outline) into AnimatedDrawings preprocessed layout + write full uid_view json."
    )
    parser.add_argument("--render_root", type=str, required=True)
    parser.add_argument("--out_preprocessed_root", type=str, required=True)

    parser.add_argument(
        "--char_id_mode",
        type=str,
        default="folder",
        choices=["folder", "index"],
        help="folder: use character folder name as char_id. index: use 0000,0001,...",
    )

    parser.add_argument("--view_regex", type=str, default=VIEW_RE_DEFAULT)
    parser.add_argument("--bg_rgb", type=str, default="240,240,240")
    parser.add_argument("--mask_alpha_thr", type=int, default=1)

    # ✅ 여기: uid_view 풀네임 리스트를 저장
    parser.add_argument(
        "--uids_json_out",
        type=str,
        required=True,
        help="Output path for full uid_view list json. Example items: 9950_00, 9950_01, ...",
    )

    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    render_root = Path(args.render_root)
    out_root = Path(args.out_preprocessed_root)
    view_re = re.compile(args.view_regex)
    bg_rgb = parse_rgb(args.bg_rgb)

    if not render_root.exists():
        raise FileNotFoundError(f"render_root not found: {render_root}")
    ensure_dir(out_root)

    char_folders = sorted([p for p in render_root.iterdir() if p.is_dir()])
    if len(char_folders) == 0:
        raise RuntimeError(f"No character folders under render_root: {render_root}")

    full_uid_list = []

    for ci, char_dir in enumerate(char_folders):
        uid = char_dir.name
        char_id = uid if args.char_id_mode == "folder" else f"{ci:04d}"

        view_files = []
        for f in sorted(char_dir.iterdir()):
            if not f.is_file():
                continue
            m = view_re.search(f.name)
            if m:
                view_idx = int(m.group(1))
                view_files.append((view_idx, f))

        if len(view_files) == 0:
            print(f"[WARN] no view files matched in {char_dir} (regex={args.view_regex})")
            continue

        for view_idx, img_path in view_files:
            view_id = f"{view_idx:02d}"
            full_uid = f"{char_id}_{view_id}"
            full_uid_list.append(full_uid)

            dst_char_dir = out_root / full_uid / "char"
            tex_path = dst_char_dir / "texture.png"
            mask_path = dst_char_dir / "mask.png"
            texbg_path = dst_char_dir / "texture_with_bg.png"

            if args.dry_run:
                print(f"[DRY] {img_path} -> {dst_char_dir}")
                continue

            ensure_dir(dst_char_dir)

            im = Image.open(img_path).convert("RGBA")
            rgba = np.array(im, dtype=np.uint8)

            # texture.png (RGBA)
            Image.fromarray(rgba, mode="RGBA").save(tex_path)

            # mask.png (0/255)
            alpha = rgba[..., 3]
            mask = alpha_to_mask(alpha, thr=args.mask_alpha_thr)
            Image.fromarray(mask, mode="L").save(mask_path)

            # texture_with_bg.png (RGB)
            rgb_bg = composite_on_solid_bg(rgba, bg_rgb=bg_rgb)
            Image.fromarray(rgb_bg, mode="RGB").save(texbg_path)

        print(f"[OK] packed: {uid} -> {char_id}_{{view}} ({len(view_files)} views)")

    # write json: list of full uid names like 9950_00
    out_json = Path(args.uids_json_out)
    if args.dry_run:
        print(f"[DRY] would write json: {out_json} (n={len(full_uid_list)})")
    else:
        ensure_dir(out_json.parent)
        # 정렬: uid, view 순으로 깔끔하게
        full_uid_list_sorted = sorted(full_uid_list)
        with open(out_json, "w") as f:
            json.dump(full_uid_list_sorted, f, indent=2)
        print(f"[OK] wrote full uid_view json: {out_json} (n={len(full_uid_list_sorted)})")

    print("Done.")


if __name__ == "__main__":
    main()

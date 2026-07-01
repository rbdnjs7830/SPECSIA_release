"""
Generate UID list files needed by the SPECSIA-15K pipeline from a raw 3DBiCar download.

Two modes:

  --dl_extracted DIR   Scan the raw download (dl_extracted/{char_id}/) and write:
                         bicar_uids.json         — char IDs for bicar_multiview_render.py

  --preprocessed DIR   Scan the packed preprocessed dir ({char_id}_{view}/) and write:
                         base_uids.json          — base char IDs for run_render_obj_parallel
                         base_uids_00.json       — front-view UIDs for run_mv / run_recon

Run Stage 0 (--dl_extracted) before GT rendering, Stage 1 (--preprocessed) after packing.

Usage:
    # Before GT render:
    python prep_3dbicar_uids.py --dl_extracted /path/to/dl_extracted

    # After pack_multiview_to_animateddrawings.py:
    python prep_3dbicar_uids.py --preprocessed /path/to/preprocessed
"""

import os
import re
import json
import argparse


def scan_dl_extracted(dl_dir: str, out_dir: str):
    char_ids = sorted(
        [d for d in os.listdir(dl_dir)
         if os.path.isdir(os.path.join(dl_dir, d)) and d.isdigit()],
        key=int,
    )
    out_path = os.path.join(out_dir, "bicar_uids.json")
    with open(out_path, "w") as f:
        json.dump(char_ids, f, indent=2)
    print(f"Found {len(char_ids)} characters in {dl_dir}")
    print(f"  → {out_path}")


def scan_preprocessed(pre_dir: str, out_dir: str):
    pattern = re.compile(r"^(\d+)_(\d+)$")
    base_ids = set()
    for name in os.listdir(pre_dir):
        m = pattern.match(name)
        if m and os.path.isdir(os.path.join(pre_dir, name)):
            base_ids.add(m.group(1))

    base_list   = sorted(base_ids, key=int)
    uid_00_list = [f"{b}_00" for b in base_list
                   if os.path.isdir(os.path.join(pre_dir, f"{b}_00"))]

    base_path   = os.path.join(out_dir, "base_uids.json")
    uid_00_path = os.path.join(out_dir, "base_uids_00.json")
    with open(base_path, "w") as f:
        json.dump(base_list, f, indent=2)
    with open(uid_00_path, "w") as f:
        json.dump(uid_00_list, f, indent=2)

    print(f"Found {len(base_list)} characters, {len(uid_00_list)} with front view (_00).")
    print(f"  → {base_path}")
    print(f"  → {uid_00_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dl_extracted", default=None,
                        help="Raw 3DBiCar download root. Writes bicar_uids.json.")
    parser.add_argument("--preprocessed", default=None,
                        help="Packed preprocessed root. Writes base_uids*.json.")
    parser.add_argument("--out_dir", default=None,
                        help="Where to write JSON files (default: same as input dir)")
    args = parser.parse_args()

    if not args.dl_extracted and not args.preprocessed:
        parser.error("Provide --dl_extracted and/or --preprocessed.")

    if args.dl_extracted:
        out = args.out_dir or args.dl_extracted
        scan_dl_extracted(args.dl_extracted, out)

    if args.preprocessed:
        out = args.out_dir or args.preprocessed
        scan_preprocessed(args.preprocessed, out)


if __name__ == "__main__":
    main()

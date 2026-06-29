"""
Upload SPECSIA-15K to HuggingFace Hub.

Each sample (one character × one viewpoint) becomes one row:
    uid       str    "{char_id}_{view_id}"
    color     Image  reconstructed render — input Z
    pos       Image  positional hint — input Z_pos
    edge      Image  edge map (derived from pos)
    mask      Image  foreground silhouette — input Z_mask
    gt        Image  GT render from 3DBiCar T-pose — target Y*

Splits (requires create_splits.py to have been run first):
    train / validation / test

Prerequisites:
    pip install datasets huggingface_hub pillow
    huggingface-cli login   (or set HF_TOKEN env var)

Usage:
    python upload_hf.py \\
        --specsia_root /path/to/SPECSIA-15k \\
        --repo_id      YOUR_HF_USERNAME/SPECSIA-15K \\
        [--dry_run]    # inspect first row without uploading
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

# Module-level root (set in main, used by map worker)
_SPECSIA_ROOT = None


def _load_for_map(example):
    """Worker function for dataset.map() — loads images for one sample."""
    from PIL import Image as PILImage
    uid = example["uid"]
    base = os.path.join(_SPECSIA_ROOT, uid)
    rest_pose = os.path.join(base, "mesh", "blender_render", "rest_pose")

    def first_png(d):
        if not os.path.isdir(d):
            return None
        pngs = sorted(f for f in os.listdir(d) if f.endswith(".png"))
        return os.path.join(d, pngs[0]) if pngs else None

    color_path = first_png(os.path.join(rest_pose, "color"))
    pos_path   = first_png(os.path.join(rest_pose, "pos"))
    edge_path  = first_png(os.path.join(rest_pose, "edge"))
    gt_path    = os.path.join(base, "char", "texture_with_bg.png")
    mask_path  = os.path.join(base, "char", "mask.png")

    missing = [n for n, p in [("color", color_path), ("pos", pos_path),
                               ("gt", gt_path), ("mask", mask_path)]
               if not p or not os.path.exists(p)]
    if missing:
        print(f"[SKIP] {uid}: missing {missing}", flush=True)
        return {"color": None, "pos": None, "edge": None, "mask": None, "gt": None}

    return {
        "color": PILImage.open(color_path).convert("RGB"),
        "pos":   PILImage.open(pos_path).convert("RGB"),
        "edge":  PILImage.open(edge_path).convert("RGB")
               if edge_path and os.path.exists(edge_path) else None,
        "mask":  PILImage.open(mask_path).convert("L"),
        "gt":    PILImage.open(gt_path).convert("RGB"),
    }


def load_sample(root: str, uid: str) -> Optional[Dict[str, Any]]:
    """Load one (uid, viewpoint) sample. Returns None if any required file is missing."""
    from PIL import Image
    base = os.path.join(root, uid)
    rest_pose = os.path.join(base, "mesh", "blender_render", "rest_pose")

    color_dir = os.path.join(rest_pose, "color")
    pos_dir   = os.path.join(rest_pose, "pos")
    edge_dir  = os.path.join(rest_pose, "edge")
    char_dir  = os.path.join(base, "char")

    def first_png(d):
        if not os.path.isdir(d):
            return None
        pngs = sorted(f for f in os.listdir(d) if f.endswith(".png"))
        return os.path.join(d, pngs[0]) if pngs else None

    color_path = first_png(color_dir)
    pos_path   = first_png(pos_dir)
    edge_path  = first_png(edge_dir)
    gt_path    = os.path.join(char_dir, "texture_with_bg.png")
    mask_path  = os.path.join(char_dir, "mask.png")

    missing = [n for n, p in [("color", color_path), ("pos", pos_path),
                               ("gt", gt_path), ("mask", mask_path)]
               if not p or not os.path.exists(p)]
    if missing:
        print(f"[SKIP] {uid}: missing {missing}")
        return None

    return {
        "uid":   uid,
        "color": Image.open(color_path).convert("RGB"),
        "pos":   Image.open(pos_path).convert("RGB"),
        "edge":  Image.open(edge_path).convert("RGB") if edge_path and os.path.exists(edge_path) else None,
        "mask":  Image.open(mask_path).convert("L"),
        "gt":    Image.open(gt_path).convert("RGB"),
    }


def main():
    global _SPECSIA_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument("--specsia_root", required=True,
                        help="SPECSIA-15k root directory")
    parser.add_argument("--repo_id", required=True,
                        help="HuggingFace repo, e.g. Kyu0528/SPECSIA-15K")
    parser.add_argument("--splits_dir", default=None,
                        help="Directory with *_uids.json (default: <specsia_root>/splits)")
    parser.add_argument("--private", action="store_true",
                        help="Create as private repository")
    parser.add_argument("--num_proc", type=int, default=8,
                        help="Parallel workers for image loading (default: 8)")
    parser.add_argument("--token", default=None,
                        help="HuggingFace write token (skips huggingface-cli login)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Load first 3 samples and print info without uploading")
    args = parser.parse_args()

    if args.token:
        from huggingface_hub import login as hf_login
        hf_login(token=args.token, add_to_git_credential=False)

    _SPECSIA_ROOT = args.specsia_root

    splits_dir = args.splits_dir or os.path.join(args.specsia_root, "splits")
    for name in ("train_uids.json", "val_uids.json", "test_uids.json"):
        if not os.path.exists(os.path.join(splits_dir, name)):
            raise FileNotFoundError(
                f"{name} not found. Run create_splits.py first.\n"
                f"  python create_splits.py --specsia_root {args.specsia_root}"
            )

    with open(os.path.join(splits_dir, "train_uids.json")) as f:
        train_uids = json.load(f)
    with open(os.path.join(splits_dir, "val_uids.json")) as f:
        val_uids = json.load(f)
    with open(os.path.join(splits_dir, "test_uids.json")) as f:
        test_uids = json.load(f)

    print(f"Split sizes: train={len(train_uids)}, val={len(val_uids)}, test={len(test_uids)}")

    if args.dry_run:
        print("\n[DRY RUN] Loading first 3 samples from train split ...")
        for uid in train_uids[:3]:
            s = load_sample(args.specsia_root, uid)
            if s:
                print(f"  {s['uid']}: color={s['color'].size}, gt={s['gt'].size}")
        print("[DRY RUN] Done. No upload performed.")
        return

    # ── Actual upload (parallel map — fast image loading, concurrent shard upload) ─
    from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage

    features = Features({
        "uid":   Value("string"),
        "color": HFImage(),
        "pos":   HFImage(),
        "edge":  HFImage(),
        "mask":  HFImage(),
        "gt":    HFImage(),
    })

    def build_split_parallel(uids, split_name, num_proc):
        print(f"\nBuilding {split_name} split ({len(uids)} samples, {num_proc} workers)...")
        uid_ds = Dataset.from_dict({"uid": uids})
        ds = uid_ds.map(
            _load_for_map,
            num_proc=num_proc,
            features=features,
            desc=f"Loading {split_name}",
        )
        # Drop rows where color is None (missing files)
        ds = ds.filter(lambda x: x["color"] is not None, num_proc=num_proc)
        print(f"  {split_name}: {len(ds)} samples ready")
        return ds

    ds = DatasetDict({
        "train":      build_split_parallel(train_uids, "train",      args.num_proc),
        "validation": build_split_parallel(val_uids,   "validation", args.num_proc),
        "test":       build_split_parallel(test_uids,  "test",       args.num_proc),
    })

    print(f"\nPushing to https://huggingface.co/datasets/{args.repo_id} ...")
    ds.push_to_hub(
        args.repo_id,
        private=args.private,
        commit_message="Upload SPECSIA-15K dataset",
        max_shard_size="500MB",
        num_shards={"train": 26, "validation": 2, "test": 2},
    )
    print("Upload complete.")


if __name__ == "__main__":
    main()

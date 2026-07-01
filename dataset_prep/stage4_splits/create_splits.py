"""
Create reproducible train / val / test splits for SPECSIA-15K.

Splits characters (not views) so that all 10 views of a character stay
in the same split — preventing GT-style leakage across splits.

Output (written to <specsia_root>/splits/):
    train_uids.json   — {char}_{view} pairs for training   (~12,980)
    val_uids.json     — {char}_{view} pairs for validation  (~1,000)
    test_uids.json    — {char}_{view} pairs for testing     (~1,000)
    split_chars.json  — {train:[...], val:[...], test:[...]} at char level

Split: 100 val chars, 100 test chars, remaining → train  (fixed seed=42)

Usage:
    python create_splits.py --specsia_root /path/to/SPECSIA-15k
"""

import os
import json
import random
import argparse


NUM_VAL  = 100
NUM_TEST = 100
SEED     = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specsia_root", required=True,
                        help="SPECSIA-15k root directory")
    parser.add_argument("--val",  type=int, default=NUM_VAL)
    parser.add_argument("--test", type=int, default=NUM_TEST)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    root = args.specsia_root
    splits_dir = os.path.join(root, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    # Discover all {char}_{view} UIDs present in the dataset
    import re
    pattern = re.compile(r"^(\d+)_(\d+)$")
    all_uid_views = sorted(
        [d for d in os.listdir(root)
         if pattern.match(d) and os.path.isdir(os.path.join(root, d))],
        key=lambda x: (int(x.split("_")[0]), int(x.split("_")[1]))
    )

    # Group by base char id
    char_to_views: dict[str, list[str]] = {}
    for uid in all_uid_views:
        char_id = uid.split("_")[0]
        char_to_views.setdefault(char_id, []).append(uid)

    all_chars = sorted(char_to_views.keys(), key=int)
    n_chars   = len(all_chars)
    print(f"Found {n_chars} characters, {len(all_uid_views)} total samples.")

    if n_chars < args.val + args.test + 1:
        raise ValueError(
            f"Not enough characters ({n_chars}) for val={args.val} + test={args.test}."
        )

    rng = random.Random(args.seed)
    shuffled = all_chars[:]
    rng.shuffle(shuffled)

    test_chars  = shuffled[:args.test]
    val_chars   = shuffled[args.test:args.test + args.val]
    train_chars = shuffled[args.test + args.val:]

    def expand(chars):
        uids = []
        for c in sorted(chars, key=int):
            uids.extend(sorted(char_to_views[c],
                               key=lambda x: int(x.split("_")[1])))
        return uids

    train_uids = expand(train_chars)
    val_uids   = expand(val_chars)
    test_uids  = expand(test_chars)

    print(f"  train: {len(train_chars)} chars, {len(train_uids)} samples")
    print(f"  val:   {len(val_chars)} chars, {len(val_uids)} samples")
    print(f"  test:  {len(test_chars)} chars, {len(test_uids)} samples")

    with open(os.path.join(splits_dir, "train_uids.json"), "w") as f:
        json.dump(train_uids, f, indent=2)
    with open(os.path.join(splits_dir, "val_uids.json"), "w") as f:
        json.dump(val_uids, f, indent=2)
    with open(os.path.join(splits_dir, "test_uids.json"), "w") as f:
        json.dump(test_uids, f, indent=2)
    with open(os.path.join(splits_dir, "split_chars.json"), "w") as f:
        json.dump({
            "seed": args.seed,
            "train": sorted(train_chars, key=int),
            "val":   sorted(val_chars,   key=int),
            "test":  sorted(test_chars,  key=int),
        }, f, indent=2)

    print(f"\nSplit files written to: {splits_dir}")


if __name__ == "__main__":
    main()

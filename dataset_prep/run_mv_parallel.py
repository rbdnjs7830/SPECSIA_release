"""
Parallel Wonder3D (mv.py) launcher for 3DBiCar.

Splits UIDs across GPUs and spawns one mv_worker.py per GPU.
Each worker loads the pipeline once and processes its share sequentially,
avoiding per-UID model reload overhead.

CUDA_VISIBLE_DEVICES remapping
-------------------------------
mv.py hardcodes cuda:0. Setting CUDA_VISIBLE_DEVICES={idx} remaps cuda:0
to the chosen physical GPU. Pass the CUDA ordinal indices (not nvidia-smi
indices, which may differ if GPU 0 is broken/absent).

Usage:
    python run_mv_parallel.py \\
        --dsu_dir  /path/to/DrawingSpinUp \\
        --data_dir /path/to/3dbicar/preprocessed \\
        [--gpus 0 1 2 3] [--resume]
"""

import os
import sys
import json
import argparse
import tempfile
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

WORKER = os.path.join(os.path.dirname(__file__), "mv_worker.py")


def already_done(uid: str, data_dir: str, num_views: int = 6) -> bool:
    color_dir = os.path.join(data_dir, uid, "mv", "color")
    if not os.path.isdir(color_dir):
        return False
    return len([f for f in os.listdir(color_dir) if f.endswith(".png")]) >= num_views


def run_worker(gpu: int, uid_list_path: str, dsu_dir: str, config: str, resume: bool) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    cmd = [sys.executable, WORKER, "--uid_list", uid_list_path, "--dsu_dir", dsu_dir]
    if config:
        cmd += ["--config", config]
    if resume:
        cmd.append("--resume")

    print(f"[GPU {gpu}] Starting worker ...", flush=True)
    mv_dir = os.path.join(dsu_dir, "2_charactor_reconstructor")
    proc = subprocess.run(cmd, env=env, cwd=mv_dir)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser("Parallel Wonder3D for 3DBiCar")
    parser.add_argument("--dsu_dir",  default=os.environ.get("DSU_DIR"),
                        help="DrawingSpinUp root (or set $DSU_DIR)")
    parser.add_argument("--data_dir", default=os.environ.get("BICAR_PREPROCESSED"),
                        help="3DBiCar preprocessed root (or set $BICAR_PREPROCESSED)")
    parser.add_argument("--uid_list", default=None,
                        help="JSON file listing _00 UIDs "
                             "(default: <data_dir>/base_uids_00.json)")
    parser.add_argument("--config",   default=None,
                        help="Override mv config yaml")
    parser.add_argument("--gpus",     nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--resume",   action="store_true",
                        help="Skip UIDs whose mv/ output already exists")
    args = parser.parse_args()

    if not args.dsu_dir:
        parser.error("--dsu_dir is required (or set $DSU_DIR)")
    if not args.data_dir:
        parser.error("--data_dir is required (or set $BICAR_PREPROCESSED)")

    uid_list_path = args.uid_list or os.path.join(args.data_dir, "base_uids_00.json")
    with open(uid_list_path) as f:
        all_uids = json.load(f)

    if args.resume:
        pending = [u for u in all_uids if not already_done(u, args.data_dir)]
        print(f"Resume: {len(all_uids) - len(pending)} done, {len(pending)} remaining")
    else:
        pending = all_uids

    if not pending:
        print("All UIDs already processed.")
        return

    gpus = args.gpus
    n    = len(gpus)
    splits = [pending[i::n] for i in range(n)]

    tmpdir = tempfile.mkdtemp(prefix="mv_split_")
    split_paths = []
    for gpu, uids in zip(gpus, splits):
        p = os.path.join(tmpdir, f"uids_gpu{gpu}.json")
        with open(p, "w") as f:
            json.dump(uids, f)
        split_paths.append(p)
        print(f"  GPU {gpu}: {len(uids)} UIDs")

    print(f"\nLaunching {n} workers across GPUs {gpus} ...")

    with ThreadPoolExecutor(max_workers=n) as exe:
        futs = {
            exe.submit(run_worker, gpu, path, args.dsu_dir, args.config, args.resume): gpu
            for gpu, path in zip(gpus, split_paths)
        }
        for fut in as_completed(futs):
            gpu = futs[fut]
            print(f"[GPU {gpu}] worker finished (exit={fut.result()})", flush=True)

    shutil.rmtree(tmpdir, ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()

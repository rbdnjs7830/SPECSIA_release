"""
Parallel instant-nsr (recon.py) launcher for 3DBiCar.

Distributes _00 UIDs across GPUs. Each GPU runs recon.py sequentially
(one UID at a time) to avoid VRAM contention from the hash-grid NeRF.

Prerequisites per UID: mv/ output must exist (run run_mv_parallel.py first).

Usage:
    python run_recon_parallel.py \\
        --dsu_dir  /path/to/DrawingSpinUp \\
        --data_dir /path/to/3dbicar/preprocessed \\
        [--gpus 0 1 2 3] [--resume]
"""

import os
import sys
import json
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


def already_done(uid: str, data_dir: str) -> bool:
    mesh_dir = os.path.join(data_dir, uid, "mesh")
    if not os.path.isdir(mesh_dir):
        return False
    return any(f.endswith(".obj") for f in os.listdir(mesh_dir))


def mv_exists(uid: str, data_dir: str, num_views: int = 6) -> bool:
    color_dir = os.path.join(data_dir, uid, "mv", "color")
    if not os.path.isdir(color_dir):
        return False
    return len([f for f in os.listdir(color_dir) if f.endswith(".png")]) >= num_views


def run_recon_on_gpu(gpu: int, uids: list, dsu_dir: str, data_dir: str,
                     config: str, resume: bool) -> None:
    recon_dir = os.path.join(dsu_dir, "2_charactor_reconstructor")
    recon_py  = os.path.join(recon_dir, "recon.py")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"[GPU {gpu}] Processing {len(uids)} UIDs ...", flush=True)
    for i, uid in enumerate(uids):
        if resume and already_done(uid, data_dir):
            print(f"[GPU {gpu}] [{i+1}/{len(uids)}] skip {uid} (done)", flush=True)
            continue
        if not mv_exists(uid, data_dir):
            print(f"[GPU {gpu}] [{i+1}/{len(uids)}] skip {uid} (no mv output)", flush=True)
            continue

        cmd = [sys.executable, recon_py, "--uid", uid, "--config", config]
        print(f"[GPU {gpu}] [{i+1}/{len(uids)}] recon {uid}", flush=True)
        proc = subprocess.run(cmd, env=env, cwd=recon_dir)
        status = "OK" if proc.returncode == 0 else f"FAIL(exit={proc.returncode})"
        print(f"[GPU {gpu}] [{i+1}/{len(uids)}] {status} {uid}", flush=True)

    print(f"[GPU {gpu}] Done.", flush=True)


def main():
    parser = argparse.ArgumentParser("Parallel instant-nsr for 3DBiCar")
    parser.add_argument("--dsu_dir",  default=os.environ.get("DSU_DIR"),
                        help="DrawingSpinUp root (or set $DSU_DIR)")
    parser.add_argument("--data_dir", default=os.environ.get("BICAR_PREPROCESSED"),
                        help="3DBiCar preprocessed root (or set $BICAR_PREPROCESSED)")
    parser.add_argument("--uid_list", default=None,
                        help="JSON file listing _00 UIDs "
                             "(default: <data_dir>/base_uids_00.json)")
    parser.add_argument("--config",   default=None,
                        help="Override recon config yaml")
    parser.add_argument("--gpus",     nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--resume",   action="store_true",
                        help="Skip UIDs whose mesh/*.obj already exists")
    args = parser.parse_args()

    if not args.dsu_dir:
        parser.error("--dsu_dir is required (or set $DSU_DIR)")
    if not args.data_dir:
        parser.error("--data_dir is required (or set $BICAR_PREPROCESSED)")

    recon_dir = os.path.join(args.dsu_dir, "2_charactor_reconstructor")
    config    = args.config or os.path.join(
        recon_dir, "configs", "neuralangelo-ortho-wmask-3dbicar.yaml"
    )
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

    gpus   = args.gpus
    n      = len(gpus)
    splits = [pending[i::n] for i in range(n)]

    for gpu, uids in zip(gpus, splits):
        print(f"  GPU {gpu}: {len(uids)} UIDs")
    print(f"\nLaunching {n} workers across GPUs {gpus} ...")

    with ThreadPoolExecutor(max_workers=n) as exe:
        futs = {
            exe.submit(run_recon_on_gpu, gpu, uids, args.dsu_dir, args.data_dir,
                       config, args.resume): gpu
            for gpu, uids in zip(gpus, splits)
        }
        for fut in as_completed(futs):
            gpu = futs[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"[GPU {gpu}] ERROR: {e}", flush=True)

    print("All done.")


if __name__ == "__main__":
    main()

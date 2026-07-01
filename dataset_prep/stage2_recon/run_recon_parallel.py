"""
Parallel instant-nsr (recon.py) launcher for 3DBiCar.

Distributes _00 UIDs across GPUs. Each GPU runs recon.py sequentially
(one UID at a time) to avoid VRAM contention from the hash-grid NeRF.

Prerequisites per UID: mv/ output must exist (run run_mv_parallel.py first).

Usage:
    python run_recon_parallel.py \\
        --data_dir /path/to/3dbicar/preprocessed \\
        [--gpus 0 1 2 3] [--resume]
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed


def _default_dsu_dir() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return (
        os.environ.get("DSU_DIR")
        or os.path.join(repo_root, "external", "DrawingSpinUp")
    )


def _make_recon_config(template_path: str, data_dir: str, uid_list_path: str) -> str:
    """
    Write a temp YAML with data_root, uid_list_file, thinning_uid_list_file
    substituted in.  recon.py doesn't expose CLI config overrides, so we patch
    the YAML before passing it.

    Uses raw string replacement (not yaml.load/dump) to preserve OmegaConf
    interpolation syntax like ${calc_exp_lr_decay_rate:...} that yaml.dump
    would corrupt by adding quotes.

    Returns the path to the temp file (caller must delete it).
    """
    with open(template_path) as f:
        raw = f.read()

    thinning_path = os.path.join(data_dir, "thinning_uids.json")
    if not os.path.exists(thinning_path):
        with open(thinning_path, "w") as f:
            json.dump([], f)

    raw = raw.replace("data_root: PLACEHOLDER",
                      f"data_root: '{data_dir}'")
    raw = raw.replace("uid_list_file: PLACEHOLDER",
                      f"uid_list_file: '{uid_list_path}'")
    raw = raw.replace("thinning_uid_list_file: PLACEHOLDER",
                      f"thinning_uid_list_file: '{thinning_path}'")

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        prefix="recon_3dbicar_"
    )
    tmp.write(raw)
    tmp.close()
    return tmp.name


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
    parser.add_argument("--dsu_dir",  default=_default_dsu_dir(),
                        help="DrawingSpinUp root (default: external/DrawingSpinUp or $DSU_DIR)")
    parser.add_argument("--data_dir", default=os.environ.get("BICAR_PREPROCESSED"),
                        help="3DBiCar preprocessed root (or set $BICAR_PREPROCESSED)")
    parser.add_argument("--uid_list", default=None,
                        help="JSON file listing _00 UIDs "
                             "(default: <data_dir>/base_uids_00.json)")
    parser.add_argument("--config",   default=None,
                        help="Recon config YAML template (default: dataset_prep/configs/neuralangelo-3dbicar.yaml). "
                             "data_root, uid_list_file, thinning_uid_list_file are injected at runtime.")
    parser.add_argument("--gpus",     nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--resume",   action="store_true",
                        help="Skip UIDs whose mesh/*.obj already exists")
    args = parser.parse_args()

    if not args.dsu_dir or not os.path.isdir(args.dsu_dir):
        parser.error(
            f"DrawingSpinUp not found at '{args.dsu_dir}'. "
            "Run `bash setup.sh` to initialize the submodule, "
            "or set --dsu_dir / $DSU_DIR."
        )
    if not args.data_dir:
        parser.error("--data_dir is required (or set $BICAR_PREPROCESSED)")

    uid_list_path = args.uid_list or os.path.join(args.data_dir, "base_uids_00.json")

    # Default config template: dataset_prep/configs/neuralangelo-3dbicar.yaml
    template = args.config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "configs", "neuralangelo-3dbicar.yaml"
    )
    if not os.path.exists(template):
        parser.error(f"Config template not found: {template}")

    # Inject data paths into the config (recon.py has no CLI override mechanism)
    tmp_config = _make_recon_config(template, args.data_dir, uid_list_path)

    with open(uid_list_path) as f:
        all_uids = json.load(f)

    if args.resume:
        pending = [u for u in all_uids if not already_done(u, args.data_dir)]
        print(f"Resume: {len(all_uids) - len(pending)} done, {len(pending)} remaining")
    else:
        pending = all_uids

    if not pending:
        print("All UIDs already processed.")
        os.unlink(tmp_config)
        return

    gpus   = args.gpus
    n      = len(gpus)
    splits = [pending[i::n] for i in range(n)]

    for gpu, uids in zip(gpus, splits):
        print(f"  GPU {gpu}: {len(uids)} UIDs")
    print(f"\nLaunching {n} workers across GPUs {gpus} ...")

    try:
        with ThreadPoolExecutor(max_workers=n) as exe:
            futs = {
                exe.submit(run_recon_on_gpu, gpu, uids, args.dsu_dir, args.data_dir,
                           tmp_config, args.resume): gpu
                for gpu, uids in zip(gpus, splits)
            }
            for fut in as_completed(futs):
                gpu = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"[GPU {gpu}] ERROR: {e}", flush=True)
    finally:
        os.unlink(tmp_config)

    print("All done.")


if __name__ == "__main__":
    main()

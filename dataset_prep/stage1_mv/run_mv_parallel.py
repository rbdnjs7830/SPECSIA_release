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


def _default_dsu_dir() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return (
        os.environ.get("DSU_DIR")
        or os.path.join(repo_root, "external", "DrawingSpinUp")
    )


def _make_mv_config(template_path: str, data_dir: str, uid_list_path: str) -> str:
    """Inject data_root and uid_list_file into the mv config template.
    Uses string replacement to avoid corrupting YAML structure.
    Returns path to temp file (caller must delete it).
    """
    with open(template_path) as f:
        raw = f.read()
    raw = raw.replace("data_root: PLACEHOLDER", f"data_root: '{data_dir}'")
    raw = raw.replace("uid_list_file: PLACEHOLDER", f"uid_list_file: '{uid_list_path}'")
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="mv_3dbicar_"
    )
    tmp.write(raw)
    tmp.close()
    return tmp.name


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
    parser.add_argument("--dsu_dir",  default=_default_dsu_dir(),
                        help="DrawingSpinUp root (default: external/DrawingSpinUp or $DSU_DIR)")
    parser.add_argument("--data_dir", default=os.environ.get("BICAR_PREPROCESSED"),
                        help="3DBiCar preprocessed root (or set $BICAR_PREPROCESSED)")
    parser.add_argument("--uid_list", default=None,
                        help="JSON file listing _00 UIDs "
                             "(default: <data_dir>/base_uids_00.json)")
    parser.add_argument("--config",   default=None,
                        help="mv config YAML template (default: dataset_prep/configs/mvdiffusion-3dbicar.yaml). "
                             "data_root and uid_list_file are injected at runtime.")
    parser.add_argument("--gpus",     nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--resume",   action="store_true",
                        help="Skip UIDs whose mv/ output already exists")
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

    # Resolve and inject paths into the mv config template
    template = args.config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "configs", "mvdiffusion-3dbicar.yaml"
    )
    if not os.path.exists(template):
        parser.error(f"mv config template not found: {template}")
    tmp_config = _make_mv_config(template, args.data_dir, uid_list_path)

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

    try:
        with ThreadPoolExecutor(max_workers=n) as exe:
            futs = {
                exe.submit(run_worker, gpu, path, args.dsu_dir, tmp_config, args.resume): gpu
                for gpu, path in zip(gpus, split_paths)
            }
            for fut in as_completed(futs):
                gpu = futs[fut]
                print(f"[GPU {gpu}] worker finished (exit={fut.result()})", flush=True)
    finally:
        os.unlink(tmp_config)
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("Done.")


if __name__ == "__main__":
    main()

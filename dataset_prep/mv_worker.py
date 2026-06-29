"""
Wonder3D worker: loads the pipeline ONCE per process and processes all assigned UIDs.
Spawned by run_mv_parallel.py with CUDA_VISIBLE_DEVICES set externally.

The worker stays alive for all assigned UIDs (pipeline loaded once, no per-UID overhead).
mv.py accesses a global `args` namespace set before __main__, so we inject a fake one.
"""

import sys
import os
import argparse
import json
import types


def already_done(uid: str, data_root: str, num_views: int = 6) -> bool:
    color_dir = os.path.join(data_root, uid, "mv", "color")
    if not os.path.isdir(color_dir):
        return False
    return len([f for f in os.listdir(color_dir) if f.endswith(".png")]) >= num_views


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uid_list", required=True,
                        help="JSON file listing UIDs assigned to this worker")
    parser.add_argument("--dsu_dir",  required=True,
                        help="DrawingSpinUp root directory")
    parser.add_argument("--config",   default=None,
                        help="mv config yaml (default: <dsu_dir>/2_charactor_reconstructor/"
                             "configs/mvdiffusion-joint-ortho-6views-3dbicar.yaml)")
    parser.add_argument("--img_fn",   default="char/ffc_resnet_inpainted.png",
                        help="Input image filename relative to each UID dir "
                             "(falls back to char/texture.png inside mv.py)")
    parser.add_argument("--resume",   action="store_true")
    cli = parser.parse_args()

    mv_dir = os.path.join(cli.dsu_dir, "2_charactor_reconstructor")
    config_path = cli.config or os.path.join(
        mv_dir, "configs", "mvdiffusion-joint-ortho-6views-3dbicar.yaml"
    )

    sys.path.insert(0, mv_dir)
    os.chdir(mv_dir)  # mv.py resolves model weights via relative paths

    # mv.py reads `args` (global argparse namespace) inside mv().
    # Inject a fake namespace before import so the module picks it up.
    fake_args = types.SimpleNamespace(
        config=config_path, uid="", img_fn=cli.img_fn, save_folder="mv", all=False,
    )
    import mv as mv_module
    mv_module.args = fake_args

    from mv import load_config, load_wonder3d_pipeline, mv as run_mv
    import torch

    config = load_config(config_path)

    with open(cli.uid_list) as f:
        uid_list = json.load(f)

    if cli.resume:
        uid_list = [u for u in uid_list if not already_done(u, config.data_root)]
        print(f"[worker] resume: {len(uid_list)} UIDs remaining", flush=True)

    device = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"[worker] Loading pipeline (CUDA_VISIBLE_DEVICES={device}) ...", flush=True)
    pipeline = load_wonder3d_pipeline(config)
    torch.set_grad_enabled(False)
    pipeline.to("cuda:0")
    print(f"[worker] Pipeline ready. Processing {len(uid_list)} UIDs ...", flush=True)

    for i, uid in enumerate(uid_list):
        mv_module.args.img_fn = cli.img_fn
        mv_module.args.save_folder = "mv"
        try:
            run_mv(uid, pipeline, config)
            print(f"[{i+1}/{len(uid_list)}] OK  {uid}", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(uid_list)}] ERR {uid}: {e}", flush=True)


if __name__ == "__main__":
    main()

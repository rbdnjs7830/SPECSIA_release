#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bicar_multiview_outline.py
- Single-file launcher + Blender render script
- Per object: multi-view PNGs with outline overlaid on RGBA (transparent background)
- No separate contour-only outputs.

Run (launcher mode; normal python):
 python bicar_multiview_render.py \
    --input_models_path /mnt/hdd/kyuwon/3DBiCar_test/bicar_uids.json \
    --bicar_root /mnt/hdd/kyuwon/3DBiCar_test/raw \
    --save_folder /mnt/hdd/kyuwon/3DBiCar_test/render \
    --blender_install_path /mnt/hdd/kyuwon/DrawingSpinUp_kyu/blender-3.6.14-linux-x64/blender \
    --start_i 0 --end_i 1500 \
    --ortho_scale 1.35 \
    --num_views 10 --yaw_start 0 --yaw_end 360\
    --resolution 512

Optional:
  --random_pose
  --outline_thickness 2.0
  --outline_rgba 0,0,0,1   (black)
"""

import os
import sys
import json
import math
import argparse
import subprocess
import time


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _find_blender() -> str:
    """Return path to blender: $BLENDER env → repo-bundled → PATH."""
    from_env = os.environ.get("BLENDER")
    if from_env and os.path.isfile(from_env) and os.access(from_env, os.X_OK):
        return from_env
    bundled = os.path.join(_repo_root(), "blender-3.6.14-linux-x64", "blender")
    if os.path.isfile(bundled) and os.access(bundled, os.X_OK):
        return bundled
    import shutil
    which = shutil.which("blender")
    if which:
        return which
    raise RuntimeError(
        "Blender not found. Run `source dataset_prep/setup.sh` to download it, "
        "or set the $BLENDER environment variable."
    )


# ------------------------------------------------------------
# Detect Blender environment
# ------------------------------------------------------------
IN_BLENDER = False
try:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore
    import numpy as np  # type: ignore
    IN_BLENDER = True
except Exception:
    IN_BLENDER = False


# ============================================================
# Launcher mode (normal python)
# ============================================================

def launcher_main():
    parser = argparse.ArgumentParser(description="3DBiCar multi-view outline renderer (single-file).")

    parser.add_argument('--input_models_path', type=str, default='../../dataset/3DBiCar/bicar_uids.json')
    parser.add_argument('--start_i', type=int, default=0)
    parser.add_argument('--end_i', type=int, default=1500)

    parser.add_argument('--bicar_root', type=str, default='../../dataset/3DBiCar/raw')
    parser.add_argument('--save_folder', type=str, default='../../dataset/3DBiCar/multiview_outline')

    parser.add_argument('--blender_install_path', type=str, default=None,
                        help="Path to blender executable (default: auto-detect via $BLENDER or repo bundle)")

    parser.add_argument('--ortho_scale', type=float, default=1.35)
    parser.add_argument('--resolution', type=int, default=512)

    # Multi-view
    parser.add_argument('--num_views', type=int, default=10)
    parser.add_argument('--yaw_start', type=float, default=0.0, help='degrees')
    parser.add_argument('--yaw_end', type=float, default=360.0, help='degrees')

    # Outline style
    parser.add_argument('--outline_thickness', type=float, default=2.0)
    parser.add_argument('--outline_rgba', type=str, default='0,0,0,1', help='e.g., 0,0,0,1 for black')

    parser.add_argument('--random_pose', action='store_true')

    # GPU env
    parser.add_argument('--cuda_visible_devices', type=str, default='0')

    args = parser.parse_args()

    with open(args.input_models_path, "r") as f:
        model_paths = json.load(f)

    start_i = max(0, args.start_i)
    end_i = len(model_paths) if args.end_i > len(model_paths) else args.end_i

    script_path = os.path.abspath(__file__)
    blender_path = os.path.abspath(args.blender_install_path or _find_blender())
    save_root = os.path.abspath(args.save_folder)
    bicar_root = os.path.abspath(args.bicar_root)
    os.makedirs(save_root, exist_ok=True)

    for item in model_paths[start_i:end_i]:
        obj_path = os.path.join(bicar_root, item)
        if not os.path.exists(obj_path):
            print(f"[WARN] object_path not found, skip: {obj_path}")
            continue

        cmd = [
            blender_path,
            "--background",
            "-E", "CYCLES",
            "--python", script_path,
            "--",
            "--mode", "render",
            "--object_path", obj_path,
            "--output_folder", save_root,
            "--resolution", str(args.resolution),
            "--ortho_scale", str(args.ortho_scale),
            "--num_views", str(args.num_views),
            "--yaw_start", str(args.yaw_start),
            "--yaw_end", str(args.yaw_end),
            "--outline_thickness", str(args.outline_thickness),
            "--outline_rgba", args.outline_rgba,
        ]
        if args.random_pose:
            cmd.append("--random_pose")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

        print(f"\n[LAUNCH] {obj_path}")
        subprocess.run(cmd, env=env, check=False)


# ============================================================
# Blender render mode (inside Blender)
# ============================================================

if IN_BLENDER:
    def scene_root_objects():
        for obj in bpy.context.scene.objects.values():
            if not obj.parent:
                yield obj

    def scene_meshes():
        for obj in bpy.context.scene.objects.values():
            if isinstance(obj.data, (bpy.types.Mesh)):
                yield obj

    def scene_bbox(single_obj=None, ignore_matrix=False):
        bbox_min = (math.inf,) * 3
        bbox_max = (-math.inf,) * 3
        found = False
        for obj in scene_meshes() if single_obj is None else [single_obj]:
            found = True
            for coord in obj.bound_box:
                coord = Vector(coord)
                if not ignore_matrix:
                    coord = obj.matrix_world @ coord
                bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
        if not found:
            raise RuntimeError("no objects in scene to compute bounding box for")
        return Vector(bbox_min), Vector(bbox_max)

    def normalize_scene():
        bbox_min, bbox_max = scene_bbox()
        scale = 1 / max(bbox_max - bbox_min)
        for obj in scene_root_objects():
            obj.scale = obj.scale * scale
        bpy.context.view_layer.update()
        bbox_min, bbox_max = scene_bbox()
        offset = -(bbox_min + bbox_max) / 2
        for obj in scene_root_objects():
            obj.matrix_world.translation += offset
        bpy.ops.object.select_all(action="DESELECT")

    def reset_scene():
        for obj in list(bpy.data.objects):
            if obj.type not in {"CAMERA", "LIGHT"}:
                bpy.data.objects.remove(obj, do_unlink=True)
        for material in list(bpy.data.materials):
            bpy.data.materials.remove(material, do_unlink=True)
        for texture in list(bpy.data.textures):
            bpy.data.textures.remove(texture, do_unlink=True)
        for image in list(bpy.data.images):
            bpy.data.images.remove(image, do_unlink=True)

        # world background (doesn't matter much since film_transparent=True)
        try:
            world_tree = bpy.context.scene.world.node_tree
            back_node = world_tree.nodes['Background']
            back_node.inputs['Color'].default_value = Vector([1.0, 1.0, 1.0, 1.0])
            back_node.inputs['Strength'].default_value = 1.0
        except Exception:
            pass

    def load_object(object_path):
        """
        Expects:
          object_path/tpose/m.obj, e.obj and m.bmp, e.bmp (case issues handled)
        """
        m_bmp = os.path.join(object_path, 'tpose', 'm.bmp')
        if not os.path.exists(m_bmp):
            alt = m_bmp.replace('bmp', 'BMP')
            if os.path.exists(alt):
                os.rename(alt, m_bmp)

        e_bmp = os.path.join(object_path, 'tpose', 'e.bmp')
        if not os.path.exists(e_bmp):
            alt = e_bmp.replace('bmp', 'BMP')
            if os.path.exists(alt):
                os.rename(alt, e_bmp)

        m_obj = os.path.join(object_path, 'tpose', 'm.obj')
        e_obj = os.path.join(object_path, 'tpose', 'e.obj')
        if not (os.path.exists(m_obj) and os.path.exists(e_obj)):
            raise FileNotFoundError(f"Missing {m_obj} or {e_obj}")

        bpy.ops.wm.obj_import(filepath=m_obj)
        bpy.ops.wm.obj_import(filepath=e_obj)

    def get_camera_pose_from_location(loc):
        location = Vector([loc[0], loc[1], loc[2]])
        direction = -location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        rotation_euler = rot_quat.to_euler()
        return location, rotation_euler

    def parse_args_after_double_dash():
        argv = sys.argv
        if "--" not in argv:
            return []
        return argv[argv.index("--") + 1:]

    def parse_rgba_list(s):
        # "r,g,b,a" -> tuple(float)
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 4:
            raise ValueError("outline_rgba must be 'r,g,b,a' (4 values)")
        return tuple(float(x) for x in parts)

    def setup_outline_freestyle(thickness, rgba):
        """
        Freestyle로 외곽선을 최종 렌더에 오버레이.
        contour-only 파일 저장은 안 함.
        """
        scene = bpy.data.scenes['Scene']
        scene.render.use_freestyle = True

        # Freestyle settings
        freestyle_settings = scene.view_layers['ViewLayer'].freestyle_settings

        # Ensure a lineset exists
        if len(freestyle_settings.linesets) == 0:
            lineset = freestyle_settings.linesets.new("LineSet")
        else:
            # use first
            lineset = freestyle_settings.linesets[0]

        # select external contour only (너가 원하던 'front_contour' 느낌)
        lineset.select_silhouette = False
        lineset.select_crease = False
        lineset.select_border = False
        lineset.select_external_contour = True

        # Ensure linestyle exists
        if lineset.linestyle is None:
            # creating a linestyle through operator tends to create one automatically,
            # but we handle it robustly
            try:
                bpy.ops.scene.freestyle_linestyle_new()
                lineset.linestyle = freestyle_settings.linesets[0].linestyle
            except Exception:
                pass

        linestyle = lineset.linestyle
        # Style
        linestyle.thickness = float(thickness)
        # linestyle.thickness_position = 'INSIDE'
        linestyle.thickness_position = 'CENTER'
        linestyle.caps = 'ROUND'
        linestyle.chaining = 'PLAIN'  # 안정적
        try:
            linestyle.color = rgba  # (r,g,b,a)
        except Exception:
            # 일부 버전에선 color가 없을 수 있어 fallback
            pass

    def blender_render_main():
        parser = argparse.ArgumentParser(description="Blender internal render mode.")
        parser.add_argument("--mode", type=str, default="render")
        parser.add_argument("--object_path", type=str, required=True)
        parser.add_argument("--output_folder", type=str, required=True)
        parser.add_argument("--resolution", type=int, default=512)
        parser.add_argument("--ortho_scale", type=float, default=1.25)

        parser.add_argument("--num_views", type=int, default=10)
        parser.add_argument("--yaw_start", type=float, default=0.0)
        parser.add_argument("--yaw_end", type=float, default=360.0)

        parser.add_argument("--outline_thickness", type=float, default=2.0)
        parser.add_argument("--outline_rgba", type=str, default="0,0,0,1")

        parser.add_argument("--random_pose", action="store_true")

        args = parser.parse_args(parse_args_after_double_dash())

        t0 = time.time()
        reset_scene()

        object_dir = args.object_path
        object_uid = os.path.basename(os.path.normpath(object_dir))
        out_root = os.path.abspath(args.output_folder)
        out_dir = os.path.join(out_root, object_uid)
        os.makedirs(out_dir, exist_ok=True)

        load_object(object_dir)
        normalize_scene()

        # Create empty at origin
        cam_empty = bpy.data.objects.new("Empty", None)
        cam_empty.location = (0, 0, 0)
        bpy.context.scene.collection.objects.link(cam_empty)

        # Add camera at front
        cam_loc = np.array([0, -2.0, 0])
        _loc, _rot = get_camera_pose_from_location(cam_loc)
        bpy.ops.object.camera_add(location=_loc, rotation=_rot)
        cam = bpy.context.selected_objects[0]
        bpy.context.scene.camera = cam

        # Track-to origin
        c = cam.constraints.new(type='TRACK_TO')
        c.track_axis = 'TRACK_NEGATIVE_Z'
        c.up_axis = 'UP_Y'
        c.target = cam_empty
        c.owner_space = 'LOCAL'

        cam.parent = cam_empty

        # Ortho camera
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = args.ortho_scale

        # Render settings
        scene = bpy.context.scene
        scene.render.resolution_x = args.resolution
        scene.render.resolution_y = args.resolution
        scene.render.resolution_percentage = 100

        scene.render.film_transparent = True
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'

        # Outline
        rgba = parse_rgba_list(args.outline_rgba)
        setup_outline_freestyle(args.outline_thickness, rgba)

        bpy.context.view_layer.update()

        # Base random pose offsets (shared across views)
        if args.random_pose:
            base_z = float(np.random.uniform(-45, 45, 1))  # yaw
            base_x = float(np.random.uniform(-15, 15, 1))  # pitch
            base_y = 0.0
        else:
            base_z = base_x = base_y = 0.0

        yaw0 = args.yaw_start
        yaw1 = args.yaw_end

        for vid in range(args.num_views):
            # evenly spaced, avoid duplicating 0 & 360
            t = vid / args.num_views
            yaw_deg = yaw0 + (yaw1 - yaw0) * t

            cam_empty.rotation_euler = (
                math.radians(base_x),
                math.radians(base_y),
                math.radians(base_z + yaw_deg),
            )
            bpy.context.view_layer.update()

            # Output (single combined image)
            scene.render.filepath = os.path.join(out_dir, f"view_{vid:02d}_rgba_outline")
            bpy.ops.render.render(write_still=True)

        print(f"[BLENDER] Finished {object_dir} in {time.time() - t0:.3f}s")


# ============================================================
# Entrypoint
# ============================================================

def main():
    if IN_BLENDER:
        argv_after = []
        try:
            idx = sys.argv.index("--")
            argv_after = sys.argv[idx + 1:]
        except ValueError:
            argv_after = []

        if "--mode" in argv_after:
            blender_render_main()
    else:
        launcher_main()

if __name__ == "__main__":
    main()

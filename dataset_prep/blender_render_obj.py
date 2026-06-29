"""
Blender script: render color + pos (or mask) from a reconstructed OBJ.
No FBX / armature required — works for static rest-pose meshes.

Usage (called from run_render_obj_parallel.py via subprocess):
    blender -b config_ortho.blend -E BLENDER_EEVEE --python blender_render_obj.py -- \
        --obj_file /path/to/mesh.obj \
        --output_dir /path/to/out \
        --yaw_deg 36.0 \
        [--render_mask_only] [--loc_x 0.05] [--loc_z -0.05] [--res 256]
"""

import bpy
import os
import sys
import argparse

import numpy as np
import trimesh
from mathutils import Vector


# ---------------------------------------------------------------------------
# Render settings
# ---------------------------------------------------------------------------

def configure_render(res: int = 512):
    scene = bpy.context.scene
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    try:
        scene.view_settings.look = "None"
    except Exception:
        pass
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.resolution_percentage = 100


def render_to(output_path: str):
    os.makedirs(output_path, exist_ok=True)
    bpy.context.scene.render.filepath = output_path + "/"
    bpy.ops.render.render(animation=True)


# ---------------------------------------------------------------------------
# OBJ import
# ---------------------------------------------------------------------------

def import_obj(obj_file: str):
    before = set(bpy.data.objects.keys())
    try:
        bpy.ops.wm.obj_import(filepath=obj_file)      # Blender 3.3+
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=obj_file)   # Blender 2.x
    return [o for o in bpy.data.objects if o.name not in before]


def find_mesh_objects(imported, min_verts: int = 20):
    objs = [o for o in imported if o.type == "MESH" and len(o.data.vertices) >= min_verts]
    return sorted(objs, key=lambda o: len(o.data.vertices), reverse=True)


# ---------------------------------------------------------------------------
# Vertex color helpers
# ---------------------------------------------------------------------------

def ensure_vc_layer(mesh, name: str):
    layer = mesh.vertex_colors.get(name)
    if layer is None:
        layer = mesh.vertex_colors.new(name=name)
    return layer


def make_emission_mat(layer_name: str):
    mat = bpy.data.materials.new(name=f"mat_{layer_name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    vcol = nodes.new(type="ShaderNodeVertexColor")
    vcol.layer_name = layer_name
    emis = nodes.new(type="ShaderNodeEmission")
    out  = nodes.new(type="ShaderNodeOutputMaterial")
    links.new(vcol.outputs["Color"], emis.inputs["Color"])
    links.new(emis.outputs["Emission"], out.inputs["Surface"])
    return mat


def force_material(mesh_objs, mat):
    for obj in mesh_objs:
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        for poly in obj.data.polygons:
            poly.material_index = 0


# ---------------------------------------------------------------------------
# Color: read from trimesh (vertex colors or UV texture, fallback to pos)
# ---------------------------------------------------------------------------

def get_colors_from_trimesh(tri) -> np.ndarray:
    """Return float32 [N_verts, 3] RGB colors in [0, 1]."""
    visual = tri.visual

    if hasattr(visual, "vertex_colors") and visual.vertex_colors is not None:
        c = np.asarray(visual.vertex_colors, dtype=np.float32)
        return c[:, :3] / 255.0

    if hasattr(visual, "uv") and visual.uv is not None:
        mat = getattr(visual, "material", None)
        img = getattr(mat, "image", None) if mat is not None else None
        if img is not None:
            tex = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
            h, w = tex.shape[:2]
            uv = np.asarray(visual.uv, dtype=np.float32)
            px = np.clip(np.round(uv[:, 0] * (w - 1)).astype(int), 0, w - 1)
            py = np.clip(np.round((1.0 - uv[:, 1]) * (h - 1)).astype(int), 0, h - 1)
            return tex[py, px]

    # fallback: normalized XYZ pseudo-color
    v = np.asarray(tri.vertices, dtype=np.float32)
    return np.clip((v - v.min(0)) / np.maximum(v.max(0) - v.min(0), 1e-8), 0, 1)


def assign_color_vc(mesh_objs, tri):
    src = get_colors_from_trimesh(tri)
    n = len(src)
    for obj in mesh_objs:
        mesh = obj.data
        layer = ensure_vc_layer(mesh, "Color_VC")
        for poly in mesh.polygons:
            for loop_idx in poly.loop_indices:
                vid = mesh.loops[loop_idx].vertex_index
                c = src[vid] if vid < n else np.zeros(3, dtype=np.float32)
                layer.data[loop_idx].color = (float(c[0]), float(c[1]), float(c[2]), 1.0)


# ---------------------------------------------------------------------------
# Pos: normalized world-space XYZ
# ---------------------------------------------------------------------------

def assign_pos_vc(mesh_objs):
    all_pts = []
    for obj in mesh_objs:
        for v in obj.data.vertices:
            p = obj.matrix_world @ v.co
            all_pts.append((p.x, p.y, p.z))
    all_pts = np.array(all_pts, dtype=np.float32)
    v_min = all_pts.min(0)
    v_max = all_pts.max(0)
    denom = np.maximum(v_max - v_min, 1e-8)

    for obj in mesh_objs:
        mesh = obj.data
        layer = ensure_vc_layer(mesh, "POS_VC")
        for poly in mesh.polygons:
            for loop_idx in poly.loop_indices:
                vid = mesh.loops[loop_idx].vertex_index
                p = obj.matrix_world @ mesh.vertices[vid].co
                c = np.clip((np.array([p.x, p.y, p.z], dtype=np.float32) - v_min) / denom, 0, 1)
                layer.data[loop_idx].color = (float(c[0]), float(c[1]), float(c[2]), 1.0)


# ---------------------------------------------------------------------------
# Mask (silhouette)
# ---------------------------------------------------------------------------

def make_silhouette_mat():
    mat = bpy.data.materials.new("SilhouetteMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out  = nodes.new(type="ShaderNodeOutputMaterial")
    emis = nodes.new(type="ShaderNodeEmission")
    emis.inputs["Color"].default_value    = (1, 1, 1, 1)
    emis.inputs["Strength"].default_value = 1.0
    links.new(emis.outputs["Emission"], out.inputs["Surface"])
    return mat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    try:
        idx = sys.argv.index("--")
        args_list = sys.argv[idx + 1:]
    except ValueError:
        args_list = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_file",         required=True,  type=str)
    parser.add_argument("--output_dir",       required=True,  type=str)
    parser.add_argument("--yaw_deg",          default=0.0,    type=float)
    parser.add_argument("--render_mask_only", action="store_true")
    parser.add_argument("--loc_x",            default=0.0,    type=float)
    parser.add_argument("--loc_y",            default=0.0,    type=float)
    parser.add_argument("--loc_z",            default=0.0,    type=float)
    parser.add_argument("--pitch_deg",        default=0.0,    type=float)
    parser.add_argument("--res",              default=512,    type=int)
    args = parser.parse_args(args_list)

    # import
    imported   = import_obj(args.obj_file)
    mesh_objs  = find_mesh_objects(imported)
    if not mesh_objs:
        raise RuntimeError(f"No mesh found in {args.obj_file}")

    # apply transform (yaw + pitch + location offset)
    for obj in mesh_objs:
        obj.delta_rotation_euler[0] = np.radians(float(args.pitch_deg))
        obj.delta_rotation_euler[2] = np.radians(-float(args.yaw_deg))
        obj.delta_location[0] += float(args.loc_x)
        obj.delta_location[1] += float(args.loc_y)
        obj.delta_location[2] += float(args.loc_z)

    bpy.context.view_layer.update()

    # single-frame render
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end   = 1
    scene.frame_set(1)

    configure_render(args.res)

    # --- mask only (for alignment search) ---
    if args.render_mask_only:
        force_material(mesh_objs, make_silhouette_mat())
        render_to(os.path.join(args.output_dir, "mask_render"))
        return

    # --- color ---
    tri = trimesh.load_mesh(args.obj_file, process=False)
    if isinstance(tri, trimesh.Scene):
        parts = list(tri.geometry.values())
        tri = trimesh.util.concatenate(parts)

    assign_color_vc(mesh_objs, tri)
    force_material(mesh_objs, make_emission_mat("Color_VC"))
    render_to(os.path.join(args.output_dir, "color"))

    # --- pos ---
    assign_pos_vc(mesh_objs)
    force_material(mesh_objs, make_emission_mat("POS_VC"))
    render_to(os.path.join(args.output_dir, "pos"))


if __name__ == "__main__":
    main()

import os
import re
import numpy as np
from plyfile import PlyData, PlyElement
from pathlib import Path

# ===================== CONFIG =====================
DEFAULT_CFG = {
    "sh_degree": 0,
    "source_path": "",
    "model_path": "",
    "images": "images",
    "depths": "",
    "resolution": 2,
    "white_background": False,
    "train_test_exp": False,
    "data_device": "cuda",
    "eval": False
}


# ===================================================

def format_value_for_namespace(v):
    if isinstance(v, str):
        return f'"{v}"'
    elif v is None:
        return "None"
    return str(v)

def generate_cfg_args(root_dir: str):
    cfg = DEFAULT_CFG.copy()
    cfg["model_path"] = str(Path(root_dir))
    cfg["source_path"] = str(Path(root_dir).parent)
    items = [f"{k}={format_value_for_namespace(v)}" for k, v in cfg.items()]
    text = "Namespace(" + ", ".join(items) + ")"
    out_path = os.path.join(root_dir, "cfg_args")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[cfg_args] saved -> {out_path}\n")

def extract_step_number(filename: str) -> int:
    nums = re.findall(r"\d+", filename)
    return int(nums[-1]) if nums else -1

def find_latest_ply(ply_dir: str):
    candidates = []
    for fname in os.listdir(ply_dir):
        if fname.lower().endswith(".ply"):
            full = os.path.join(ply_dir, fname)
            candidates.append((extract_step_number(fname), full))
    if not candidates:
        raise FileNotFoundError(f"No ply in {ply_dir}")
    candidates.sort(reverse=True)
    step, path = candidates[0]
    print(f"Latest ply: step={step}, {path}\n")
    return path, step

def build_attribute_list(sh_degree):
    """复刻原版 construct_list_of_attributes"""
    attr = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # f_dc always 3
    for i in range(3):
        attr.append(f"f_dc_{i}")
    # f_rest count = 3 * ((sh_degree+1)**2 - 1)
    num_rest_coeff = (sh_degree + 1) ** 2 - 1
    if num_rest_coeff > 0:
        for i in range(num_rest_coeff * 3):
            attr.append(f"f_rest_{i}")
    attr.append("opacity")
    attr.extend(["scale_0", "scale_1", "scale_2"])
    attr.extend(["rot_0", "rot_1", "rot_2", "rot_3"])
    return attr

def gsplat_to_vanilla_ply(src_ply, dst_ply, sh_degree=3):
    ply_in = PlyData.read(src_ply)
    arr = ply_in["vertex"].data
    N = arr.shape[0]

    # 1. read raw data from gsplat ply
    xyz = np.column_stack([arr["x"], arr["y"], arr["z"]])
    normals = np.zeros_like(xyz, dtype=np.float32)

    f_dc = np.column_stack([arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"]])

    # collect all f_rest fields
    num_rest_coeff = (sh_degree + 1) ** 2 - 1
    if num_rest_coeff > 0:
        f_rest_cols = []

        for i in range(num_rest_coeff * 3):
            f_rest_cols.append(arr[f"f_rest_{i}"][:, None])
        f_rest = np.hstack(f_rest_cols)

    gs_opacity = arr["opacity"][:, None]
    gs_scale = np.column_stack([arr["scale_0"], arr["scale_1"], arr["scale_2"]])
    rotation = np.column_stack([arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"]])

    # 2. 关键变换，对齐 vanilla save_ply
    ply_opacity = gs_opacity
    ply_scale = gs_scale

    # 3. 拼接顺序严格和原版一致
    if num_rest_coeff == 0:
        concat_data = np.concatenate([
            xyz,
            normals,
            f_dc,
            ply_opacity,
            ply_scale,
            rotation
        ], axis=1)
    else:
        concat_data = np.concatenate([
                xyz,
                normals,
                f_dc,
                f_rest,
                ply_opacity,
                ply_scale,
                rotation
            ], axis=1)
        
    attr_list = build_attribute_list(sh_degree)
    dtype = [(name, "f4") for name in attr_list]
    elements = np.empty(N, dtype=dtype)
    elements[:] = list(map(tuple, concat_data))

    el = PlyElement.describe(elements, "vertex")
    os.makedirs(os.path.dirname(dst_ply), exist_ok=True)
    PlyData([el]).write(dst_ply)
    print(f"✅ Export vanilla3dgs standard ply: {dst_ply}")

if __name__ == "__main__":
    ROOT = '/media/zc/SSD/ar/ABXBJQ_BJ/B3/small_section/gsplat_voxel_plane_mcmc'
    generate_cfg_args(ROOT)
    latest_ply, step = find_latest_ply(os.path.join(ROOT, 'ply'))    
    out_iter_dir = os.path.join(ROOT, "point_cloud", f"iteration_{step}")
    out_ply_path = os.path.join(out_iter_dir, "point_cloud.ply")

    gsplat_to_vanilla_ply(latest_ply, out_ply_path, sh_degree=0)

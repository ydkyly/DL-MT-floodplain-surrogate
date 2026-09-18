# -*- coding: utf-8 -*-
import os, random, numpy as np, h5py
from .paths import XY_PATH

def _all_hdf_paths(hdf_dir):
    out=[]
    for name in os.listdir(hdf_dir):
        if name.lower().endswith(('.hdf', '.h5', '.hdf5')):
            out.append(os.path.join(hdf_dir, name))
    out.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
             if os.path.splitext(os.path.basename(p))[0].isdigit() else os.path.basename(p))
    return out

def _read_xy(path):
    with h5py.File(path, "r") as h5:
        if XY_PATH not in h5:
            raise KeyError(f"{os.path.basename(path)} 中找不到坐标数据集：{XY_PATH}")
        xy = h5[XY_PATH][:].astype(np.float32)  # [H,2]
    return xy

def dump_xy(args):
    hdf_dir = args.hdf_dir
    out = args.out
    tol = float(args.tolerance)
    max_check = int(args.max_check)

    files = _all_hdf_paths(hdf_dir)
    assert files, f"{hdf_dir} 内未找到 .hdf/.h5/.hdf5"

    # 读第一份作为参考
    ref_xy = _read_xy(files[0])
    H = ref_xy.shape[0]
    print(f"[dump_xy] 参考文件：{os.path.basename(files[0])}  XY形状={ref_xy.shape}")

    # 抽查其它文件是否同构（形状一致 + 数值一致/近似一致）
    others = files[1:]
    if max_check > 0 and len(others) > max_check:
        others = random.sample(others, max_check)
        print(f"[dump_xy] 抽查 {max_check} 份 HDF 的坐标一致性（总数={len(files)}）")
    else:
        print(f"[dump_xy] 检查全部 {len(files)} 份 HDF 的坐标一致性")

    for p in others:
        xy = _read_xy(p)
        if xy.shape != ref_xy.shape:
            raise ValueError(f"坐标形状不一致：{os.path.basename(p)} {xy.shape} vs 参考 {ref_xy.shape}")
        # 数值一致性（容差内）
        diff = np.linalg.norm(xy - ref_xy, axis=1)
        max_err = float(diff.max())
        if max_err > tol:
            raise ValueError(f"坐标数值不一致（max_err={max_err:.3e} > tol={tol:.3e}）：{os.path.basename(p)}")
    # 保存
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.save(out, ref_xy)
    print(f"[dump_xy] OK → {out}  (H={H})")

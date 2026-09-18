# -*- coding: utf-8 -*-
"""
basis_builder.py
用 TruncatedSVD 在 (N_floods*T)×H 的“水深矩阵”上构建空间基 B（K×H），可选：
 - center=True ：对列做均值中心化，并保存 mean.npy（全H），meta.center=True
 - wet_threshold：湿区筛列（仅保留任一时刻≥阈值的列参与构基），回填到全H
 - append_mean：将 μ 作为最后一行拼入B，meta.append_mean=True（与旧流程兼容，不推荐）
默认路径来自 paths.py 的 DEPTH_PATH，也可 data_path 覆盖。
"""
import os
import glob
import json
import time
from typing import List, Optional, Tuple

import h5py
import numpy as np
from sklearn.decomposition import TruncatedSVD

from .paths import DEPTH_PATH


def _list_hdf_files(hdf_dir: str, pattern: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(hdf_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"未在 {hdf_dir} 中找到匹配文件：{pattern}")
    return files

def _probe_shape(file: str, data_path: str) -> Tuple[int, int]:
    with h5py.File(file, "r") as h5:
        Y = h5[data_path][...]
    if Y.ndim != 2:
        raise ValueError(f"{data_path} 期望为 2D (T,H)，但在 {file} 中得到 {Y.shape}")
    T, H = int(Y.shape[0]), int(Y.shape[1])
    return T, H

def _compute_wet_mask(files: List[str], data_path: str, T: int, H: int, wet_threshold: float,
                      dtype=np.float32) -> np.ndarray:
    mask = np.zeros((H,), dtype=bool)
    t0 = time.time()
    for i, fp in enumerate(files, 1):
        with h5py.File(fp, "r") as h5:
            Y = h5[data_path][...].astype(dtype, copy=False)  # (T,H)
        Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
        mask |= (Y >= wet_threshold).any(axis=0)
        if i % 10 == 0 or i == len(files):
            print(f"[WetMask] {i}/{len(files)} processed | wet_cols={mask.sum()} / {H} | {time.time()-t0:.1f}s")
    if mask.sum() == 0:
        print("[WetMask][Warn] 未找到满足阈值的列。将保留全部列。")
        mask[:] = True
    return mask

def build_basis(
    hdf_dir: str,
    pattern: str,
    rank_k: int,
    save_path: str,
    data_path: Optional[str] = None,
    svd_iter: int = 5,
    random_state: int = 42,
    wet_threshold: Optional[float] = None,
    center: bool = False,
    append_mean: bool = False,
    dtype = np.float32
) -> str:
    data_path = data_path or DEPTH_PATH
    t_all = time.time()
    files = _list_hdf_files(hdf_dir, pattern)
    T, H = _probe_shape(files[0], data_path)
    print(f"[Basis] files={len(files)}  T={T}  H={H}  K={rank_k}  wet_threshold={wet_threshold}  center={center}  append_mean={append_mean}")

    # 湿区筛列
    use_mask = False
    wet_mask = None
    H_eff = H
    if wet_threshold is not None:
        use_mask = True
        wet_mask = _compute_wet_mask(files, data_path, T, H, float(wet_threshold), dtype=dtype)
        H_eff = int(wet_mask.sum())
        print(f"[Basis] 使用湿区掩膜：保留 {H_eff}/{H} 列（{H_eff/H*100:.2f}%）")

    # 分配 (N_floods*T, H_eff)
    N = len(files) * T
    X = np.empty((N, H_eff), dtype=dtype)

    # 读入
    wr = 0; t0 = time.time()
    for i, fp in enumerate(files, 1):
        with h5py.File(fp, "r") as h5:
            Y = h5[data_path][...].astype(dtype, copy=False)  # (T,H)
        Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
        if use_mask: Y = Y[:, wet_mask]  # (T, H_eff)
        X[wr:wr+T, :] = Y
        wr += T
        if i % 10 == 0 or i == len(files):
            print(f"[Load] {i}/{len(files)} loaded | {time.time()-t0:.1f}s")

    # 计算 μ（按列）
    mu_eff = X.mean(axis=0) if center else np.zeros((H_eff,), dtype=dtype)
    if center:
        X = X - mu_eff[None, :]

    # Truncated SVD
    print(f"[SVD] start: samples={X.shape[0]}, features={X.shape[1]}, K={rank_k}, n_iter={svd_iter}")
    svd = TruncatedSVD(n_components=rank_k, n_iter=svd_iter, random_state=random_state)
    svd.fit(X)
    B_eff = svd.components_.astype(dtype, copy=False)  # (K, H_eff)
    evr_sum = float(svd.explained_variance_ratio_.sum())
    print(f"[SVD] done. explained_variance_ratio_sum={evr_sum:.4f}")

    # 回填到全 H
    if use_mask:
        B_full = np.zeros((rank_k, H), dtype=dtype)
        B_full[:, wet_mask] = B_eff
        mu_full = np.zeros((H,), dtype=dtype)
        mu_full[wet_mask] = mu_eff
    else:
        B_full = B_eff
        mu_full = mu_eff if center else np.zeros((H,), dtype=dtype)

    # 可选：append_mean（兼容旧流程，不推荐）
    out_B = B_full
    if append_mean:
        out_B = np.vstack([B_full, mu_full[None, :]]).astype(dtype, copy=False)

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, out_B)
    meta = {
        "hdf_dir": hdf_dir,
        "pattern": pattern,
        "data_path": data_path,
        "rank_k": int(rank_k),
        "T": int(T),
        "H": int(H),
        "H_eff": int(H_eff),
        "files": int(len(files)),
        "explained_variance_ratio_sum": evr_sum,
        "svd_iter": int(svd_iter),
        "random_state": int(random_state),
        "wet_threshold": None if wet_threshold is None else float(wet_threshold),
        "center": bool(center),
        "append_mean": bool(append_mean),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    meta_path = os.path.splitext(save_path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 若 center=True 且未 append_mean，则单独保存 mean.npy
    if center and (not append_mean):
        mean_path = os.path.join(os.path.dirname(save_path), "mean.npy")
        np.save(mean_path, mu_full.astype(dtype, copy=False))
        print(f"[Save] mean  → {mean_path}")

    print(f"[Save] basis → {save_path} (shape={out_B.shape})")
    print(f"[Save] meta  → {meta_path}")
    print(f"[Done] total time: {time.time()-t_all:.1f}s")
    return save_path

def build_basis_from_args(args):
    save_path = args.save
    if os.path.isdir(save_path) or save_path.endswith(("/", "\\")):
        save_path = os.path.join(save_path, f"basis_k{args.rank_k}.npy")
    return build_basis(
        hdf_dir=args.hdf_dir,
        pattern=args.pattern,
        rank_k=args.rank_k,
        save_path=save_path,
        data_path=getattr(args, "data_path", None),
        svd_iter=getattr(args, "svd_iter", 5),
        random_state=getattr(args, "random_state", 42),
        wet_threshold=getattr(args, "wet_threshold", None),
        center=bool(getattr(args, "center", False)),
        append_mean=bool(getattr(args, "append_mean", False))
    )

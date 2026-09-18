# -*- coding: utf-8 -*-
"""
basis_checker.py
对给定 basis.npy 做重建自检：逐 HDF 文件计算 (Y_hat - Y) 的 RMSE/MAE 等统计，导出 CSV，并可选绘图。
支持含均值项的 PCA/EOF 基：
  - append_mean=False：自动读取同目录 mean.npy（若存在）
  - append_mean=True ：自动把 basis.npy 最后一行当作 mu，前 K 行当作正交基 B
重建公式：
  Y_hat = ((Y - mu) @ B.T) @ B + mu
"""

import os, glob, json, random, time
import numpy as np
import h5py
from typing import List, Optional, Tuple
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


def _load_basis_and_mean(basis_path: str) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
    """
    返回：(B_main[K,H], mu[H] or None, used_mu(bool))
    兼容：
      - 同目录 basis_meta.json 指示 append_mean
      - 同目录 mean.npy
      - 若 append_mean=True，则 basis.npy 最后一行作为 mu
    """
    if not os.path.isfile(basis_path):
        raise FileNotFoundError(f"basis 文件不存在：{basis_path}")
    B = np.load(basis_path)
    if B.ndim != 2:
        raise ValueError(f"basis 期望为 2D (K,H)，得到 {B.shape}")

    out_dir = os.path.dirname(basis_path)
    meta_path = os.path.join(out_dir, "basis_meta.json")
    append_mean = None
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            append_mean = bool(meta.get("append_mean", False))
        except Exception:
            append_mean = None  # 读失败则走后续兜底

    mu = None
    used_mu = False

    # 1) 明确 append_mean=True：最后一行是 mu
    if append_mean is True:
        if B.shape[0] < 2:
            raise ValueError(f"append_mean=True 但 basis 行数过小：{B.shape}")
        mu = B[-1].astype(np.float32)
        B = B[:-1].astype(np.float32)
        used_mu = True
        return B, mu, used_mu

    # 2) append_mean=False 或未知：优先找 mean.npy
    mean_path = os.path.join(out_dir, "mean.npy")
    if os.path.isfile(mean_path):
        mu = np.load(mean_path).astype(np.float32)
        B = B.astype(np.float32)
        used_mu = True
        return B, mu, used_mu

    # 3) 元信息缺失且 mean.npy 不存在：不加 mu（相当于对原场做低秩投影）
    B = B.astype(np.float32)
    return B, None, False


def _recon_err_one(h5_path: str, data_path: str, B: np.ndarray, mu: Optional[np.ndarray]) -> dict:
    with h5py.File(h5_path, "r") as h5:
        Y = h5[data_path][...]  # (T,H)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # 低秩重建（PCA/EOF 正确形式）：Y_hat = ((Y - mu) B^T) B + mu
    if mu is not None:
        if mu.ndim != 1 or mu.shape[0] != Y.shape[1]:
            raise ValueError(f"mu 形状 {mu.shape} 与数据 H={Y.shape[1]} 不一致")
        Ym = Y - mu[None, :]
        C  = Ym @ B.T
        Yh = C @ B + mu[None, :]
    else:
        # 无均值项：退化为 Y_hat = (Y B^T) B
        C  = Y @ B.T
        Yh = C @ B

    diff = Yh - Y
    adiff = np.abs(diff)

    rmse = float(np.sqrt(np.mean(diff**2)))
    mae  = float(np.mean(adiff))
    p95  = float(np.percentile(adiff, 95))
    p99  = float(np.percentile(adiff, 99))
    maxa = float(np.max(adiff))
    return dict(rmse=rmse, mae=mae, p95=p95, p99=p99, max_abs=maxa, T=int(Y.shape[0]), H=int(Y.shape[1]))


def check_basis(
    basis_path: str,
    hdf_dir: str,
    pattern: str = "*.hdf",
    data_path: Optional[str] = None,
    sample_n: Optional[int] = None,      # 随机抽样 N 个文件（None=用全部）
    random_state: int = 42,
    save_dir: Optional[str] = None,
    save_fig: bool = True
) -> str:
    """
    对 basis.npy 进行重建自检：
      - 逐文件计算 RMSE/MAE/|diff| p95/p99/max_abs
      - 导出 CSV 与汇总 JSON
      - 可选保存直方图与箱线图
    返回：统计 CSV 的保存路径
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = time.time()
    data_path = data_path or DEPTH_PATH

    B, mu, used_mu = _load_basis_and_mean(basis_path)  # B:(K,Hb), mu:(H) or None
    K, Hb = int(B.shape[0]), int(B.shape[1])

    files = _list_hdf_files(hdf_dir, pattern)
    if sample_n is not None and sample_n > 0 and sample_n < len(files):
        random.Random(random_state).shuffle(files)
        files = files[:sample_n]

    # 形状探测
    T0, H0 = _probe_shape(files[0], data_path)
    if H0 != Hb:
        raise ValueError(f"basis H={Hb} 与数据 H={H0} 不一致，请检查 basis 是否匹配数据。")
    if mu is not None and mu.shape[0] != H0:
        raise ValueError(f"mu H={mu.shape[0]} 与数据 H={H0} 不一致。")

    # 输出目录
    save_dir = save_dir or os.path.join(os.path.dirname(basis_path), "CheckBasis")
    os.makedirs(save_dir, exist_ok=True)

    print(f"[CheckBasis] basis={os.path.abspath(basis_path)}  K={K} H={H0}  used_mu={used_mu}")

    # 逐文件评估
    rows = []
    for i, fp in enumerate(files, 1):
        st = time.time()
        stat = _recon_err_one(fp, data_path, B, mu)
        rows.append((os.path.basename(fp), stat["rmse"], stat["mae"], stat["p95"], stat["p99"],
                     stat["max_abs"], stat["T"], stat["H"]))
        if i % 10 == 0 or i == len(files):
            print(f"[CheckBasis] {i}/{len(files)}  last_rmse={stat['rmse']:.4f}  ({time.time()-st:.2f}s)")

    # 保存 CSV
    csv_path = os.path.join(save_dir, f"basis_check_k{K}_{len(files)}files.csv")
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "rmse", "mae", "p95_abs", "p99_abs", "max_abs", "T", "H"])
        for r in rows:
            w.writerow(list(r))

    # 汇总
    arr = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in rows], dtype=np.float32)
    summary = {
        "files": len(files), "K": K, "H": H0, "T": T0,
        "rmse_mean": float(arr[:, 0].mean()), "rmse_median": float(np.median(arr[:, 0])),
        "mae_mean":  float(arr[:, 1].mean()), "mae_median":  float(np.median(arr[:, 1])),
        "p95_mean":  float(arr[:, 2].mean()), "p95_median":  float(np.median(arr[:, 2])),
        "p99_mean":  float(arr[:, 3].mean()), "p99_median":  float(np.median(arr[:, 3])),
        "max_abs_mean": float(arr[:, 4].mean()), "max_abs_median": float(np.median(arr[:, 4])),
        "basis_path": os.path.abspath(basis_path),
        "used_mu": bool(used_mu),
        "mean_path": os.path.join(os.path.dirname(basis_path), "mean.npy") if used_mu else None,
        "data_path": data_path,
        "hdf_dir": os.path.abspath(hdf_dir),
        "pattern": pattern,
        "sample_n": None if (sample_n is None) else int(len(files)),
        "random_state": int(random_state),
        "elapsed_sec": float(time.time() - t0),
    }
    with open(os.path.join(save_dir, f"basis_check_k{K}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[CheckBasis][Summary]", json.dumps(summary, ensure_ascii=False, indent=2))

    # 绘图（可选）
    if save_fig:
        rmse = arr[:, 0]
        mae  = arr[:, 1]

        plt.figure(figsize=(6, 4), dpi=140)
        plt.hist(rmse, bins=30)
        plt.xlabel("RMSE (m)")
        plt.ylabel("Count")
        plt.title(f"Basis Reconstruction RMSE (K={K}, n={len(files)}, mu={used_mu})")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"hist_rmse_k{K}.png"))
        plt.close()

        plt.figure(figsize=(6, 4), dpi=140)
        plt.hist(mae, bins=30)
        plt.xlabel("MAE (m)")
        plt.ylabel("Count")
        plt.title(f"Basis Reconstruction MAE (K={K}, n={len(files)}, mu={used_mu})")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"hist_mae_k{K}.png"))
        plt.close()

        plt.figure(figsize=(5, 4), dpi=140)
        plt.boxplot([rmse, mae], labels=["RMSE", "MAE"], showfliers=False)
        plt.title(f"Reconstruction Errors (K={K}, mu={used_mu})")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"box_err_k{K}.png"))
        plt.close()

    print(f"[CheckBasis] CSV → {csv_path}")
    return csv_path


# 适配 argparse.Namespace 的薄封装（供 flood_main.py 直接调用）
def check_basis_from_args(args):
    return check_basis(
        basis_path=args.basis,
        hdf_dir=args.hdf_dir,
        pattern=args.pattern,
        data_path=getattr(args, "data_path", None),
        sample_n=getattr(args, "sample_n", None),
        random_state=getattr(args, "random_state", 42),
        save_dir=getattr(args, "save_dir", None),
        save_fig=bool(getattr(args, "save_fig", True))
    )

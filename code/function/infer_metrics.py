# -*- coding: utf-8 -*-
"""
postmetrics_npy_hdf.py

功能：
1) 从 pred_dir 扫描并读取各洪水场次(sid)的 npy 预测：
   - 深度：{sid}_Y_pred*.npy / {sid}_depth_pred*.npy
   - 流速分量：{sid}_Vx_pred*.npy, {sid}_Vy_pred*.npy
2) 从 gt_dir 读取真值 HDF5/HDF：
   - 自动在文件里搜索 depth / vx / vy 数据集（支持关键词匹配与形状筛选）
3) 计算并保存指标：
   - 整场洪水（全时段汇总）：RMSE, MAE, PCC, F1, POD, FAR
   - 每个时刻：同上
   - Vx/Vy 及 |v|：RMSE, MAE, PCC（默认仅在真值受淹 mask 上统计）
4) 输出：
   - out_dir/metrics_event_summary.csv
   - out_dir/metrics_timestep.csv
   - out_dir/metrics_runinfo.json

依赖：numpy, pandas, h5py, torch
"""

from __future__ import annotations

import os
import re
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd
import torch

try:
    import h5py
except Exception as e:
    raise RuntimeError("需要安装 h5py 才能读取 HDF5/HDF 文件：pip install h5py") from e

# ==== 顶部：优先导入项目内路径常量 ====
try:
    # 确保 postmetrics_npy_hdf.py 与 paths.py 在同一工程可 import 的位置
    from paths import DEPTH_PATH as _DEPTH_PATH, VX_PATH as _VX_PATH, VY_PATH as _VY_PATH
except Exception:
    _DEPTH_PATH, _VX_PATH, _VY_PATH = None, None, None


def _try_project_dataset_path(h5: "h5py.File", path: Optional[str]) -> Optional[str]:
    """
    优先使用项目 paths.py 的精确路径；若不存在，再尝试替换 2D Flow Area 名称（如 Perimeter 1 → 其他）。
    """
    if not path:
        return None
    if path in h5:
        return path

    # 常见差异：2D Flow Areas/Perimeter 1/ ... 中 "Perimeter 1" 名称可能不同
    marker = "2D Flow Areas/"
    if marker not in path:
        return None

    pre, rest = path.split(marker, 1)
    parts = rest.split("/", 1)
    if len(parts) != 2:
        return None

    _area_name = parts[0]
    suffix = parts[1]
    grp_path = (pre + marker).rstrip("/")

    if grp_path in h5:
        grp = h5[grp_path]
        # 遍历实际存在的 2D Flow Area 名称，拼回完整路径
        for k in grp.keys():
            cand = f"{pre}{marker}{k}/{suffix}"
            if cand in h5:
                return cand

    return None


def _read_TxH_from_h5(h5: "h5py.File", ds_path: str, timesteps: Optional[int] = None) -> np.ndarray:
    """
    从 h5 数据集读出并规整为 [T,H]
    """
    arr = np.array(h5[ds_path])
    arr = _normalize_TxH(arr, timesteps=timesteps).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def load_gt_fields_from_hdf(
    hdf_path: Path,
    timesteps: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    返回 dict:
      depth: [T,H]
      vx   : [T,H]（若找到）
      vy   : [T,H]（若找到）

    读取优先级：
      1) 项目 paths.py 的 DEPTH_PATH/VX_PATH/VY_PATH（含 Flow Area 名称替换尝试）
      2) 关键词/形状自动搜索（兜底）
    """
    out: Dict[str, np.ndarray] = {}
    with h5py.File(hdf_path, "r") as h5:
        # ---- 1) 优先按项目路径读取 ----
        depth_path = _try_project_dataset_path(h5, _DEPTH_PATH)
        vx_path    = _try_project_dataset_path(h5, _VX_PATH)
        vy_path    = _try_project_dataset_path(h5, _VY_PATH)

        # ---- 2) 若项目路径读不到，兜底用关键词搜索 ----
        if depth_path is None:
            depth_path = _pick_dataset(h5, _DEPTH_KEYWORDS)
        if depth_path is None:
            raise RuntimeError(f"在 HDF 中未找到 depth 数据集（项目路径+关键词均失败）：{hdf_path}")

        out["depth"] = _read_TxH_from_h5(h5, depth_path, timesteps=timesteps)

        if vx_path is None:
            vx_path = _pick_dataset(h5, _VX_KEYWORDS)
        if vx_path is not None:
            out["vx"] = _read_TxH_from_h5(h5, vx_path, timesteps=timesteps)

        if vy_path is None:
            vy_path = _pick_dataset(h5, _VY_KEYWORDS)
        if vy_path is not None:
            out["vy"] = _read_TxH_from_h5(h5, vy_path, timesteps=timesteps)

    # 清洗 + 深度非负
    out["depth"] = np.maximum(out["depth"], 0.0)
    for k in ("vx", "vy"):
        if k in out:
            out[k] = np.nan_to_num(out[k], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return out


# -----------------------------
# 基础指标（torch实现，便于与你训练期一致）
# -----------------------------
def _to_1d_torch(a: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(a))
    if t.dtype != torch.float32 and t.dtype != torch.float64:
        t = t.float()
    return t.reshape(-1)


def rmse_t(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    p = np.asarray(pred)
    g = np.asarray(gt)
    if mask is not None:
        p = p[mask]
        g = g[mask]
        if p.size == 0:
            return float("nan")
    pt = _to_1d_torch(p).float()
    gt = _to_1d_torch(g).float()
    return torch.sqrt(torch.mean((pt - gt) ** 2)).item()


def mae_t(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    p = np.asarray(pred)
    g = np.asarray(gt)
    if mask is not None:
        p = p[mask]
        g = g[mask]
        if p.size == 0:
            return float("nan")
    pt = _to_1d_torch(p).float()
    gt = _to_1d_torch(g).float()
    return torch.mean(torch.abs(pt - gt)).item()


def pcc_t(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    if mask is not None:
        p = p[mask]
        g = g[mask]
        if p.size == 0:
            return float("nan")

    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

    pt = _to_1d_torch(p).double()
    gt = _to_1d_torch(g).double()

    pt = pt - pt.mean()
    gt = gt - gt.mean()

    denom = torch.sqrt(torch.sum(pt ** 2)) * torch.sqrt(torch.sum(gt ** 2))
    if denom.item() == 0.0:
        return 0.0
    return (torch.sum(pt * gt) / denom).item()


def f1_pod_far_t(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.1) -> Tuple[float, float, float]:
    """
    受淹二分类指标（与你提供的实现一致）：
      p = pred >= threshold; t = gt >= threshold
      POD = recall = TP/(TP+FN)
      FAR = FP/(TP+FP)
      F1  = 2*precision*recall/(precision+recall)
    """
    p = torch.from_numpy(np.asarray(pred)) >= threshold
    t = torch.from_numpy(np.asarray(gt)) >= threshold
    TP = torch.logical_and(p, t).sum().item()
    FP = torch.logical_and(p, ~t).sum().item()
    FN = torch.logical_and(~p, t).sum().item()
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # POD
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    far = FP / (TP + FP) if (TP + FP) > 0 else 0.0
    return float(f1), float(recall), float(far)


# -----------------------------
# 数据读取：npy
# -----------------------------
_PRED_SUFFIX_PRIORITY = {
    "depth": [
        "_Y_pred_sanitized.npy",
        "_Y_pred.npy",
        "_Y_pred_raw.npy",
        "_depth_pred_sanitized.npy",
        "_depth_pred.npy",
        "_depth_pred_raw.npy",
    ],
    "vx": [
        "_Vx_pred.npy",
        "_vx_pred.npy",
        "_Vx_pred_raw.npy",
    ],
    "vy": [
        "_Vy_pred.npy",
        "_vy_pred.npy",
        "_Vy_pred_raw.npy",
    ],
}


def _discover_sids(pred_dir: Path) -> List[str]:
    """
    从 pred_dir 里扫描深度预测文件，抽取 sid。
    支持：{sid}_Y_pred*.npy / {sid}_depth_pred*.npy
    """
    pred_dir = Path(pred_dir)
    files = list(pred_dir.rglob("*.npy"))
    sid_set = set()

    pat_list = [
        re.compile(r"^(?P<sid>.+?)_Y_pred.*\.npy$", re.IGNORECASE),
        re.compile(r"^(?P<sid>.+?)_depth_pred.*\.npy$", re.IGNORECASE),
    ]
    for f in files:
        name = f.name
        for pat in pat_list:
            m = pat.match(name)
            if m:
                sid_set.add(m.group("sid"))
                break

    return sorted(sid_set)


def _find_pred_file(pred_dir: Path, sid: str, kind: str) -> Optional[Path]:
    pred_dir = Path(pred_dir)
    for suf in _PRED_SUFFIX_PRIORITY.get(kind, []):
        cand = pred_dir / f"{sid}{suf}"
        if cand.is_file():
            return cand
        # 递归：允许放在子目录
        hits = list(pred_dir.rglob(f"{sid}{suf}"))
        if hits:
            hits.sort(key=lambda p: len(str(p)))
            return hits[0]
    return None


def _normalize_TxH(arr: np.ndarray, timesteps: Optional[int] = None) -> np.ndarray:
    """
    统一为 [T,H]。
    常见情况：
      - [T,H]
      - [H,T]（通过 timesteps 或启发式转置）
      - [T,H,1] / [1,T,H]（先 squeeze）
    """
    a = np.asarray(arr)
    # squeeze 单维
    while a.ndim > 2 and 1 in a.shape:
        a = np.squeeze(a)

    if a.ndim == 1:
        a = a[None, :]  # [1,H]
    if a.ndim != 2:
        raise ValueError(f"期望二维数组 [T,H]，但得到 shape={a.shape}")

    T, H = a.shape

    # 用 timesteps 判定
    if timesteps is not None:
        if T == timesteps:
            return a
        if H == timesteps:
            return a.T

    # 启发式：通常 H >> T
    if T > H and H >= 2:
        return a.T
    return a


def load_pred_TxH(pred_path: Path, timesteps: Optional[int] = None) -> np.ndarray:
    a = np.load(pred_path)
    a = _normalize_TxH(a, timesteps=timesteps).astype(np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


# -----------------------------
# 数据读取：HDF5/HDF（自动找数据集）
# -----------------------------
_DEPTH_KEYWORDS = ["depth", "water depth"]
_VX_KEYWORDS = ["vx", "velx", "velocityx", "u", "x-velocity", "x velocity", "velocity x"]
_VY_KEYWORDS = ["vy", "vely", "velocityy", "v", "y-velocity", "y velocity", "velocity y"]


def _iter_datasets(h5: "h5py.File") -> Iterable[Tuple[str, "h5py.Dataset"]]:
    out: List[Tuple[str, "h5py.Dataset"]] = []

    def _visit(name, obj):
        import h5py as _h5py  # local import for type
        if isinstance(obj, _h5py.Dataset):
            out.append((name, obj))

    h5.visititems(_visit)
    return out


def _score_dataset(name: str, ds, keywords: List[str]) -> float:
    """
    给数据集打分：关键词命中 + 形状偏好（二维优先、数据量大优先）
    """
    n = name.lower()
    hit = 0
    for k in keywords:
        if k in n:
            hit += 1
    shape = ds.shape
    ndim = len(shape)
    size = int(np.prod(shape)) if shape is not None else 0
    score = hit * 10.0
    # 形状偏好
    if ndim == 2:
        score += 3.0
    elif ndim == 3 and (shape[-1] in (1, 2)):
        score += 2.0
    # 越大越可能是 [T,H]
    score += math.log10(size + 1.0) * 0.5
    return score


def _pick_dataset(h5: "h5py.File", keywords: List[str]) -> Optional[str]:
    cands = []
    for name, ds in _iter_datasets(h5):
        s = _score_dataset(name, ds, keywords)
        if s > 0:
            cands.append((s, name, ds.shape))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]


def _find_gt_file(gt_dir: Path, sid: str) -> Optional[Path]:
    """
    gt_dir 可以是：
      - 目录：里面每个 sid 一个 hdf/h5/hdf5
      - 单个文件：直接返回该文件
    """
    gt_dir = Path(gt_dir)
    if gt_dir.is_file():
        return gt_dir

    exts = [".hdf", ".h5", ".hdf5", ".hdf5.gz"]
    for ext in exts:
        cand = gt_dir / f"{sid}{ext}"
        if cand.is_file():
            return cand

    # 宽松匹配：sid 开头 or 包含 sid
    hits = []
    for f in gt_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in [".hdf", ".h5", ".hdf5"]:
            fn = f.stem
            if fn == sid or fn.startswith(sid) or (sid in fn):
                hits.append(f)
    if hits:
        hits.sort(key=lambda p: len(str(p)))
        return hits[0]
    return None



# -----------------------------
# 评估主逻辑
# -----------------------------
@dataclass
class EvalConfig:
    pred_dir: str
    gt_dir: str
    out_dir: str
    flood_threshold: float = 0.1

    # velocity 指标是否只在“真值受淹区”统计（推荐 True）
    vel_on_wet_only: bool = True

    # 若 T 不一致，是否裁剪到 min(T_pred, T_gt)
    allow_time_crop: bool = True

    # 逐 sid 单独输出每时刻 CSV（会多生成一些文件）
    split_timestep_csv_per_sid: bool = False


def _align_time(pred: np.ndarray, gt: np.ndarray, allow_crop: bool) -> Tuple[np.ndarray, np.ndarray]:
    Tp = pred.shape[0]
    Tg = gt.shape[0]
    if Tp == Tg:
        return pred, gt
    if not allow_crop:
        raise ValueError(f"T 不一致：pred T={Tp}, gt T={Tg}")
    Tm = min(Tp, Tg)
    return pred[:Tm], gt[:Tm]


def _align_space(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if pred.shape[1] != gt.shape[1]:
        raise ValueError(f"H 不一致：pred H={pred.shape[1]}, gt H={gt.shape[1]}")
    return pred, gt


def _compute_depth_metrics_event(pred: np.ndarray, gt: np.ndarray, thr: float) -> Dict[str, float]:
    return {
        "RMSE_depth": rmse_t(pred, gt),
        "MAE_depth": mae_t(pred, gt),
        "PCC_depth": pcc_t(pred, gt),
        "F1_depth":  f1_pod_far_t(pred, gt, threshold=thr)[0],
        "POD_depth": f1_pod_far_t(pred, gt, threshold=thr)[1],
        "FAR_depth": f1_pod_far_t(pred, gt, threshold=thr)[2],
    }


def _compute_depth_metrics_frame(pred_h: np.ndarray, gt_h: np.ndarray, thr: float) -> Dict[str, float]:
    f1, pod, far = f1_pod_far_t(pred_h, gt_h, threshold=thr)
    return {
        "RMSE_depth": rmse_t(pred_h, gt_h),
        "MAE_depth": mae_t(pred_h, gt_h),
        "PCC_depth": pcc_t(pred_h, gt_h),
        "F1_depth":  f1,
        "POD_depth": pod,
        "FAR_depth": far,
    }


def _compute_cont_metrics_event(name: str, pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray]) -> Dict[str, float]:
    return {
        f"RMSE_{name}": rmse_t(pred, gt, mask=mask),
        f"MAE_{name}":  mae_t(pred, gt, mask=mask),
        f"PCC_{name}":  pcc_t(pred, gt, mask=mask),
    }


def _compute_cont_metrics_frame(name: str, pred_h: np.ndarray, gt_h: np.ndarray, mask_h: Optional[np.ndarray]) -> Dict[str, float]:
    return {
        f"RMSE_{name}": rmse_t(pred_h, gt_h, mask=mask_h),
        f"MAE_{name}":  mae_t(pred_h, gt_h, mask=mask_h),
        f"PCC_{name}":  pcc_t(pred_h, gt_h, mask=mask_h),
    }


def evaluate_predictions(cfg: EvalConfig) -> Tuple[Path, Path]:
    """
    执行评估，返回：
      (event_summary_csv_path, timestep_csv_path)
    """
    pred_dir = Path(cfg.pred_dir)
    gt_dir = Path(cfg.gt_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sids = _discover_sids(pred_dir)
    if not sids:
        raise RuntimeError(f"在 pred_dir 未发现深度预测文件（*_Y_pred*.npy / *_depth_pred*.npy）：{pred_dir}")

    event_rows: List[Dict[str, object]] = []
    timestep_rows: List[Dict[str, object]] = []

    for sid in sids:
        depth_pred_path = _find_pred_file(pred_dir, sid, "depth")
        if depth_pred_path is None:
            continue

        vx_pred_path = _find_pred_file(pred_dir, sid, "vx")
        vy_pred_path = _find_pred_file(pred_dir, sid, "vy")

        # 读预测
        depth_pred = load_pred_TxH(depth_pred_path)

        vx_pred = load_pred_TxH(vx_pred_path, timesteps=depth_pred.shape[0]) if vx_pred_path else None
        vy_pred = load_pred_TxH(vy_pred_path, timesteps=depth_pred.shape[0]) if vy_pred_path else None

        # 找真值文件并读取
        gt_path = _find_gt_file(gt_dir, sid)
        if gt_path is None:
            raise RuntimeError(f"未找到 sid={sid} 的真值 HDF 文件：gt_dir={gt_dir}")

        gt_fields = load_gt_fields_from_hdf(gt_path, timesteps=depth_pred.shape[0])
        depth_gt = gt_fields["depth"]
        vx_gt = gt_fields.get("vx", None)
        vy_gt = gt_fields.get("vy", None)

        # 对齐 T
        depth_pred, depth_gt = _align_time(depth_pred, depth_gt, allow_crop=cfg.allow_time_crop)
        depth_pred, depth_gt = _align_space(depth_pred, depth_gt)

        if vx_pred is not None and vx_gt is not None:
            vx_pred, vx_gt = _align_time(vx_pred, vx_gt, allow_crop=cfg.allow_time_crop)
            vx_pred, vx_gt = _align_space(vx_pred, vx_gt)

        if vy_pred is not None and vy_gt is not None:
            vy_pred, vy_gt = _align_time(vy_pred, vy_gt, allow_crop=cfg.allow_time_crop)
            vy_pred, vy_gt = _align_space(vy_pred, vy_gt)

        T = depth_pred.shape[0]

        # 速度 mask：真值受淹区（更合理）
        wet_mask_TxH = (depth_gt >= cfg.flood_threshold) if cfg.vel_on_wet_only else None

        # ---- 整场汇总 ----
        row_event: Dict[str, object] = {
            "sid": sid,
            "T": int(T),
            "H": int(depth_pred.shape[1]),
            "pred_depth_file": str(depth_pred_path),
            "gt_file": str(gt_path),
        }
        row_event.update(_compute_depth_metrics_event(depth_pred, depth_gt, cfg.flood_threshold))

        # vx/vy
        if vx_pred is not None and vx_gt is not None:
            mask = wet_mask_TxH if cfg.vel_on_wet_only else None
            row_event.update(_compute_cont_metrics_event("vx", vx_pred, vx_gt, mask=mask))
        if vy_pred is not None and vy_gt is not None:
            mask = wet_mask_TxH if cfg.vel_on_wet_only else None
            row_event.update(_compute_cont_metrics_event("vy", vy_pred, vy_gt, mask=mask))

        # |v|
        if (vx_pred is not None and vy_pred is not None and vx_gt is not None and vy_gt is not None):
            vmag_pred = np.sqrt(vx_pred**2 + vy_pred**2)
            vmag_gt = np.sqrt(vx_gt**2 + vy_gt**2)
            mask = wet_mask_TxH if cfg.vel_on_wet_only else None
            row_event.update(_compute_cont_metrics_event("vmag", vmag_pred, vmag_gt, mask=mask))

        event_rows.append(row_event)

        # ---- 每时刻 ----
        per_sid_rows: List[Dict[str, object]] = []
        for t in range(T):
            row_t: Dict[str, object] = {"sid": sid, "t": int(t)}
            row_t.update(_compute_depth_metrics_frame(depth_pred[t], depth_gt[t], cfg.flood_threshold))

            wet_mask_h = (wet_mask_TxH[t] if wet_mask_TxH is not None else None)

            if vx_pred is not None and vx_gt is not None:
                row_t.update(_compute_cont_metrics_frame("vx", vx_pred[t], vx_gt[t], mask_h=wet_mask_h))
            if vy_pred is not None and vy_gt is not None:
                row_t.update(_compute_cont_metrics_frame("vy", vy_pred[t], vy_gt[t], mask_h=wet_mask_h))

            if (vx_pred is not None and vy_pred is not None and vx_gt is not None and vy_gt is not None):
                vmag_pred_h = np.sqrt(vx_pred[t]**2 + vy_pred[t]**2)
                vmag_gt_h = np.sqrt(vx_gt[t]**2 + vy_gt[t]**2)
                row_t.update(_compute_cont_metrics_frame("vmag", vmag_pred_h, vmag_gt_h, mask_h=wet_mask_h))

            timestep_rows.append(row_t)
            per_sid_rows.append(row_t)

        if cfg.split_timestep_csv_per_sid:
            df_sid = pd.DataFrame(per_sid_rows)
            sid_csv = out_dir / f"{sid}_metrics_timestep.csv"
            df_sid.to_csv(sid_csv, index=False, encoding="utf-8-sig")

        print(
            f"[PostMetrics] {sid}  "
            f"RMSE={row_event.get('RMSE_depth', float('nan')):.4f}  "
            f"MAE={row_event.get('MAE_depth', float('nan')):.4f}  "
            f"PCC={row_event.get('PCC_depth', float('nan')):.4f}  "
            f"F1={row_event.get('F1_depth', float('nan')):.3f}  "
            f"POD={row_event.get('POD_depth', float('nan')):.3f}  "
            f"FAR={row_event.get('FAR_depth', float('nan')):.3f}"
        )

    # 输出 CSV
    df_event = pd.DataFrame(event_rows)
    df_time = pd.DataFrame(timestep_rows)

    event_csv = out_dir / "metrics_event_summary.csv"
    time_csv = out_dir / "metrics_timestep.csv"
    df_event.to_csv(event_csv, index=False, encoding="utf-8-sig")
    df_time.to_csv(time_csv, index=False, encoding="utf-8-sig")

    # 输出 runinfo
    runinfo = {
        "config": asdict(cfg),
        "num_sids": len(event_rows),
        "sids": [r["sid"] for r in event_rows],
        "event_csv": str(event_csv),
        "timestep_csv": str(time_csv),
        "note": {
            "F1/POD/FAR": "默认基于 depth>=threshold 的受淹二分类",
            "velocity_mask": "vel_on_wet_only=True 时，vx/vy/vmag 仅在真值受淹网格上统计",
        },
    }
    with open(out_dir / "metrics_runinfo.json", "w", encoding="utf-8") as f:
        json.dump(runinfo, f, ensure_ascii=False, indent=2)

    print(f"[PostMetrics] Saved: {event_csv}")
    print(f"[PostMetrics] Saved: {time_csv}")
    return event_csv, time_csv


# -----------------------------
# 可选：命令行
# -----------------------------
def _build_argparser():
    import argparse
    p = argparse.ArgumentParser("postmetrics_npy_hdf")
    p.add_argument("--pred_dir", default=r'D:\Work\qyb\ResNet-18\V2.4\Infer_PCA_res', help="预测 npy 所在目录")
    p.add_argument("--gt_dir", default=r'D:\Work\qyb\ResNet-18\data\test\hdf', help="真值 HDF 所在目录或单个文件")
    p.add_argument("--out_dir", default='Infer_PCA_07271', help="输出目录")
    p.add_argument("--flood_threshold", type=float, default=0.1, help="受淹阈值（用于 F1/POD/FAR 与 wet mask）")
    p.add_argument("--vel_on_wet_only", type=int, default=0, help="1: 流速仅在真值受淹网格统计；0: 全域统计")
    p.add_argument("--allow_time_crop", type=int, default=1, help="1: T 不一致则裁剪到 min(T)；0: 直接报错")
    p.add_argument("--split_timestep_csv_per_sid", type=int, default=1, help="1: 每个 sid 另存一份逐时刻 CSV")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    cfg = EvalConfig(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        out_dir=args.out_dir,
        flood_threshold=float(args.flood_threshold),
        vel_on_wet_only=bool(args.vel_on_wet_only),
        allow_time_crop=bool(args.allow_time_crop),
        split_timestep_csv_per_sid=bool(args.split_timestep_csv_per_sid),
    )
    evaluate_predictions(cfg)

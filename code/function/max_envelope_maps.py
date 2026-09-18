# -*- coding: utf-8 -*-
"""
生成单场洪水的最大水深包络图与最大流速包络图。

输入：
1. 预测水深 NPY，形状为 [T, H]；
2. 预测流速：优先读取 Vmag NPY；若未提供，则由预测 Vx/Vy NPY 计算；
3. HEC-RAS 真值 HDF，读取水深、Velocity X、Velocity Y；
4. HEC-RAS 网格几何，默认直接使用真值 HDF 内的 Geometry。

输出（每张图单独保存，不生成联合图）：
- depth_envelope_pred.*
- depth_envelope_true.*
- depth_envelope_diff.*
- speed_envelope_pred.*
- speed_envelope_true.*
- speed_envelope_diff.*
- depth_envelope_spatial_confusion.*
- speed_envelope_spatial_confusion.*
- envelope_metrics.csv / envelope_metrics.json
- confusion_matrix_metrics.csv / confusion_matrix_metrics.json
- 各包络及差值的 NPY 文件

核心定义：
- 最大水深包络：每个网格单元沿时间维取最大水深；
- 最大流速包络：先逐时计算 sqrt(Vx^2 + Vy^2)，再沿时间维取最大值；
- 差值：预测包络 - 真值包络；
- 水深连续指标同时给出全有效网格和包络联合湿区结果；
- 流速指标仅在包络联合湿区上计算，与项目现有评价逻辑一致。

运行示例：
python max_envelope_maps.py \
  --pred-depth D:/results/8_Y_pred_sanitized.npy \
  --pred-vmag D:/results/8_Vmag_pred.npy \
  --true-hdf D:/HDF/8.hdf \
  --output-dir D:/results/envelope_8 \
  --threshold 0.1 \
  --formats png,pdf

若没有预测 Vmag，而有预测 Vx/Vy：
python max_envelope_maps.py \
  --pred-depth D:/results/8_Y_pred_sanitized.npy \
  --pred-vx D:/results/8_Vx_pred.npy \
  --pred-vy D:/results/8_Vy_pred.npy \
  --true-hdf D:/HDF/8.hdf \
  --output-dir D:/results/envelope_8
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


DEFAULT_AREA_NAME = "Perimeter 1"


def _results_base(area_name: str) -> str:
    return "/".join([
        "Results", "Unsteady", "Output", "Output Blocks", "Base Output",
        "Unsteady Time Series", "2D Flow Areas", area_name,
    ])


def depth_hdf_path(area_name: str) -> str:
    return _results_base(area_name) + "/Cell Hydraulic Depth"


def vx_hdf_path(area_name: str) -> str:
    return _results_base(area_name) + "/Cell Velocity - Velocity X"


def vy_hdf_path(area_name: str) -> str:
    return _results_base(area_name) + "/Cell Velocity - Velocity Y"


def geom_fp_path(area_name: str) -> str:
    return f"Geometry/2D Flow Areas/{area_name}/FacePoints Coordinate"


def geom_cfi_path(area_name: str) -> str:
    return f"Geometry/2D Flow Areas/{area_name}/Cells FacePoint Indexes"


def geom_perimeter_path(area_name: str) -> str:
    return f"Geometry/2D Flow Areas/{area_name}/Perimeter"


@dataclass
class Geometry:
    perimeter_xy: np.ndarray
    cell_polys: List[np.ndarray]
    keep_idx: np.ndarray
    extent: Tuple[float, float, float, float]
    n_cells_total: int


@dataclass
class Background:
    data: Optional[np.ndarray] = None
    extent: Optional[Tuple[float, float, float, float]] = None
    alpha: float = 0.45


def set_paper_style(font_family: str = "Times New Roman", base_fontsize: int = 9) -> None:
    plt.rcParams["font.family"] = font_family
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = base_fontsize
    plt.rcParams["axes.labelsize"] = base_fontsize
    plt.rcParams["axes.titlesize"] = base_fontsize + 1
    plt.rcParams["xtick.labelsize"] = base_fontsize
    plt.rcParams["ytick.labelsize"] = base_fontsize


def load_background(path: Optional[str], alpha: float) -> Background:
    if not path:
        return Background(alpha=float(alpha))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"背景文件不存在：{path}")

    try:
        import rasterio
    except ImportError as exc:
        raise ImportError("使用 --background 时需要安装 rasterio。") from exc

    with rasterio.open(path) as ds:
        if ds.count >= 3:
            arr = np.transpose(ds.read([1, 2, 3]), (1, 2, 0)).astype(np.float32)
        else:
            arr = ds.read(1).astype(np.float32)
        bounds = ds.bounds

    finite = np.isfinite(arr)
    if np.any(finite):
        lo, hi = np.nanpercentile(arr[finite], [2.0, 98.0])
        if hi > lo:
            arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

    return Background(
        data=arr,
        extent=(float(bounds.left), float(bounds.right), float(bounds.bottom), float(bounds.top)),
        alpha=float(alpha),
    )


def load_geometry(hdf_path: str, area_name: str) -> Geometry:
    if not os.path.isfile(hdf_path):
        raise FileNotFoundError(f"几何 HDF 不存在：{hdf_path}")

    with h5py.File(hdf_path, "r") as h5:
        fp_path = geom_fp_path(area_name)
        cfi_path = geom_cfi_path(area_name)
        per_path = geom_perimeter_path(area_name)

        for path in (fp_path, cfi_path, per_path):
            if path not in h5:
                raise KeyError(f"HDF 中缺少几何数据集：{path}")

        face_points = np.asarray(h5[fp_path][:], dtype=np.float64)
        cell_indexes = np.asarray(h5[cfi_path][:], dtype=np.int64)
        perimeter_xy = np.asarray(h5[per_path][:], dtype=np.float64)

    if face_points.ndim != 2 or face_points.shape[1] < 2:
        raise ValueError(f"FacePoints Coordinate 形状异常：{face_points.shape}")
    if cell_indexes.ndim != 2:
        raise ValueError(f"Cells FacePoint Indexes 形状异常：{cell_indexes.shape}")

    n_face_points = int(face_points.shape[0])
    valid_raw = cell_indexes[cell_indexes >= 0]
    if valid_raw.size and int(valid_raw.max()) >= n_face_points:
        cell_indexes = cell_indexes - 1

    cell_polys: List[np.ndarray] = []
    keep_idx: List[int] = []

    for cell_id, row in enumerate(cell_indexes):
        idx = row[(row >= 0) & (row < n_face_points)]
        if idx.size < 3:
            continue

        # 保持原始顶点顺序，同时移除重复顶点。
        unique_idx = []
        seen = set()
        for value in idx.tolist():
            if value not in seen:
                unique_idx.append(value)
                seen.add(value)

        if len(unique_idx) < 3:
            continue

        cell_polys.append(face_points[np.asarray(unique_idx, dtype=np.int64), :2])
        keep_idx.append(cell_id)

    if not cell_polys:
        raise RuntimeError("未能从 HDF 构建有效二维网格多边形。")

    xmin = float(np.nanmin(perimeter_xy[:, 0]))
    xmax = float(np.nanmax(perimeter_xy[:, 0]))
    ymin = float(np.nanmin(perimeter_xy[:, 1]))
    ymax = float(np.nanmax(perimeter_xy[:, 1]))

    return Geometry(
        perimeter_xy=perimeter_xy[:, :2],
        cell_polys=cell_polys,
        keep_idx=np.asarray(keep_idx, dtype=np.int64),
        extent=(xmin, xmax, ymin, ymax),
        n_cells_total=int(cell_indexes.shape[0]),
    )


def read_hdf_field(hdf_path: str, dataset_path: str) -> np.ndarray:
    with h5py.File(hdf_path, "r") as h5:
        if dataset_path not in h5:
            raise KeyError(f"HDF 中缺少数据集：{dataset_path}")
        arr = np.asarray(h5[dataset_path][:], dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def load_npy_th(path: str, field_name: str) -> np.ndarray:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"{field_name} NPY 不存在：{path}")
    arr = np.asarray(np.load(path), dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{field_name} 必须为 [T,H] 二维数组，实际形状：{arr.shape}")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def ensure_time_cell(arr: np.ndarray, n_cells: int, field_name: str) -> np.ndarray:
    """统一为 [T,H]，允许输入误存为 [H,T]。"""
    if arr.ndim != 2:
        raise ValueError(f"{field_name} 不是二维数组：{arr.shape}")
    if arr.shape[1] == n_cells:
        return arr
    if arr.shape[0] == n_cells:
        print(f"[Warn] {field_name} 检测为 [H,T]，已自动转置为 [T,H]。")
        return arr.T
    raise ValueError(
        f"{field_name} 的网格维与 HEC-RAS 几何不一致：arr={arr.shape}, H={n_cells}"
    )


def infer_velocity_files(pred_depth_path: str) -> Dict[str, Optional[str]]:
    """按项目 infer.py 的命名规则自动寻找预测流速文件。"""
    suffixes = (
        "_Y_pred_sanitized.npy",
        "_Y_pred_raw.npy",
        "_Y_pred.npy",
        "_pred.npy",
    )
    stem = pred_depth_path
    for suffix in suffixes:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    else:
        stem = os.path.splitext(stem)[0]

    candidates = {
        "vmag": [stem + "_Vmag_pred.npy", stem + "_Vmag_pred_sanitized.npy"],
        "vx": [stem + "_Vx_pred.npy", stem + "_Vx_pred_sanitized.npy"],
        "vy": [stem + "_Vy_pred.npy", stem + "_Vy_pred_sanitized.npy"],
    }
    found: Dict[str, Optional[str]] = {}
    for key, paths in candidates.items():
        found[key] = next((p for p in paths if os.path.isfile(p)), None)
    return found


def choose_common_timesteps(arrays: Dict[str, np.ndarray], requested: int) -> int:
    lengths = {name: int(arr.shape[0]) for name, arr in arrays.items()}
    if requested > 0:
        too_short = {name: n for name, n in lengths.items() if n < requested}
        if too_short:
            raise ValueError(f"指定 timesteps={requested}，但以下数组不足：{too_short}")
        return int(requested)

    common_t = min(lengths.values())
    if len(set(lengths.values())) > 1:
        print(f"[Warn] 各数组时间步不一致：{lengths}；统一截取前 {common_t} 步。")
    return int(common_t)


def velocity_magnitude(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    return np.sqrt(vx * vx + vy * vy, dtype=np.float32)


def clean_nonnegative(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(arr, 0.0).astype(np.float32, copy=False)


def compute_metrics(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(true, dtype=np.float64)
    m = np.asarray(mask, dtype=bool) & np.isfinite(p) & np.isfinite(g)

    n = int(m.sum())
    if n == 0:
        return {"n_cells": 0, "RMSE": np.nan, "MAE": np.nan, "PCC": np.nan}

    p = p[m]
    g = g[m]
    diff = p - g
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))

    if n < 2 or np.std(p) < 1e-12 or np.std(g) < 1e-12:
        pcc = np.nan
    else:
        pcc = float(np.corrcoef(p, g)[0, 1])

    return {"n_cells": n, "RMSE": rmse, "MAE": mae, "PCC": pcc}


def flood_metrics(pred_depth: np.ndarray, true_depth: np.ndarray, threshold: float) -> Dict[str, float]:
    p = np.asarray(pred_depth) >= float(threshold)
    g = np.asarray(true_depth) >= float(threshold)

    tp = int(np.sum(p & g))
    fp = int(np.sum(p & (~g)))
    fn = int(np.sum((~p) & g))
    tn = int(np.sum((~p) & (~g)))

    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2.0 * precision * pod / (precision + pod) if (precision + pod) > 0 else 0.0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "F1": float(f1),
        "POD": float(pod),
        "FAR": float(far),
    }



def binary_confusion_metrics(
    pred_values: np.ndarray,
    true_values: np.ndarray,
    threshold: float,
    domain_mask: np.ndarray,
) -> Dict[str, float]:
    """在指定网格域内，将连续包络值按阈值二分类并计算混淆矩阵指标。"""
    pred = np.asarray(pred_values, dtype=np.float64)
    true = np.asarray(true_values, dtype=np.float64)
    domain = np.asarray(domain_mask, dtype=bool)

    valid = domain & np.isfinite(pred) & np.isfinite(true)
    pred_pos = pred[valid] >= float(threshold)
    true_pos = true[valid] >= float(threshold)

    tp = int(np.sum(pred_pos & true_pos))
    fp = int(np.sum(pred_pos & (~true_pos)))
    fn = int(np.sum((~pred_pos) & true_pos))
    tn = int(np.sum((~pred_pos) & (~true_pos)))
    n = int(valid.sum())

    accuracy = (tp + tn) / n if n > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        "n_cells": n,
        "threshold": float(threshold),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Accuracy": float(accuracy) if np.isfinite(accuracy) else np.nan,
        "Precision": float(precision),
        "Recall_POD": float(recall),
        "Specificity_TNR": float(specificity),
        "F1": float(f1),
        "FAR": float(far),
    }


def spatial_confusion_codes(
    pred_values: np.ndarray,
    true_values: np.ndarray,
    threshold: float,
    domain_mask: np.ndarray,
) -> np.ndarray:
    """
    为每个有效绘图网格生成空间混淆分类编码。

    编码：
    -1：不参与评价
     0：TN
     1：TP
     2：FP
     3：FN
    """
    pred = np.asarray(pred_values, dtype=np.float64)
    true = np.asarray(true_values, dtype=np.float64)
    domain = np.asarray(domain_mask, dtype=bool)

    if pred.shape != true.shape or pred.shape != domain.shape:
        raise ValueError(
            f"空间混淆分类数组形状不一致：pred={pred.shape}, "
            f"true={true.shape}, domain={domain.shape}"
        )

    valid = domain & np.isfinite(pred) & np.isfinite(true)
    pred_pos = pred >= float(threshold)
    true_pos = true >= float(threshold)

    codes = np.full(pred.shape, -1, dtype=np.int8)
    codes[valid & (~pred_pos) & (~true_pos)] = 0  # TN
    codes[valid & pred_pos & true_pos] = 1        # TP
    codes[valid & pred_pos & (~true_pos)] = 2     # FP
    codes[valid & (~pred_pos) & true_pos] = 3     # FN
    return codes


def save_spatial_confusion_map(
    *,
    codes: np.ndarray,
    metrics: Dict[str, float],
    geometry: Geometry,
    output_stem: str,
    formats: Sequence[str],
    dpi: int,
    figsize: Tuple[float, float],
    title: str,
    show_title: bool,
    background: Background,
    boundary_color: str,
    boundary_lw: float,
    rasterized: bool,
    positive_name: str,
    show_tn: bool,
    tp_color: str,
    fp_color: str,
    fn_color: str,
    tn_color: str,
    category_alpha: float,
) -> None:
    """把 TP、FP、FN、TN 分类直接绘制到 HEC-RAS 二维网格上。"""
    codes = np.asarray(codes, dtype=np.int8)
    if codes.ndim != 1 or codes.size != geometry.keep_idx.size:
        raise ValueError(
            f"空间混淆编码必须与有效网格一一对应："
            f"codes={codes.shape}, valid_H={geometry.keep_idx.size}"
        )

    fig, ax = plt.subplots(figsize=figsize, dpi=int(dpi))

    if background.data is not None:
        bg_extent = background.extent if background.extent is not None else geometry.extent
        ax.imshow(
            background.data,
            extent=bg_extent,
            origin="upper",
            alpha=float(background.alpha),
            zorder=0,
        )

    # 默认 TN 与评价域外网格透明，只突出 TP、FP、FN。
    facecolors = np.zeros((codes.size, 4), dtype=np.float32)
    color_table = {
        0: matplotlib.colors.to_rgba(tn_color, alpha=category_alpha if show_tn else 0.0),
        1: matplotlib.colors.to_rgba(tp_color, alpha=category_alpha),
        2: matplotlib.colors.to_rgba(fp_color, alpha=category_alpha),
        3: matplotlib.colors.to_rgba(fn_color, alpha=category_alpha),
    }
    for code, rgba in color_table.items():
        facecolors[codes == code] = rgba

    collection = PolyCollection(
        geometry.cell_polys,
        facecolors=facecolors,
        edgecolors="none",
        linewidths=0.0,
        antialiased=False,
        rasterized=bool(rasterized),
        zorder=2,
    )
    ax.add_collection(collection)

    draw_perimeter(ax, geometry.perimeter_xy, boundary_color, boundary_lw)
    ax.set_xlim(geometry.extent[0], geometry.extent[1])
    ax.set_ylim(geometry.extent[2], geometry.extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")

    count_line = (
        f"TN={int(metrics['TN'])}   FP={int(metrics['FP'])}   "
        f"FN={int(metrics['FN'])}   TP={int(metrics['TP'])}"
    )
    if show_title:
        ax.set_title(f"{title}\n{count_line}", pad=7)
    else:
        ax.set_title(count_line, pad=7)

    legend_handles = [
        Patch(facecolor=tp_color, edgecolor="none", label=f"TP (Correct {positive_name})"),
        Patch(facecolor=fp_color, edgecolor="none", label="FP (False alarm)"),
        Patch(facecolor=fn_color, edgecolor="none", label=f"FN (Missed {positive_name})"),
    ]
    if show_tn:
        legend_handles.append(
            Patch(facecolor=tn_color, edgecolor="none", label=f"TN (Correct non-{positive_name})")
        )

    legend = ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        borderpad=0.6,
        labelspacing=0.35,
        handlelength=1.3,
        handletextpad=0.5,
    )
    legend.get_frame().set_linewidth(0.7)

    fig.tight_layout(pad=0.4)
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        fig.savefig(f"{output_stem}.{fmt}", dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)

def positive_vmax(*arrays: np.ndarray, percentile: float) -> float:
    values = []
    for arr in arrays:
        a = np.asarray(arr, dtype=np.float64).reshape(-1)
        a = a[np.isfinite(a) & (a > 0)]
        if a.size:
            values.append(a)
    if not values:
        return 1.0
    merged = np.concatenate(values)
    return max(float(np.percentile(merged, percentile)), 1e-6)


def symmetric_vmax(arr: np.ndarray, percentile: float) -> float:
    a = np.abs(np.asarray(arr, dtype=np.float64).reshape(-1))
    a = a[np.isfinite(a)]
    if not a.size:
        return 1.0
    return max(float(np.percentile(a, percentile)), 1e-6)


def transparent_cmap(name: str):
    cmap = plt.get_cmap(name)
    try:
        cmap = cmap.copy()
        cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    except Exception:
        pass
    return cmap


def draw_perimeter(ax, perimeter_xy: np.ndarray, color: str, linewidth: float) -> None:
    p = np.asarray(perimeter_xy, dtype=np.float64)
    if not np.allclose(p[0], p[-1]):
        p = np.vstack([p, p[0]])
    ax.plot(
        p[:, 0], p[:, 1],
        color=color,
        linewidth=float(linewidth),
        linestyle="-",
        solid_capstyle="round",
        zorder=10,
    )


def save_single_map(
    *,
    values: np.ndarray,
    geometry: Geometry,
    output_stem: str,
    formats: Sequence[str],
    dpi: int,
    cmap_name: str,
    norm,
    colorbar_label: str,
    title: str,
    show_title: bool,
    background: Background,
    boundary_color: str,
    boundary_lw: float,
    figsize: Tuple[float, float],
    rasterized: bool,
) -> None:
    vals = np.asarray(values, dtype=np.float32)
    if vals.ndim != 1 or vals.size != geometry.keep_idx.size:
        raise ValueError(
            f"绘图值必须与有效网格一一对应：values={vals.shape}, valid_H={geometry.keep_idx.size}"
        )

    fig, ax = plt.subplots(figsize=figsize, dpi=int(dpi))

    if background.data is not None:
        bg_extent = background.extent if background.extent is not None else geometry.extent
        ax.imshow(
            background.data,
            extent=bg_extent,
            origin="upper",
            alpha=float(background.alpha),
            zorder=0,
        )

    cmap = transparent_cmap(cmap_name)
    masked_values = np.ma.masked_invalid(vals)
    collection = PolyCollection(
        geometry.cell_polys,
        array=masked_values,
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        linewidths=0.0,
        antialiased=False,
        rasterized=bool(rasterized),
        zorder=2,
    )
    ax.add_collection(collection)

    draw_perimeter(ax, geometry.perimeter_xy, boundary_color, boundary_lw)
    ax.set_xlim(geometry.extent[0], geometry.extent[1])
    ax.set_ylim(geometry.extent[2], geometry.extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    if show_title:
        ax.set_title(title, pad=6)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="4.8%", pad=0.10)
    colorbar = fig.colorbar(collection, cax=cax, orientation="horizontal")
    colorbar.set_label(colorbar_label, labelpad=4)
    colorbar.ax.tick_params(length=3, width=0.8)

    fig.tight_layout(pad=0.4)
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        fig.savefig(f"{output_stem}.{fmt}", dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def nan_for_display(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    out[~np.asarray(mask, dtype=bool)] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def main(args: argparse.Namespace) -> None:
    set_paper_style(args.font_family, args.font_size)
    os.makedirs(args.output_dir, exist_ok=True)

    geom_hdf = args.geom_hdf or args.true_hdf
    geometry = load_geometry(geom_hdf, args.area_name)
    print(
        f"[Geometry] total cells={geometry.n_cells_total}, "
        f"valid polygon cells={geometry.keep_idx.size}"
    )

    # -------------------------
    # 读取预测和真值
    # -------------------------
    pred_depth = ensure_time_cell(
        load_npy_th(args.pred_depth, "预测水深"),
        geometry.n_cells_total,
        "预测水深",
    )

    true_depth = ensure_time_cell(
        read_hdf_field(args.true_hdf, depth_hdf_path(args.area_name)),
        geometry.n_cells_total,
        "真值水深",
    )
    true_vx = ensure_time_cell(
        read_hdf_field(args.true_hdf, vx_hdf_path(args.area_name)),
        geometry.n_cells_total,
        "真值 Vx",
    )
    true_vy = ensure_time_cell(
        read_hdf_field(args.true_hdf, vy_hdf_path(args.area_name)),
        geometry.n_cells_total,
        "真值 Vy",
    )

    discovered = infer_velocity_files(args.pred_depth)
    pred_vmag_path = args.pred_vmag or discovered["vmag"]
    pred_vx_path = args.pred_vx or discovered["vx"]
    pred_vy_path = args.pred_vy or discovered["vy"]

    if pred_vmag_path:
        pred_vmag = ensure_time_cell(
            load_npy_th(pred_vmag_path, "预测流速大小"),
            geometry.n_cells_total,
            "预测流速大小",
        )
        velocity_source = pred_vmag_path
    elif pred_vx_path and pred_vy_path:
        pred_vx = ensure_time_cell(
            load_npy_th(pred_vx_path, "预测 Vx"),
            geometry.n_cells_total,
            "预测 Vx",
        )
        pred_vy = ensure_time_cell(
            load_npy_th(pred_vy_path, "预测 Vy"),
            geometry.n_cells_total,
            "预测 Vy",
        )
        pred_vmag = velocity_magnitude(pred_vx, pred_vy)
        velocity_source = f"Vx={pred_vx_path}; Vy={pred_vy_path}"
    else:
        raise FileNotFoundError(
            "未找到预测流速。请提供 --pred-vmag，或同时提供 --pred-vx 与 --pred-vy。\n"
            f"自动检索结果：{discovered}"
        )

    true_vmag = velocity_magnitude(true_vx, true_vy)

    arrays = {
        "pred_depth": pred_depth,
        "true_depth": true_depth,
        "pred_vmag": pred_vmag,
        "true_vmag": true_vmag,
    }
    timesteps = choose_common_timesteps(arrays, args.timesteps)
    print(f"[Time] 使用前 {timesteps} 个时间步构建包络。")

    pred_depth = clean_nonnegative(pred_depth[:timesteps])
    true_depth = clean_nonnegative(true_depth[:timesteps])
    pred_vmag = clean_nonnegative(pred_vmag[:timesteps])
    true_vmag = clean_nonnegative(true_vmag[:timesteps])

    # 与 infer.py 的物理掩膜一致：干区速度置 0。
    pred_vmag[pred_depth < args.threshold] = 0.0
    true_vmag[true_depth < args.threshold] = 0.0

    # -------------------------
    # 构建包络
    # -------------------------
    depth_pred_env = np.max(pred_depth, axis=0)
    depth_true_env = np.max(true_depth, axis=0)
    speed_pred_env = np.max(pred_vmag, axis=0)
    speed_true_env = np.max(true_vmag, axis=0)

    depth_diff_env = depth_pred_env - depth_true_env
    speed_diff_env = speed_pred_env - speed_true_env

    # 只保留能够绘制的有效 HEC-RAS 网格；指标与图件使用完全相同的网格。
    idx = geometry.keep_idx
    depth_pred_v = depth_pred_env[idx]
    depth_true_v = depth_true_env[idx]
    depth_diff_v = depth_diff_env[idx]
    speed_pred_v = speed_pred_env[idx]
    speed_true_v = speed_true_env[idx]
    speed_diff_v = speed_diff_env[idx]

    valid_all = (
        np.isfinite(depth_pred_v)
        & np.isfinite(depth_true_v)
        & np.isfinite(speed_pred_v)
        & np.isfinite(speed_true_v)
    )
    wet_union = valid_all & (
        (depth_pred_v >= args.threshold) | (depth_true_v >= args.threshold)
    )

    # -------------------------
    # 指标
    # -------------------------
    depth_all_metrics = compute_metrics(depth_pred_v, depth_true_v, valid_all)
    depth_wet_metrics = compute_metrics(depth_pred_v, depth_true_v, wet_union)
    speed_wet_metrics = compute_metrics(speed_pred_v, speed_true_v, wet_union)
    classification = flood_metrics(
        depth_pred_v[valid_all], depth_true_v[valid_all], args.threshold
    )

    # 最大水深包络：在全部有效网格上按水深阈值划分淹没/未淹没。
    depth_confusion = binary_confusion_metrics(
        depth_pred_v,
        depth_true_v,
        threshold=args.threshold,
        domain_mask=valid_all,
    )

    # 最大流速包络：只在水深包络联合湿区内按流速阈值划分低速/高速。
    # 这样可避免大量永久干区 TN 主导混淆矩阵。
    speed_confusion = binary_confusion_metrics(
        speed_pred_v,
        speed_true_v,
        threshold=args.speed_threshold,
        domain_mask=wet_union,
    )

    # 空间混淆分类编码，与指标使用完全相同的网格和阈值。
    depth_confusion_codes = spatial_confusion_codes(
        depth_pred_v,
        depth_true_v,
        threshold=args.threshold,
        domain_mask=valid_all,
    )
    speed_confusion_codes = spatial_confusion_codes(
        speed_pred_v,
        speed_true_v,
        threshold=args.speed_threshold,
        domain_mask=wet_union,
    )

    metrics_rows = [
        {
            "field": "depth_envelope",
            "evaluation_domain": "all_valid_grid_cells",
            **depth_all_metrics,
            "F1": classification["F1"],
            "POD": classification["POD"],
            "FAR": classification["FAR"],
        },
        {
            "field": "depth_envelope",
            "evaluation_domain": "wet_union_grid_cells",
            **depth_wet_metrics,
            "F1": classification["F1"],
            "POD": classification["POD"],
            "FAR": classification["FAR"],
        },
        {
            "field": "speed_envelope",
            "evaluation_domain": "wet_union_grid_cells",
            **speed_wet_metrics,
            "F1": np.nan,
            "POD": np.nan,
            "FAR": np.nan,
        },
    ]

    metrics_csv = os.path.join(args.output_dir, "envelope_metrics.csv")
    pd.DataFrame(metrics_rows).to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    confusion_rows = [
        {
            "field": "depth_envelope",
            "evaluation_domain": "all_valid_grid_cells",
            "negative_class": f"depth < {args.threshold:g} m",
            "positive_class": f"depth >= {args.threshold:g} m",
            **depth_confusion,
        },
        {
            "field": "speed_envelope",
            "evaluation_domain": "depth_envelope_wet_union",
            "negative_class": f"speed < {args.speed_threshold:g} m/s",
            "positive_class": f"speed >= {args.speed_threshold:g} m/s",
            **speed_confusion,
        },
    ]
    confusion_csv = os.path.join(args.output_dir, "confusion_matrix_metrics.csv")
    pd.DataFrame(confusion_rows).to_csv(
        confusion_csv, index=False, encoding="utf-8-sig"
    )

    summary = {
        "inputs": {
            "pred_depth": os.path.abspath(args.pred_depth),
            "pred_velocity": velocity_source,
            "true_hdf": os.path.abspath(args.true_hdf),
            "geom_hdf": os.path.abspath(geom_hdf),
            "area_name": args.area_name,
        },
        "settings": {
            "timesteps": timesteps,
            "depth_threshold_m": float(args.threshold),
            "speed_threshold_mps": float(args.speed_threshold),
            "valid_polygon_cells": int(geometry.keep_idx.size),
            "wet_union_cells": int(wet_union.sum()),
        },
        "depth_envelope_all_grid": depth_all_metrics,
        "depth_envelope_wet_union": depth_wet_metrics,
        "depth_inundation": classification,
        "depth_envelope_confusion": depth_confusion,
        "speed_envelope_wet_union": speed_wet_metrics,
        "speed_envelope_confusion": speed_confusion,
        "maxima": {
            "pred_depth_max_m": float(np.nanmax(depth_pred_v)),
            "true_depth_max_m": float(np.nanmax(depth_true_v)),
            "pred_speed_max_mps": float(np.nanmax(speed_pred_v)),
            "true_speed_max_mps": float(np.nanmax(speed_true_v)),
        },
    }
    metrics_json = os.path.join(args.output_dir, "envelope_metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as file:
        json.dump(json_safe(summary), file, ensure_ascii=False, indent=2)

    confusion_json = os.path.join(args.output_dir, "confusion_matrix_metrics.json")
    with open(confusion_json, "w", encoding="utf-8") as file:
        json.dump(
            json_safe({
                "depth_envelope": confusion_rows[0],
                "speed_envelope": confusion_rows[1],
            }),
            file,
            ensure_ascii=False,
            indent=2,
        )

    # 保存原始全网格包络，保持 H 维与模型输出一致。
    np.save(os.path.join(args.output_dir, "depth_envelope_pred.npy"), depth_pred_env.astype(np.float32))
    np.save(os.path.join(args.output_dir, "depth_envelope_true.npy"), depth_true_env.astype(np.float32))
    np.save(os.path.join(args.output_dir, "depth_envelope_diff.npy"), depth_diff_env.astype(np.float32))
    np.save(os.path.join(args.output_dir, "speed_envelope_pred.npy"), speed_pred_env.astype(np.float32))
    np.save(os.path.join(args.output_dir, "speed_envelope_true.npy"), speed_true_env.astype(np.float32))
    np.save(os.path.join(args.output_dir, "speed_envelope_diff.npy"), speed_diff_env.astype(np.float32))


    # 保存空间混淆分类编码。全网格编码：-1 不参与评价，0 TN，1 TP，2 FP，3 FN。
    depth_confusion_full = np.full(geometry.n_cells_total, -1, dtype=np.int8)
    speed_confusion_full = np.full(geometry.n_cells_total, -1, dtype=np.int8)
    depth_confusion_full[idx] = depth_confusion_codes
    speed_confusion_full[idx] = speed_confusion_codes
    np.save(
        os.path.join(args.output_dir, "depth_envelope_spatial_confusion_code.npy"),
        depth_confusion_full,
    )
    np.save(
        os.path.join(args.output_dir, "speed_envelope_spatial_confusion_code.npy"),
        speed_confusion_full,
    )

    # -------------------------
    # 单图输出
    # -------------------------
    formats = tuple(x.strip() for x in args.formats.split(",") if x.strip())
    background = load_background(args.background, args.background_alpha)

    # ---------------------------------------------------------
    # 统一色棒范围
    # 预测图和真值图对同一变量严格共用同一组上下限。
    # 若命令行未指定 vmax，则根据预测值和真值共同自动计算。
    # 差值图采用以 0 为中心的对称色棒。
    # ---------------------------------------------------------
    depth_vmin = float(args.depth_vmin)
    speed_vmin = float(args.speed_vmin)

    depth_vmax = (
        float(args.depth_vmax)
        if args.depth_vmax is not None
        else positive_vmax(
            depth_pred_v[wet_union],
            depth_true_v[wet_union],
            percentile=args.value_percentile,
        )
    )
    speed_vmax = (
        float(args.speed_vmax)
        if args.speed_vmax is not None
        else positive_vmax(
            speed_pred_v[wet_union],
            speed_true_v[wet_union],
            percentile=args.value_percentile,
        )
    )
    depth_diff_vmax = (
        float(args.depth_diff_vmax)
        if args.depth_diff_vmax is not None
        else symmetric_vmax(depth_diff_v[wet_union], args.diff_percentile)
    )
    speed_diff_vmax = (
        float(args.speed_diff_vmax)
        if args.speed_diff_vmax is not None
        else symmetric_vmax(speed_diff_v[wet_union], args.diff_percentile)
    )

    if not depth_vmax > depth_vmin:
        raise ValueError(
            f"水深色棒范围无效：depth_vmin={depth_vmin}, depth_vmax={depth_vmax}"
        )
    if not speed_vmax > speed_vmin:
        raise ValueError(
            f"流速色棒范围无效：speed_vmin={speed_vmin}, speed_vmax={speed_vmax}"
        )
    if not depth_diff_vmax > 0:
        raise ValueError(f"depth_diff_vmax 必须大于 0，当前为 {depth_diff_vmax}")
    if not speed_diff_vmax > 0:
        raise ValueError(f"speed_diff_vmax 必须大于 0，当前为 {speed_diff_vmax}")

    print(
        "[Colorbar] 水深预测/真值统一范围："
        f"[{depth_vmin:.6g}, {depth_vmax:.6g}] m"
    )
    print(
        "[Colorbar] 流速预测/真值统一范围："
        f"[{speed_vmin:.6g}, {speed_vmax:.6g}] m/s"
    )
    print(
        "[Colorbar] 水深差值统一范围："
        f"[-{depth_diff_vmax:.6g}, {depth_diff_vmax:.6g}] m"
    )
    print(
        "[Colorbar] 流速差值统一范围："
        f"[-{speed_diff_vmax:.6g}, {speed_diff_vmax:.6g}] m/s"
    )

    depth_show_mask = valid_all & (
        (depth_pred_v >= args.min_show_depth) | (depth_true_v >= args.min_show_depth)
    )
    speed_show_mask = wet_union & (
        (speed_pred_v >= args.min_show_speed) | (speed_true_v >= args.min_show_speed)
    )

    # 预测图和真值图使用相同色阶，确保单图之间仍可直接比较。
    common_kwargs = dict(
        geometry=geometry,
        formats=formats,
        dpi=args.dpi,
        show_title=args.show_title,
        background=background,
        boundary_color=args.boundary_color,
        boundary_lw=args.boundary_lw,
        figsize=(args.figure_width, args.figure_height),
        rasterized=args.rasterized,
    )

    save_single_map(
        values=nan_for_display(depth_pred_v, depth_pred_v >= args.min_show_depth),
        output_stem=os.path.join(args.output_dir, "depth_envelope_pred"),
        cmap_name=args.depth_cmap,
        norm=Normalize(vmin=depth_vmin, vmax=depth_vmax),
        colorbar_label="Maximum water depth (m)",
        title="Predicted maximum water depth envelope",
        **common_kwargs,
    )
    save_single_map(
        values=nan_for_display(depth_true_v, depth_true_v >= args.min_show_depth),
        output_stem=os.path.join(args.output_dir, "depth_envelope_true"),
        cmap_name=args.depth_cmap,
        norm=Normalize(vmin=depth_vmin, vmax=depth_vmax),
        colorbar_label="Maximum water depth (m)",
        title="Reference maximum water depth envelope",
        **common_kwargs,
    )
    save_single_map(
        values=nan_for_display(depth_diff_v, depth_show_mask),
        output_stem=os.path.join(args.output_dir, "depth_envelope_diff"),
        cmap_name=args.diff_cmap,
        norm=TwoSlopeNorm(vmin=-depth_diff_vmax, vcenter=0.0, vmax=depth_diff_vmax),
        colorbar_label="Prediction - reference (m)",
        title="Maximum water depth envelope error",
        **common_kwargs,
    )

    save_single_map(
        values=nan_for_display(speed_pred_v, wet_union & (speed_pred_v >= args.min_show_speed)),
        output_stem=os.path.join(args.output_dir, "speed_envelope_pred"),
        cmap_name=args.speed_cmap,
        norm=Normalize(vmin=speed_vmin, vmax=speed_vmax),
        colorbar_label="Maximum velocity magnitude (m/s)",
        title="Predicted maximum velocity envelope",
        **common_kwargs,
    )
    save_single_map(
        values=nan_for_display(speed_true_v, wet_union & (speed_true_v >= args.min_show_speed)),
        output_stem=os.path.join(args.output_dir, "speed_envelope_true"),
        cmap_name=args.speed_cmap,
        norm=Normalize(vmin=speed_vmin, vmax=speed_vmax),
        colorbar_label="Maximum velocity magnitude (m/s)",
        title="Reference maximum velocity envelope",
        **common_kwargs,
    )
    save_single_map(
        values=nan_for_display(speed_diff_v, speed_show_mask),
        output_stem=os.path.join(args.output_dir, "speed_envelope_diff"),
        cmap_name=args.diff_cmap,
        norm=TwoSlopeNorm(vmin=-speed_diff_vmax, vcenter=0.0, vmax=speed_diff_vmax),
        colorbar_label="Prediction - reference (m/s)",
        title="Maximum velocity envelope error",
        **common_kwargs,
    )

    # 空间混淆矩阵图：把每个网格单元的 TP/FP/FN/TN 分类画回二维网格。
    save_spatial_confusion_map(
        codes=depth_confusion_codes,
        metrics=depth_confusion,
        geometry=geometry,
        output_stem=os.path.join(args.output_dir, "depth_envelope_spatial_confusion"),
        formats=formats,
        dpi=args.dpi,
        figsize=(args.figure_width, args.figure_height),
        title=f"Depth-envelope spatial confusion map (threshold = {args.threshold:g} m)",
        show_title=args.show_title,
        background=background,
        boundary_color=args.boundary_color,
        boundary_lw=args.boundary_lw,
        rasterized=args.rasterized,
        positive_name="inundation",
        show_tn=args.show_tn,
        tp_color=args.tp_color,
        fp_color=args.fp_color,
        fn_color=args.fn_color,
        tn_color=args.tn_color,
        category_alpha=args.confusion_alpha,
    )
    save_spatial_confusion_map(
        codes=speed_confusion_codes,
        metrics=speed_confusion,
        geometry=geometry,
        output_stem=os.path.join(args.output_dir, "speed_envelope_spatial_confusion"),
        formats=formats,
        dpi=args.dpi,
        figsize=(args.figure_width, args.figure_height),
        title=(
            f"Speed-envelope spatial confusion map "
            f"(threshold = {args.speed_threshold:g} m/s)"
        ),
        show_title=args.show_title,
        background=background,
        boundary_color=args.boundary_color,
        boundary_lw=args.boundary_lw,
        rasterized=args.rasterized,
        positive_name="high velocity",
        show_tn=args.show_tn,
        tp_color=args.tp_color,
        fp_color=args.fp_color,
        fn_color=args.fn_color,
        tn_color=args.tn_color,
        category_alpha=args.confusion_alpha,
    )

    print("\n[Metrics] 最大水深包络：全有效网格")
    print(
        f"  RMSE={depth_all_metrics['RMSE']:.6f} m, "
        f"MAE={depth_all_metrics['MAE']:.6f} m, "
        f"PCC={depth_all_metrics['PCC']:.6f}"
    )
    print("[Metrics] 最大水深包络：联合湿区")
    print(
        f"  RMSE={depth_wet_metrics['RMSE']:.6f} m, "
        f"MAE={depth_wet_metrics['MAE']:.6f} m, "
        f"PCC={depth_wet_metrics['PCC']:.6f}, "
        f"F1={classification['F1']:.6f}, "
        f"POD={classification['POD']:.6f}, "
        f"FAR={classification['FAR']:.6f}"
    )
    print("[Metrics] 最大流速包络：联合湿区")
    print(
        f"  RMSE={speed_wet_metrics['RMSE']:.6f} m/s, "
        f"MAE={speed_wet_metrics['MAE']:.6f} m/s, "
        f"PCC={speed_wet_metrics['PCC']:.6f}"
    )
    print("[Confusion] 最大水深包络：全有效网格")
    print(
        f"  TN={depth_confusion['TN']}, FP={depth_confusion['FP']}, "
        f"FN={depth_confusion['FN']}, TP={depth_confusion['TP']}, "
        f"Accuracy={depth_confusion['Accuracy']:.6f}, "
        f"F1={depth_confusion['F1']:.6f}"
    )
    print("[Confusion] 最大流速包络：水深包络联合湿区")
    print(
        f"  TN={speed_confusion['TN']}, FP={speed_confusion['FP']}, "
        f"FN={speed_confusion['FN']}, TP={speed_confusion['TP']}, "
        f"Accuracy={speed_confusion['Accuracy']:.6f}, "
        f"F1={speed_confusion['F1']:.6f}"
    )
    print(f"\n[Done] 图件与指标已保存到：{os.path.abspath(args.output_dir)}")
    print(f"[Done] envelope metrics CSV: {metrics_csv}")
    print(f"[Done] envelope metrics JSON: {metrics_json}")
    print(f"[Done] confusion metrics CSV: {confusion_csv}")
    print(f"[Done] confusion metrics JSON: {confusion_json}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="生成最大水深/流速包络单图、差值图、网格指标及空间混淆分类图。"
    )
    parser.add_argument("--pred-depth", required=True, help="预测水深 NPY，[T,H]")
    parser.add_argument("--pred-vmag", default="", help="预测流速大小 NPY，[T,H]；优先使用")
    parser.add_argument("--pred-vx", default="", help="预测 Vx NPY，[T,H]")
    parser.add_argument("--pred-vy", default="", help="预测 Vy NPY，[T,H]")
    parser.add_argument("--true-hdf", required=True, help="HEC-RAS 真值 HDF")
    parser.add_argument("--geom-hdf", default="", help="网格几何 HDF；为空时使用 true-hdf")
    parser.add_argument("--area-name", default=DEFAULT_AREA_NAME, help="HEC-RAS 2D Flow Area 名称")
    parser.add_argument("--output-dir", required=True, help="输出目录")

    parser.add_argument("--timesteps", type=int, default=0,
                        help="参与包络计算的前 T 步；0 表示自动取各数组共同时间长度")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="湿区水深阈值，单位 m；用于 F1/POD/FAR 和流速评价掩膜")
    parser.add_argument("--min-show-depth", type=float, default=0.1,
                        help="水深图最小显示值，单位 m")
    parser.add_argument("--min-show-speed", type=float, default=0.001,
                        help="流速图最小显示值，单位 m/s")
    parser.add_argument("--speed-threshold", type=float, default=0.05,
                        help="最大流速包络混淆矩阵的高速阈值，单位 m/s")

    parser.add_argument("--formats", default="png", help="输出格式，例如 png 或 png,pdf,svg")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--figure-width", type=float, default=7.0)
    parser.add_argument("--figure-height", type=float, default=7.0)
    parser.add_argument("--show-title", action="store_true", help="在单图内部显示英文标题")
    parser.add_argument("--font-family", default="Times New Roman")
    parser.add_argument("--font-size", type=int, default=9)

    parser.add_argument("--depth-cmap", default="Blues")
    parser.add_argument("--speed-cmap", default="viridis")
    parser.add_argument("--diff-cmap", default="RdBu_r")

    # 同类图统一色棒范围。vmax 留空时由预测和真值共同自动计算。
    parser.add_argument(
        "--depth-vmin", type=float, default=0.0,
        help="水深预测图和真值图共用的色棒下限，单位 m",
    )
    parser.add_argument(
        "--depth-vmax", type=float, default=None,
        help="水深预测图和真值图共用的色棒上限，单位 m；默认自动计算",
    )
    parser.add_argument(
        "--speed-vmin", type=float, default=0.0,
        help="流速预测图和真值图共用的色棒下限，单位 m/s",
    )
    parser.add_argument(
        "--speed-vmax", type=float, default=None,
        help="流速预测图和真值图共用的色棒上限，单位 m/s；默认自动计算",
    )
    parser.add_argument(
        "--depth-diff-vmax", type=float, default=None,
        help="水深差值图对称色棒绝对值上限，单位 m；默认自动计算",
    )
    parser.add_argument(
        "--speed-diff-vmax", type=float, default=None,
        help="流速差值图对称色棒绝对值上限，单位 m/s；默认自动计算",
    )
    parser.add_argument("--show-tn", action="store_true",
                        help="空间混淆图中显示 TN 网格；默认 TN 透明")
    parser.add_argument("--tp-color", default="#2C7FB8", help="TP 网格颜色")
    parser.add_argument("--fp-color", default="#D73027", help="FP 网格颜色")
    parser.add_argument("--fn-color", default="#FFD43B", help="FN 网格颜色")
    parser.add_argument("--tn-color", default="#D9D9D9", help="TN 网格颜色")
    parser.add_argument("--confusion-alpha", type=float, default=0.90,
                        help="空间混淆分类网格透明度")
    parser.add_argument("--value-percentile", type=float, default=100.0,
                        help="预测/真值公共色标上限分位数，默认100即真实最大值")
    parser.add_argument("--diff-percentile", type=float, default=99.0,
                        help="差值图对称色标的绝对值分位数")

    parser.add_argument("--boundary-color", default="red")
    parser.add_argument("--boundary-lw", type=float, default=1.4)
    parser.add_argument("--background", default="", help="可选 GeoTIFF 背景底图")
    parser.add_argument("--background-alpha", type=float, default=0.45)
    parser.add_argument("--rasterized", action="store_true",
                        help="将网格 PolyCollection 栅格化，显著减小 PDF/SVG 文件体积")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())

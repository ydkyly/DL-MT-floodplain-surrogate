# -*- coding: utf-8 -*-
"""
postviz_poly_paper.py  (MODULE-ONLY, compatible with flood_main.py _add_viz_args)

用法（在 flood_main.py 中）：
    from postviz_poly_paper import viz_pred
    viz_pred(args)

特点：
- 不栅格化：使用 HEC-RAS 2D Cell polygons (PolyCollection)，PDF/SVG 输出为矢量
- 论文排版：Times New Roman、三联图 (a)(b)(c)、共享色条、默认不显示坐标轴
- inundation confusion：仅绘制 TP/FP/FN，TN 完全不绘制（无颜色/无透明）
- 背景底图：GeoTIFF 自动 extent；PNG/JPG 无地理信息会警告（建议用 GeoTIFF）
"""

from __future__ import annotations

import os
import glob
import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable



# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("postviz_poly_paper")


# ============================================================
# HDF paths（按你贴出来的结构）
# ============================================================
DEFAULT_2D_AREA_NAME = "Perimeter 1"

def _results_base(area_name: str) -> str:
    return "/".join([
        "Results", "Unsteady", "Output", "Output Blocks", "Base Output",
        "Unsteady Time Series", "2D Flow Areas", area_name
    ])

def depth_hdf_path(area_name: str) -> str:
    return _results_base(area_name) + "/Cell Hydraulic Depth"

def vx_hdf_path(area_name: str) -> str:
    return _results_base(area_name) + "/Cell Velocity - Velocity X"

def vy_hdf_path(area_name: str) -> str:
    return _results_base(area_name) + "/Cell Velocity - Velocity Y"

def geom_fp_path(area_name: str) -> str:
    return "/".join(["Geometry", "2D Flow Areas", area_name, "FacePoints Coordinate"])

def geom_cfi_path(area_name: str) -> str:
    return "/".join(["Geometry", "2D Flow Areas", area_name, "Cells FacePoint Indexes"])

def geom_perimeter_path(area_name: str) -> str:
    return "/".join(["Geometry", "2D Flow Areas", area_name, "Perimeter"])


# ============================================================
# 论文排版风格
# ============================================================
def set_paper_style(
    *,
    font_family: str = "Times New Roman",
    base_fontsize: int = 8,
    title_fontsize: int = 10,
    linewidth: float = 1.2,
):
    plt.rcParams["font.family"] = font_family
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False

    plt.rcParams["font.size"] = base_fontsize
    plt.rcParams["axes.titlesize"] = title_fontsize
    plt.rcParams["axes.labelsize"] = base_fontsize
    plt.rcParams["xtick.labelsize"] = base_fontsize
    plt.rcParams["ytick.labelsize"] = base_fontsize
    plt.rcParams["legend.fontsize"] = base_fontsize
    plt.rcParams["lines.linewidth"] = linewidth


# ============================================================
# 背景底图（遥感）加载
# ============================================================
@dataclass
class Background:
    data: Optional[np.ndarray] = None
    extent: Optional[Sequence[float]] = None  # [xmin, xmax, ymin, ymax]
    alpha: float = 0.45
    is_geotiff: bool = False
    crs: Optional[str] = None

_BG = Background()

def _percentile_stretch_rgb(arr: np.ndarray, p_low=2.0, p_high=98.0) -> np.ndarray:
    a = arr.astype(np.float32)
    out = np.zeros_like(a, dtype=np.float32)
    for c in range(3):
        lo, hi = np.nanpercentile(a[..., c], [p_low, p_high])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            out[..., c] = np.clip(a[..., c], 0, 1)
        else:
            out[..., c] = np.clip((a[..., c] - lo) / (hi - lo), 0, 1)
    return out

def load_background(
    path: Optional[str],
    *,
    alpha: float,
    stretch: bool = True,
    p_low: float = 2.0,
    p_high: float = 98.0,
):
    global _BG
    _BG = Background(alpha=float(alpha))

    if not path:
        logger.info("Background: OFF")
        return

    if not os.path.isfile(path):
        logger.warning(f"Background file not found: {path}")
        return

    ext = os.path.splitext(path)[1].lower()

    try:
        if ext in (".tif", ".tiff"):
            import rasterio
            with rasterio.open(path) as ds:
                if ds.count >= 3:
                    img = ds.read([1, 2, 3])
                    arr = np.transpose(img, (1, 2, 0)).astype(np.float32)
                else:
                    arr = ds.read(1).astype(np.float32)
                left, bottom, right, top = ds.bounds
                _BG.extent = [float(left), float(right), float(bottom), float(top)]
                _BG.is_geotiff = True
                _BG.crs = str(ds.crs) if ds.crs is not None else None

            if arr.ndim == 3:
                arr = arr - np.nanmin(arr)
                denom = np.nanmax(arr) - np.nanmin(arr) + 1e-12
                arr = arr / denom
                if stretch:
                    arr = _percentile_stretch_rgb(arr, p_low, p_high)

            _BG.data = arr
            logger.info(f"Background GeoTIFF loaded. extent={_BG.extent}, crs={_BG.crs}")

        else:
            logger.warning(f"Background is not GeoTIFF ({ext}). For accurate overlay, please use GeoTIFF.")
            _BG.data = None
            _BG.extent = None

    except Exception as e:
        logger.error(f"Load background failed: {e}")
        _BG = Background(alpha=float(alpha))

def _draw_background(ax, geom_extent: Sequence[float]):
    if _BG.data is None:
        return
    if _BG.extent is None:
        ax.imshow(_BG.data, extent=geom_extent, origin="upper", alpha=_BG.alpha, zorder=0)
        return
    ax.imshow(_BG.data, extent=_BG.extent, origin="upper", alpha=_BG.alpha, zorder=0)


# ============================================================
# Geometry：不栅格化关键（cell polygons + perimeter）
# ============================================================
_GEOM_CACHE: Dict[str, Dict[str, object]] = {}

def load_hecras_cell_polys(
    geom_hdf: str,
    *,
    area_name: str,
    cache: bool = True,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, Sequence[float]]:
    """
    返回：
      perimeter_xy: (N,2)
      cell_polys_valid: List[Mi,2]（只保留 >=3 点的合法 polygon）
      keep_idx: (H_valid,) 对应原始 cell 索引
      extent: [xmin,xmax,ymin,ymax]
    """
    key = f"{geom_hdf}::{area_name}"
    if cache and key in _GEOM_CACHE:
        d = _GEOM_CACHE[key]
        return d["perimeter_xy"], d["cell_polys"], d["keep_idx"], d["extent"]

    if not os.path.isfile(geom_hdf):
        raise FileNotFoundError(f"geom_hdf not found: {geom_hdf}")

    per_path = geom_perimeter_path(area_name)
    fp_path = geom_fp_path(area_name)
    cfi_path = geom_cfi_path(area_name)

    with h5py.File(geom_hdf, "r") as h5:
        if per_path not in h5:
            raise KeyError(f"Perimeter dataset not found: {per_path}")
        if fp_path not in h5 or cfi_path not in h5:
            raise KeyError(f"FacePoints/Cells indexes not found: {fp_path} / {cfi_path}")

        perimeter_xy = h5[per_path][:].astype(np.float32)
        fp = h5[fp_path][:].astype(np.float32)      # (Nfp,2)
        cfi = h5[cfi_path][:].astype(np.int64)      # (H,K)

    nfp = fp.shape[0]
    mx = int(np.max(cfi))
    if mx >= nfp:  # 兼容 1-based index
        cfi = cfi - 1

    H = cfi.shape[0]
    polys: List[np.ndarray] = []
    keep: List[int] = []

    for i in range(H):
        idx = cfi[i]
        idx = idx[(idx >= 0) & (idx < nfp)]
        if idx.size < 3:
            continue
        seen = set()
        uniq = []
        for j in idx.tolist():
            if j not in seen:
                uniq.append(j)
                seen.add(j)
        if len(uniq) < 3:
            continue
        poly = fp[np.array(uniq, dtype=np.int64)].astype(np.float32)
        polys.append(poly)
        keep.append(i)

    keep_idx = np.asarray(keep, dtype=np.int64)

    xmin, xmax = float(np.min(perimeter_xy[:, 0])), float(np.max(perimeter_xy[:, 0]))
    ymin, ymax = float(np.min(perimeter_xy[:, 1])), float(np.max(perimeter_xy[:, 1]))
    extent = [xmin, xmax, ymin, ymax]

    if cache:
        _GEOM_CACHE[key] = dict(perimeter_xy=perimeter_xy, cell_polys=polys, keep_idx=keep_idx, extent=extent)

    return perimeter_xy, polys, keep_idx, extent

def draw_perimeter(ax, perimeter_xy: np.ndarray, *, color="red", lw=1.2, zorder=10):
    p = np.asarray(perimeter_xy, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 3:
        return
    if not np.allclose(p[0], p[-1]):
        p = np.vstack([p, p[0]])
    ax.plot(
        p[:, 0], p[:, 1],
        color=color, linewidth=float(lw),
        linestyle=(0, (6, 3)),
        solid_capstyle="round",
        zorder=int(zorder)
    )


# ============================================================
# 工具函数：色阶/清洗/保存
# ============================================================
def metrics_event_flood(
    depth_pred: np.ndarray,
    depth_true: np.ndarray,
    *,
    thr: float,
    # |v|
    vmag_pred: Optional[np.ndarray] = None,
    vmag_true: Optional[np.ndarray] = None,
    # components
    vx_pred: Optional[np.ndarray] = None,
    vx_true: Optional[np.ndarray] = None,
    vy_pred: Optional[np.ndarray] = None,
    vy_true: Optional[np.ndarray] = None,
    # mask
    wet_union_all: Optional[np.ndarray] = None,   # (T,H) bool
) -> Dict[str, Any]:
    """
    整场洪水（T×H）汇总指标：
      - depth: RMSE/MAE/PCC + F1/POD/FAR（thr 二值化）
      - vmag : RMSE/MAE/PCC（默认仅在 wet_union_all 内评估）
      - Vx/Vy: RMSE/MAE/PCC（默认仅在 wet_union_all 内评估）
    """
    out: Dict[str, Any] = {}

    dp = np.asarray(depth_pred, dtype=np.float32)
    dt = np.asarray(depth_true, dtype=np.float32)
    assert dp.shape == dt.shape, f"depth_pred/true shape mismatch: {dp.shape} vs {dt.shape}"

    # ---- depth ----
    rmse, mae, pcc = _rmse_mae_pcc(dp, dt)
    f1, pod, far = _f1_pod_far(dp, dt, thr)
    out.update(dict(
        depth_RMSE=float(rmse),
        depth_MAE=float(mae),
        depth_PCC=float(pcc) if np.isfinite(pcc) else np.nan,
        depth_F1=float(f1),
        depth_POD=float(pod),
        depth_FAR=float(far),
    ))

    # ---- mask prep ----
    m = None
    if wet_union_all is not None:
        m = np.asarray(wet_union_all, dtype=bool)
        if m.shape != dp.shape:
            raise ValueError(f"wet_union_all shape mismatch: {m.shape} vs data {dp.shape}")
        out["eval_wet_cells"] = int(np.sum(m))
    else:
        out["eval_wet_cells"] = int(np.prod(dp.shape))

    def _masked_eval(a: np.ndarray, b: np.ndarray):
        """按 m 掩膜评估（m 为 None 则全域评估）"""
        if m is None:
            return a, b
        return np.where(m, a, np.nan), np.where(m, b, np.nan)

    # ---- |v| ----
    if vmag_pred is not None and vmag_true is not None:
        vp = np.asarray(vmag_pred, dtype=np.float32)
        vt = np.asarray(vmag_true, dtype=np.float32)
        assert vp.shape == vt.shape == dp.shape, f"vmag shape mismatch: {vp.shape} vs {vt.shape} vs {dp.shape}"
        vp_eval, vt_eval = _masked_eval(vp, vt)
        vrmse, vmae, vpcc = _rmse_mae_pcc(vp_eval, vt_eval)
        out.update(dict(
            vmag_RMSE=float(vrmse),
            vmag_MAE=float(vmae),
            vmag_PCC=float(vpcc) if np.isfinite(vpcc) else np.nan,
        ))

    # ---- Vx ----
    if vx_pred is not None and vx_true is not None:
        vxP = np.asarray(vx_pred, dtype=np.float32)
        vxT = np.asarray(vx_true, dtype=np.float32)
        assert vxP.shape == vxT.shape == dp.shape, f"vx shape mismatch: {vxP.shape} vs {vxT.shape} vs {dp.shape}"
        vxP_eval, vxT_eval = _masked_eval(vxP, vxT)
        xrmse, xmae, xpcc = _rmse_mae_pcc(vxP_eval, vxT_eval)
        out.update(dict(
            vx_RMSE=float(xrmse),
            vx_MAE=float(xmae),
            vx_PCC=float(xpcc) if np.isfinite(xpcc) else np.nan,
        ))

    # ---- Vy ----
    if vy_pred is not None and vy_true is not None:
        vyP = np.asarray(vy_pred, dtype=np.float32)
        vyT = np.asarray(vy_true, dtype=np.float32)
        assert vyP.shape == vyT.shape == dp.shape, f"vy shape mismatch: {vyP.shape} vs {vyT.shape} vs {dp.shape}"
        vyP_eval, vyT_eval = _masked_eval(vyP, vyT)
        yrmse, ymae, ypcc = _rmse_mae_pcc(vyP_eval, vyT_eval)
        out.update(dict(
            vy_RMSE=float(yrmse),
            vy_MAE=float(ymae),
            vy_PCC=float(ypcc) if np.isfinite(ypcc) else np.nan,
        ))

    return out


def _finite_minmax(a: np.ndarray, default=(0.0, 1.0)) -> Tuple[float, float]:
    a = np.asarray(a, dtype=np.float32)
    m = np.isfinite(a)
    if not np.any(m):
        return float(default[0]), float(default[1])
    return float(np.nanmin(a[m])), float(np.nanmax(a[m]))

def _finite_absmax(a: np.ndarray, default=1.0) -> float:
    a = np.asarray(a, dtype=np.float32)
    m = np.isfinite(a)
    if not np.any(m):
        return float(default)
    return float(np.nanmax(np.abs(a[m])))

def _nan_mask(values: np.ndarray) -> np.ma.MaskedArray:
    v = np.asarray(values, dtype=np.float32)
    v = np.where(np.isfinite(v), v, np.nan)
    return np.ma.masked_invalid(v)

def _clean(values: np.ndarray, *, min_show: Optional[float] = None) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32).copy()
    v = np.where(np.isfinite(v), v, np.nan)
    if min_show is not None:
        v[v < float(min_show)] = np.nan
    return v

def _mask_zero_component(arr: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32).copy()
    a = np.where(np.isfinite(a), a, np.nan)
    a[np.abs(a) <= float(eps)] = np.nan
    return a

def _vmax_positive_masked(arr: np.ndarray, mask: np.ndarray, p: float = 98.5, floor: float = 1e-6) -> float:
    a = np.asarray(arr).reshape(-1)
    m = np.asarray(mask).reshape(-1)
    ok = m & np.isfinite(a) & (a > 0)
    if not np.any(ok):
        return 1.0
    vmax = float(np.nanpercentile(a[ok], p))
    return max(vmax, floor)

def _vmax_sym_masked(arr: np.ndarray, mask: np.ndarray, p: float = 99.0, floor: float = 1e-6) -> float:
    a = np.asarray(arr).reshape(-1)
    m = np.asarray(mask).reshape(-1)
    ok = m & np.isfinite(a)
    if not np.any(ok):
        return 1.0
    vmax = float(np.nanpercentile(np.abs(a[ok]), p))
    return max(vmax, floor)


def _vmax_positive(arr: np.ndarray, p: float = 98.5, floor: float = 1e-6) -> float:
    a = np.asarray(arr).reshape(-1)
    a = a[np.isfinite(a) & (a > 0)]
    if a.size == 0:
        return 1.0
    vmax = float(np.nanpercentile(a, p))
    return max(vmax, floor)

def _vmax_sym(arr: np.ndarray, p: float = 99.0, floor: float = 1e-6) -> float:
    a = np.asarray(arr).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 1.0
    vmax = float(np.nanpercentile(np.abs(a), p))
    return max(vmax, floor)

def _save(fig, out_noext: str, formats: Sequence[str], dpi: int):
    for fmt in formats:
        fmt = str(fmt).lower().lstrip(".")
        fig.savefig(f"{out_noext}.{fmt}", dpi=int(dpi), bbox_inches="tight")


# ============================================================
# 论文排版：三联图（(a)(b)(c)）
# ============================================================

def _cmap_copy_set_bad(cmap_name: str):
    cm = plt.get_cmap(cmap_name)
    try:
        cm = cm.copy()
        cm.set_bad((0, 0, 0, 0))  # NaN 透明
    except Exception:
        pass
    return cm

def _add_right_colorbar(fig, ax, mappable, *, label: str = "", size: str = "6.5%", pad: float = 0.04):
    """size 越大色棒越宽；pad 越小越贴近子图。"""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cb = fig.colorbar(mappable, cax=cax)
    if label:
        cb.set_label(label)
    cb.ax.tick_params(length=3, width=0.8)
    return cb

def _paper_triptych(
    *,
    cell_polys: List[np.ndarray],
    perimeter_xy: np.ndarray,
    extent: Sequence[float],
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    titles: Tuple[str, str, str],
    norms: Optional[Tuple[Normalize, Normalize, Normalize]] = None,
    panel_labels: Tuple[str, str, str] = ("(a)", "(b)", "(c)"),
    cmaps: Tuple[str, str, str] = ("Blues", "Blues", "RdBu_r"),
    cbar_labels: Tuple[str, str, str] = ("", "", ""),
    draw_bg: bool = True,
    dpi: int = 300,
    figsize: Tuple[float, float] = (18, 6),
    boundary_lw: float = 2.0,
    boundary_color: str = "red",
    alpha: float = 0.95,
    wspace: float = 0.04,        # ✅ 图间距更小
    cbar_size: str = "6.5%",     # ✅ 色棒更宽
    cbar_pad: float = 0.04,      # ✅ 色棒更贴近
):
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=int(dpi))
    fig.subplots_adjust(wspace=float(wspace))

    for ax in axes:
        if draw_bg:
            _draw_background(ax, extent)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("white")

    data_list = [A, B, C]
    for ax, data, title, lab, cmap_name, norm, cblab in zip(
        axes, data_list, titles, panel_labels, cmaps, norms, cbar_labels
    ):
        cm = _cmap_copy_set_bad(cmap_name)
        pc = PolyCollection(
            cell_polys,
            array=_nan_mask(data),
            cmap=cm,
            norm=norm,
            edgecolors="none",
            linewidths=0.0,
            antialiased=True,
            zorder=2,
        )
        pc.set_alpha(float(alpha))
        ax.add_collection(pc)
        draw_perimeter(ax, perimeter_xy, color=boundary_color, lw=boundary_lw)
        ax.set_title(title, pad=6)
        ax.text(
            0.02, 0.98, lab,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=plt.rcParams["axes.titlesize"],
            fontweight="bold",
        )

        # ✅ 每幅图各自色棒（更宽）
        _add_right_colorbar(fig, ax, pc, label=cblab, size=cbar_size, pad=float(cbar_pad))

    return fig


def _paper_single_map(
    *,
    cell_polys: List[np.ndarray],
    perimeter_xy: np.ndarray,
    extent: Sequence[float],
    data: np.ndarray,
    title: str,
    cmap_name: str,
    norm: Optional[Normalize],
    cbar_label: str,
    draw_bg: bool = True,
    dpi: int = 300,
    figsize: Tuple[float, float] = (6.2, 5.8),
    boundary_lw: float = 2.0,
    boundary_color: str = "red",
    alpha: float = 0.95,
    cbar_size: str = "6.5%",
    cbar_pad: float = 0.04,
):
    # 单幅论文风格 map（用于 Pred/Truth/Diff 分开输出）
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=int(dpi))

    if draw_bg:
        _draw_background(ax, extent)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("white")

    cm = _cmap_copy_set_bad(cmap_name)
    pc = PolyCollection(
        cell_polys,
        array=_nan_mask(data),
        cmap=cm,
        norm=norm,
        edgecolors="none",
        linewidths=0.0,
        antialiased=True,
        zorder=2,
    )
    pc.set_alpha(float(alpha))
    ax.add_collection(pc)

    draw_perimeter(ax, perimeter_xy, color=boundary_color, lw=boundary_lw)
    ax.set_title(title, pad=6)

    _add_right_colorbar(fig, ax, pc, label=cbar_label, size=cbar_size, pad=float(cbar_pad))
    return fig


# ============================================================
# inundation confusion：只画 TP/FP/FN（TN 不绘制）
# ============================================================
def _legend_inundation(ax):
    handles = [
        Patch(facecolor="#1f77b4", edgecolor="none", label="TP (Correct Flood)"),
        Patch(facecolor="#d62728", edgecolor="none", label="FP (False Alarm)"),
        Patch(facecolor="#ffdd57", edgecolor="none", label="FN (Missed Flood)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False,
              handlelength=1.2, handletextpad=0.6, borderaxespad=0.6)

def _polycollection_subset(cell_polys: List[np.ndarray], mask: np.ndarray) -> List[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return []
    return [cell_polys[i] for i in idx.tolist()]

def _draw_inundation_confusion_poly(
    *,
    ax,
    cell_polys: List[np.ndarray],
    perimeter_xy: np.ndarray,
    extent: Sequence[float],
    pred_flood: np.ndarray,   # (H_valid,) bool
    true_flood: np.ndarray,   # (H_valid,) bool
    alpha: float = 0.22,
    draw_bg: bool = True,
    boundary_lw: float = 2.0,
    boundary_color: str = "red",
):
    if draw_bg:
        _draw_background(ax, extent)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("white")

    pred_f = np.asarray(pred_flood, dtype=bool)
    true_f = np.asarray(true_flood, dtype=bool)

    TP = pred_f & true_f
    FP = pred_f & (~true_f)
    FN = (~pred_f) & true_f

    # ✅ 关键：只画 TP/FP/FN 的 polygon 子集；TN 完全不绘制
    for mask, color in [(TP, "#1f77b4"), (FP, "#d62728"), (FN, "#ffdd57")]:
        polys = _polycollection_subset(cell_polys, mask)
        if not polys:
            continue
        pc = PolyCollection(polys, facecolors=color, edgecolors="none",
                            linewidths=0.0, antialiased=True, zorder=2)
        pc.set_alpha(float(alpha))
        ax.add_collection(pc)

    draw_perimeter(ax, perimeter_xy, color=boundary_color, lw=boundary_lw, zorder=10)
    _legend_inundation(ax)

    TNn = int((~pred_f & ~true_f).sum())
    FPn = int(FP.sum())
    FNn = int(FN.sum())
    TPn = int(TP.sum())
    ax.set_title(f"TN={TNn}  FP={FPn}  FN={FNn}  TP={TPn}", pad=6)
    return TNn, FPn, FNn, TPn


# ============================================================
# 数据读取：npy + HDF（GT 缺失可从 HDF 读或填 0）
# ============================================================
def _first_existing(*paths: str) -> Optional[str]:
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None

def _list_depth_pred_files(pred_dir: str) -> Tuple[List[str], str]:
    files = sorted(glob.glob(os.path.join(pred_dir, "*_Y_pred.npy")))
    if files:
        return files, "_Y_pred.npy"
    files = sorted(glob.glob(os.path.join(pred_dir, "*_pred.npy")))
    if not files:
        return [], "_pred.npy"

    bad = ("_Vx_", "_Vy_", "_Vmag_", "_vmag_", "_vdir_", "_angle_", "_u_", "_v_")
    files = [f for f in files if not any(k in os.path.basename(f) for k in bad)]
    return sorted(files), "_pred.npy"

def _find_gt_hdf(gt_dir: str, sid: str) -> str:
    for ext in (".hdf", ".h5", ".hdf5"):
        p = os.path.join(gt_dir, f"{sid}{ext}")
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"GT HDF not found for sid={sid} in {gt_dir}")

def _read_hdf(hdf_path: str, dset_path: str, *, T: Optional[int] = None) -> np.ndarray:
    with h5py.File(hdf_path, "r") as h5:
        if dset_path not in h5:
            raise KeyError(f"dataset not found: {dset_path}")
        arr = h5[dset_path][:]
    arr = np.asarray(arr, dtype=np.float32)
    if T is not None:
        arr = arr[:T]
    return arr


# ============================================================
# 指标（可选）
# ============================================================
def _rmse_mae_pcc(pred: np.ndarray, true: np.ndarray) -> Tuple[float, float, float]:
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    g = np.asarray(true, dtype=np.float64).reshape(-1)
    m = np.isfinite(p) & np.isfinite(g)
    if not np.any(m):
        return (np.nan, np.nan, np.nan)
    p = p[m]; g = g[m]
    d = p - g
    rmse = float(np.sqrt(np.mean(d * d)))
    mae = float(np.mean(np.abs(d)))
    if p.size < 2 or np.std(p) < 1e-12 or np.std(g) < 1e-12:
        return (rmse, mae, np.nan)
    pcc = float(np.corrcoef(p, g)[0, 1])
    return (rmse, mae, pcc)

def _f1_pod_far(pred: np.ndarray, true: np.ndarray, thr: float) -> Tuple[float, float, float]:
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    g = np.asarray(true, dtype=np.float64).reshape(-1)
    m = np.isfinite(p) & np.isfinite(g)
    if not np.any(m):
        return (0.0, 0.0, 0.0)
    p = p[m]; g = g[m]
    P = p >= thr
    T = g >= thr
    TP = int((P & T).sum())
    FP = int((P & (~T)).sum())
    FN = int(((~P) & T).sum())
    pod = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1 = (2 * precision * pod) / (precision + pod) if (precision + pod) > 0 else 0.0
    far = FP / (TP + FP) if (TP + FP) > 0 else 0.0
    return float(f1), float(pod), float(far)

def metrics_per_timestep(depth_pred: np.ndarray, depth_true: np.ndarray, *, thr: float) -> pd.DataFrame:
    T = depth_pred.shape[0]
    rows = []
    for t in range(T):
        rmse, mae, pcc = _rmse_mae_pcc(depth_pred[t], depth_true[t])
        f1, pod, far = _f1_pod_far(depth_pred[t], depth_true[t], thr)
        rows.append(dict(t=t, RMSE_depth=rmse, MAE_depth=mae, PCC_depth=pcc, F1_depth=f1, POD_depth=pod, FAR_depth=far))
    return pd.DataFrame(rows)


# ============================================================
# 入口：由 flood_main.py 调用 viz_pred(args)
# ============================================================
def viz_pred(args) -> None:
    """
    与你 _add_viz_args 完全对齐的模块入口：
      pred_dir, gt_dir, output_dir, threshold, combine, pred_only, separate_panels,
      save_depth_maps, save_vel_maps, save_vel_component_maps,
      save_metrics_per_timestep, min_show_depth, min_show_speed,
      vmax_percentile, vmax_mode, geom_hdf, area_name, confusion_render,
      dpi, formats, boundary_lw, bg_image, bg_alpha, bg_stretch, bg_p_low, bg_p_high
    """

    # --- style（你若后续在 flood_main 里添加 font_size/title_size，也可再扩展） ---
    set_paper_style()

    pred_dir = str(getattr(args, "pred_dir", "")).strip()
    gt_dir = str(getattr(args, "gt_dir", "")).strip()
    out_dir = str(getattr(args, "output_dir", "results_viz")).strip()

    if not pred_dir or (not os.path.isdir(pred_dir)):
        raise FileNotFoundError(f"pred_dir not found: {pred_dir}")
    os.makedirs(out_dir, exist_ok=True)
    min_show_vcomp = 0.001
    threshold = float(getattr(args, "threshold", 0.1))
    combine = bool(getattr(args, "combine", True))
    pred_only = bool(getattr(args, "pred_only", False))
    # 分开输出开关：每个时刻输出 3 张图（Pred / Truth / Diff），不再三联图
    separate_panels = bool(getattr(args, "separate_panels", False)) or bool(getattr(args, "separate_maps", False))
    if separate_panels:
        combine = False

    save_depth_maps = bool(getattr(args, "save_depth_maps", True))
    save_vel_maps = bool(getattr(args, "save_vel_maps", True))
    save_vel_component_maps = bool(getattr(args, "save_vel_component_maps", True))
    save_metrics = bool(getattr(args, "save_metrics_per_timestep", True))

    min_show_depth = float(getattr(args, "min_show_depth", 0.1))
    min_show_speed = float(getattr(args, "min_show_speed", 0.001))

    vmax_percentile = float(getattr(args, "vmax_percentile", 100))
    vmax_mode = str(getattr(args, "vmax_mode", "global"))
    boundary_lw = float(getattr(args, "boundary_lw", 2.0))

    dpi = int(getattr(args, "dpi", 300))
    formats = getattr(args, "formats", "png")
    if isinstance(formats, str):
        formats = tuple([x.strip() for x in formats.split(",") if x.strip()])
    else:
        formats = tuple(formats)

    geom_hdf = str(getattr(args, "geom_hdf", "")).strip()
    area_name = str(getattr(args, "area_name", DEFAULT_2D_AREA_NAME))

    confusion_render = str(getattr(args, "confusion_render", "auto")).lower()

    # background
    bg_image = str(getattr(args, "bg_image", "")).strip()
    bg_alpha = float(getattr(args, "bg_alpha", 0.5))
    bg_stretch = bool(getattr(args, "bg_stretch", True))
    bg_p_low = float(getattr(args, "bg_p_low", 2.0))
    bg_p_high = float(getattr(args, "bg_p_high", 98.0))

    if bg_image and os.path.isfile(bg_image):
        load_background(bg_image, alpha=bg_alpha, stretch=bg_stretch, p_low=bg_p_low, p_high=bg_p_high)
        draw_bg = True
    else:
        load_background(None, alpha=bg_alpha)
        draw_bg = False

    depth_cmap = "Blues"    # 修改水深色棒的参数
    vmag_cmap = "viridis"   # 修改流速色棒的参数
    diff_cmap = "RdBu_r"    # 修改差值色棒的参数
    boundary_color = "red"  # 外边界虚线的颜色调整
    confusion_alpha = 0.8   # 淹没图网格透明程度设置
    fig_triptych = (18.0, 6.0)

    pred_files, suffix = _list_depth_pred_files(pred_dir)
    if not pred_files:
        raise FileNotFoundError(f"No depth prediction found in: {pred_dir}")

    # 如果 geom_hdf 不存在：允许 per-sample 用 gt_dir/{sid}.hdf 读取 Geometry
    use_global_geom = bool(geom_hdf) and os.path.isfile(geom_hdf)
    if (not use_global_geom) and (not gt_dir or not os.path.isdir(gt_dir)):
        raise ValueError("geom_hdf is missing AND gt_dir is invalid. Need one of them to load Geometry.")

    all_event_rows: List[Dict[str, Any]] = []

    for fp in tqdm(pred_files, desc="Samples", unit="sample"):
        sid = os.path.basename(fp)[:-len(suffix)]
        sample_out = os.path.join(out_dir, sid)
        os.makedirs(sample_out, exist_ok=True)

        depth_pred_full = np.load(fp).astype(np.float32)
        if depth_pred_full.ndim != 2:
            raise ValueError(f"{sid}: depth_pred must be [T,H], got {depth_pred_full.shape}")
        T, H_full = depth_pred_full.shape

        # geometry
        geom_src = geom_hdf if use_global_geom else _find_gt_hdf(gt_dir, sid)
        perimeter_xy, cell_polys, keep_idx, extent = load_hecras_cell_polys(geom_src, area_name=area_name, cache=True)

        if keep_idx.size == 0:
            raise RuntimeError(f"{sid}: No valid polygons loaded. Check geom_hdf/area_name.")

        if int(np.max(keep_idx)) >= H_full:
            raise ValueError(
                f"{sid}: Geometry cell index exceeds prediction H. "
                f"keep_idx max={int(np.max(keep_idx))}, pred H={H_full}. "
                f"Check that prediction cell ordering matches Geometry."
            )

        depth_pred = depth_pred_full[:, keep_idx]  # (T, H_valid)

        # depth true
        depth_true = None
        if not pred_only:
            gt_npy = _first_existing(
                os.path.join(pred_dir, f"{sid}_Y_true.npy"),
                os.path.join(pred_dir, f"{sid}_true.npy"),
                os.path.join(pred_dir, f"{sid}_gt.npy"),
            )
            if gt_npy is not None:
                depth_true_full = np.load(gt_npy).astype(np.float32)[:T]
                depth_true = depth_true_full[:, keep_idx]
            else:
                gt_hdf = _find_gt_hdf(gt_dir, sid) if (gt_dir and os.path.isdir(gt_dir)) else None
                if gt_hdf:
                    try:
                        depth_true_full = _read_hdf(gt_hdf, depth_hdf_path(area_name), T=T)
                        depth_true = depth_true_full[:, keep_idx]
                    except Exception as e:
                        logger.warning(f"{sid}: read depth GT from HDF failed: {e} -> use zeros")

        if depth_true is None:
            depth_true = np.zeros_like(depth_pred, dtype=np.float32)

        # ======================
        # Wet/Dry masks (T,H)
        # ======================
        wet_pred_all = np.isfinite(depth_pred) & (depth_pred >= threshold)
        wet_true_all = np.isfinite(depth_true) & (depth_true >= threshold)
        wet_union_all = wet_pred_all | wet_true_all

        # ----------------------
        # Depth global min/max (Pred & True 分别)
        # ----------------------
        depth_pred_min, depth_pred_max = _finite_minmax(depth_pred, default=(0.0, 1.0))
        depth_true_min, depth_true_max = _finite_minmax(depth_true, default=(0.0, 1.0))

        # 深度物理上通常>=0，这里防止极小负值把色阶拉偏（仍满足“来自文件全局 min/max”的精神）
        depth_pred_min = max(0.0, depth_pred_min)
        depth_true_min = max(0.0, depth_true_min)

        # depth diff：仅在 wet_union 内统计，对称色阶
        depth_diff_all = (depth_pred - depth_true).astype(np.float32)
        depth_diff_all[~wet_union_all] = np.nan
        depth_diff_abs = _finite_absmax(depth_diff_all, default=1.0)

        # velocity (Vx/Vy)
        def _find_pred(name: str) -> Optional[str]:
            return _first_existing(os.path.join(pred_dir, f"{sid}_{name}_pred.npy"))

        def _find_true(name: str) -> Optional[str]:
            return _first_existing(os.path.join(pred_dir, f"{sid}_{name}_true.npy"))

        vx_pred = vy_pred = vx_true = vy_true = None
        vmag_pred = vmag_true = None

        if save_vel_maps or save_vel_component_maps:
            vx_p = _find_pred("Vx")
            vy_p = _find_pred("Vy")
            if vx_p and vy_p:
                vx_pred_full = np.load(vx_p).astype(np.float32)[:T]
                vy_pred_full = np.load(vy_p).astype(np.float32)[:T]
                vx_pred = vx_pred_full[:, keep_idx]
                vy_pred = vy_pred_full[:, keep_idx]

                if (not pred_only) and (gt_dir and os.path.isdir(gt_dir)):
                    vx_t = _find_true("Vx")
                    vy_t = _find_true("Vy")
                    if vx_t and vy_t:
                        vx_true_full = np.load(vx_t).astype(np.float32)[:T]
                        vy_true_full = np.load(vy_t).astype(np.float32)[:T]
                        vx_true = vx_true_full[:, keep_idx]
                        vy_true = vy_true_full[:, keep_idx]
                    else:
                        try:
                            gt_hdf = _find_gt_hdf(gt_dir, sid)
                            vx_true_full = _read_hdf(gt_hdf, vx_hdf_path(area_name), T=T)
                            vy_true_full = _read_hdf(gt_hdf, vy_hdf_path(area_name), T=T)
                            vx_true = vx_true_full[:, keep_idx]
                            vy_true = vy_true_full[:, keep_idx]
                        except Exception as e:
                            logger.warning(f"{sid}: read Vx/Vy GT from HDF failed: {e} -> use zeros")
                            vx_true = np.zeros_like(vx_pred, dtype=np.float32)
                            vy_true = np.zeros_like(vy_pred, dtype=np.float32)
                else:
                    vx_true = np.zeros_like(vx_pred, dtype=np.float32) if vx_pred is not None else None
                    vy_true = np.zeros_like(vy_pred, dtype=np.float32) if vy_pred is not None else None

        if vx_pred is not None and vy_pred is not None:
            vmag_pred = np.sqrt(vx_pred * vx_pred + vy_pred * vy_pred).astype(np.float32)
            if vx_true is not None and vy_true is not None:
                vmag_true = np.sqrt(vx_true * vx_true + vy_true * vy_true).astype(np.float32)
            else:
                vmag_true = np.zeros_like(vmag_pred, dtype=np.float32)

        # ----------------------
        # Velocity global min/max (vmag + components) —— 只在 wet 内统计
        # ----------------------
        vmag_pred_min = vmag_pred_max = vmag_true_min = vmag_true_max = None
        vmag_diff_abs = None

        vx_pred_min = vx_pred_max = vx_true_min = vx_true_max = None
        vy_pred_min = vy_pred_max = vy_true_min = vy_true_max = None
        vx_diff_abs = vy_diff_abs = None

        if vmag_pred is not None and vmag_true is not None:
            vmp = vmag_pred.astype(np.float32).copy()
            vmt = vmag_true.astype(np.float32).copy()
            vmp[~wet_pred_all] = np.nan
            vmt[~wet_true_all] = np.nan

            vmag_pred_min, vmag_pred_max = _finite_minmax(vmp, default=(0.0, 1.0))
            vmag_true_min, vmag_true_max = _finite_minmax(vmt, default=(0.0, 1.0))
            vmag_pred_min = max(0.0, vmag_pred_min)
            vmag_true_min = max(0.0, vmag_true_min)

            vmd = (vmag_pred - vmag_true).astype(np.float32)
            vmd[~wet_union_all] = np.nan
            vmag_diff_abs = _finite_absmax(vmd, default=1.0)

        if (vx_pred is not None) and (vx_true is not None) and (vy_pred is not None) and (vy_true is not None):
            vxp = vx_pred.astype(np.float32).copy()
            vxp[~wet_pred_all] = np.nan
            vxg = vx_true.astype(np.float32).copy()
            vxg[~wet_true_all] = np.nan
            vyp = vy_pred.astype(np.float32).copy()
            vyp[~wet_pred_all] = np.nan
            vyg = vy_true.astype(np.float32).copy()
            vyg[~wet_true_all] = np.nan

            vx_pred_min, vx_pred_max = _finite_minmax(vxp, default=(-1.0, 1.0))
            vx_true_min, vx_true_max = _finite_minmax(vxg, default=(-1.0, 1.0))
            vy_pred_min, vy_pred_max = _finite_minmax(vyp, default=(-1.0, 1.0))
            vy_true_min, vy_true_max = _finite_minmax(vyg, default=(-1.0, 1.0))

            vxd = (vx_pred - vx_true).astype(np.float32)
            vxd[~wet_union_all] = np.nan
            vyd = (vy_pred - vy_true).astype(np.float32)
            vyd[~wet_union_all] = np.nan
            vx_diff_abs = _finite_absmax(vxd, default=1.0)
            vy_diff_abs = _finite_absmax(vyd, default=1.0)
        # ======================
        # plotting per t
        # ======================
        for t in range(T):
            # ---- Depth ----
            if save_depth_maps:
                dp = _clean(depth_pred[t], min_show=min_show_depth)

                if pred_only:
                    # Pred-only
                    fig = plt.figure(figsize=(6.2, 5.8), dpi=int(dpi))
                    ax = fig.add_subplot(1, 1, 1)
                    if draw_bg:
                        _draw_background(ax, extent)
                    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
                    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
                    ax.set_facecolor("white")

                    cm = plt.get_cmap(depth_cmap)
                    try:
                        cm = cm.copy(); cm.set_bad((0, 0, 0, 0))
                    except Exception:
                        pass
                    pc = PolyCollection(cell_polys, array=_nan_mask(dp), cmap=cm, norm=Normalize(depth_pred_min, depth_pred_max),
                                        edgecolors="none", linewidths=0.0, antialiased=True, zorder=2)
                    pc.set_alpha(0.95)
                    ax.add_collection(pc)
                    draw_perimeter(ax, perimeter_xy, color=boundary_color, lw=boundary_lw)
                    ax.set_title("Predicted", pad=6)
                    cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.02)
                    cb.set_label("Depth (m)")
                    _save(fig, os.path.join(sample_out, f"{sid}_t{t:02d}_depth_pred"), formats, dpi)
                    plt.close(fig)

                else:
                    dt = _clean(depth_true[t], min_show=min_show_depth)
                    diff = (depth_pred[t] - depth_true[t]).astype(np.float32)

                    show = (depth_pred[t] >= threshold) | (depth_true[t] >= threshold)
                    diff[~show] = np.nan
                    diff = _clean(diff, min_show=None)
                    if separate_panels:
                        # Pred
                        figp = _paper_single_map(
                            cell_polys=cell_polys, perimeter_xy=perimeter_xy, extent=extent,
                            data=dp,
                            title="Predicted Value",
                            cmap_name=depth_cmap,
                            norm=Normalize(depth_pred_min, depth_pred_max),
                            cbar_label="h (m)",
                            draw_bg=draw_bg,
                            dpi=dpi,
                            figsize=(6.2, 5.8),
                            boundary_lw=boundary_lw, boundary_color=boundary_color
                        )
                        _save(figp, os.path.join(sample_out, f"{sid}_t{t:02d}水深_预测"), formats, dpi)
                        plt.close(figp)

                        # Truth
                        figt = _paper_single_map(
                            cell_polys=cell_polys, perimeter_xy=perimeter_xy, extent=extent,
                            data=dt,
                            title="Truth Value",
                            cmap_name=depth_cmap,
                            norm=Normalize(depth_true_min, depth_true_max),
                            cbar_label="h (m)",
                            draw_bg=draw_bg,
                            dpi=dpi,
                            figsize=(6.2, 5.8),
                            boundary_lw=boundary_lw, boundary_color=boundary_color
                        )
                        _save(figt, os.path.join(sample_out, f"{sid}_t{t:02d}水深_真值"), formats, dpi)
                        plt.close(figt)

                        # Diff
                        figd = _paper_single_map(
                            cell_polys=cell_polys, perimeter_xy=perimeter_xy, extent=extent,
                            data=diff,
                            title="Difference",
                            cmap_name=diff_cmap,
                            norm=Normalize(-depth_diff_abs, depth_diff_abs),
                            cbar_label="h (m)",
                            draw_bg=draw_bg,
                            dpi=dpi,
                            figsize=(6.2, 5.8),
                            boundary_lw=boundary_lw, boundary_color=boundary_color
                        )
                        _save(figd, os.path.join(sample_out, f"{sid}_t{t:02d}水深_差值"), formats, dpi)
                        plt.close(figd)

                    elif combine:
                        fig = _paper_triptych(
                            cell_polys=cell_polys, perimeter_xy=perimeter_xy, extent=extent,
                            A=dp, B=dt, C=diff,
                            titles=("Predicted Value", "Truth Value", "Difference"),
                            norms=(
                                Normalize(depth_pred_min, depth_pred_max),
                                Normalize(depth_true_min, depth_true_max),
                                Normalize(-depth_diff_abs, depth_diff_abs),
                            ),
                            cbar_labels=("h (m)", "h (m)", "h (m)"),
                            draw_bg=draw_bg,
                            dpi=dpi, figsize=(18.0, 6.0),
                            boundary_lw=boundary_lw, boundary_color=boundary_color
                        )
                        _save(fig, os.path.join(sample_out, f"{sid}_t{t:02d}水深图"), formats, dpi)
                        plt.close(fig)
                    # confusion (TN 不绘制)
                    if confusion_render in ("auto", "poly"):
                        figc = plt.figure(figsize=(6.2, 5.8), dpi=int(dpi))
                        axc = figc.add_subplot(1, 1, 1)
                        pred_f = depth_pred[t] >= threshold
                        true_f = depth_true[t] >= threshold
                        _draw_inundation_confusion_poly(
                            ax=axc, cell_polys=cell_polys, perimeter_xy=perimeter_xy, extent=extent,
                            pred_flood=pred_f, true_flood=true_f,
                            alpha=confusion_alpha, draw_bg=draw_bg,
                            boundary_lw=boundary_lw, boundary_color=boundary_color
                        )
                        _save(figc, os.path.join(sample_out, f"{sid}_t{t:02d}洪水淹没图"), formats, dpi)
                        plt.close(figc)
                    elif confusion_render == "raster":
                        logger.warning(f"{sid}: confusion_render=raster requested, but poly mode is used as fallback (no rasterization).")

            wet_p = wet_pred_all[t]  # (H,)
            wet_g = wet_true_all[t]
            wet_u = wet_union_all[t]

            # ======================
            # Vmag 3-in-1（|v|）
            # ======================
            if vmag_pred is not None and vmag_true is not None:
                sp = vmag_pred[t].astype(np.float32).copy()
                st = vmag_true[t].astype(np.float32).copy()

                sp[~wet_p] = np.nan
                st[~wet_g] = np.nan
                sp = _clean(sp, min_show=min_show_speed)
                st = _clean(st, min_show=min_show_speed)

                vdiff = (vmag_pred[t] - vmag_true[t]).astype(np.float32)
                vdiff[~wet_u] = np.nan
                vdiff = _clean(vdiff, min_show=None)
                if separate_panels:
                    figp = _paper_single_map(
                        cell_polys=cell_polys, perimeter_xy=perimeter_xy, extent=extent,
                        data=sp,
                        title="Predicted Value",
                        cmap_name=vmag_cmap,
                        norm=Normalize(vmag_pred_min, vmag_pred_max),
                        cbar_label="|v| (m/s)",
                        draw_bg=draw_bg,
                        dpi=dpi,
                        figsize=(6.2, 5.8),
                        boundary_lw=boundary_lw, boundary_color=boundary_color
                    )
                    _save(figp, os.path.join(sample_out, f"{sid}_t{t:02d}流速模长_预测"), formats, dpi)
                    plt.close(figp)

                    figt = _paper_single_map(
                        cell_polys=cell_polys, perimeter_xy=perimeter_xy, extent=extent,
                        data=st,
                        title="Truth Value",
                        cmap_name=vmag_cmap,
                        norm=Normalize(vmag_true_min, vmag_true_max),
                        cbar_label="|v| (m/s)",
                        draw_bg=draw_bg,
                        dpi=dpi,
                        figsize=(6.2, 5.8),
                        boundary_lw=boundary_lw, boundary_color=boundary_color
                    )
                    _save(figt, os.path.join(sample_out, f"{sid}_t{t:02d}流速模长_真值"), formats, dpi)
                    plt.close(figt)

                    figd = _paper_single_map(
                        cell_polys=cell_polys, perimeter_xy=perimeter_xy, extent=extent,
                        data=vdiff,
                        title="Difference",
                        cmap_name=diff_cmap,
                        norm=Normalize(-vmag_diff_abs, vmag_diff_abs),
                        cbar_label="Δ|v| (m/s)",
                        draw_bg=draw_bg,
                        dpi=dpi,
                        figsize=(6.2, 5.8),
                        boundary_lw=boundary_lw, boundary_color=boundary_color
                    )
                    _save(figd, os.path.join(sample_out, f"{sid}_t{t:02d}流速模长_差值"), formats, dpi)
                    plt.close(figd)

                else:
                    figv = _paper_triptych(
                        cell_polys=cell_polys, perimeter_xy=perimeter_xy, extent=extent,
                        A=sp, B=st, C=vdiff,
                        titles=("Predicted Value", "Truth Value", "Difference"),
                        norms=(
                            Normalize(vmag_pred_min, vmag_pred_max),
                            Normalize(vmag_true_min, vmag_true_max),
                            Normalize(-vmag_diff_abs, vmag_diff_abs),
                        ),
                        cmaps=(vmag_cmap, vmag_cmap, diff_cmap),
                        cbar_labels=("|v| (m/s)", "|v| (m/s)", "Δ|v| (m/s)"),
                        draw_bg=draw_bg,
                        dpi=dpi, figsize=fig_triptych,
                        boundary_lw=boundary_lw, boundary_color=boundary_color
                    )
                    _save(figv, os.path.join(sample_out, f"{sid}_t{t:02d}流速模长"), formats, dpi)
                    plt.close(figv)
            # ---- Vx/Vy ----
            if save_vel_component_maps and (vx_pred is not None) and (vx_true is not None) and (vy_pred is not None) and (vy_true is not None) and (not pred_only):
                 # ---- Vx ----
                vxP = vx_pred[t].astype(np.float32).copy()
                vxG = vx_true[t].astype(np.float32).copy()
                vxP[~wet_p] = np.nan
                vxG[~wet_g] = np.nan
                vxP[np.abs(vxP) <= min_show_vcomp] = np.nan  # ✅避免发白
                vxG[np.abs(vxG) <= min_show_vcomp] = np.nan

                vxD = (vx_pred[t] - vx_true[t]).astype(np.float32)
                vxD[~wet_u] = np.nan

                # Vx
                eps0 = 1e-6

                vx_p_raw = vx_pred[t].astype(np.float32)
                vx_t_raw = vx_true[t].astype(np.float32)

                # Diff：仅在“pred 和 gt 都为 0”时隐藏 0（否则保留差异）
                vx_d = (vx_p_raw - vx_t_raw).astype(np.float32)
                mask_both_zero = (np.abs(vx_p_raw) <= eps0) & (np.abs(vx_t_raw) <= eps0)
                vx_d[mask_both_zero] = np.nan
                if separate_panels:
                    figp = _paper_single_map(
                        cell_polys=cell_polys, perimeter_xy=perimeter_xy, extent=extent,
                        data=_clean(vxP, min_show=None),
                        title="Predicted Value",
                        cmap_name=vmag_cmap,
                        norm=Normalize(vx_pred_min, vx_pred_max),
                        cbar_label="Vx (m/s)",
                        draw_bg=draw_bg, dpi=dpi, figsize=(6.2, 5.8),
                        boundary_lw=boundary_lw, boundary_color=boundary_color
                    )
                    _save(figp, os.path.join(sample_out, f"{sid}_t{t:02d}Vx_预测"), formats, dpi)
                    plt.close(figp)

                    figt = _paper_single_map(
                        cell_polys=cell_polys, perimeter_xy=perimeter_xy, extent=extent,
                        data=_clean(vxG, min_show=None),
                        title="Truth Value",
                        cmap_name=vmag_cmap,
                        norm=Normalize(vx_true_min, vx_true_max),
                        cbar_label="Vx (m/s)",
                        draw_bg=draw_bg, dpi=dpi, figsize=(6.2, 5.8),
                        boundary_lw=boundary_lw, boundary_color=boundary_color
                    )
                    _save(figt, os.path.join(sample_out, f"{sid}_t{t:02d}Vx_真值"), formats, dpi)
                    plt.close(figt)

                    figd = _paper_single_map(
                        cell_polys=cell_polys, perimeter_xy=perimeter_xy, extent=extent,
                        data=_clean(vxD, min_show=None),
                        title="Difference",
                        cmap_name=diff_cmap,
                        norm=Normalize(-vx_diff_abs, vx_diff_abs),
                        cbar_label="ΔVx (m/s)",
                        draw_bg=draw_bg, dpi=dpi, figsize=(6.2, 5.8),
                        boundary_lw=boundary_lw, boundary_color=boundary_color
                    )
                    _save(figd, os.path.join(sample_out, f"{sid}_t{t:02d}Vx_差值"), formats, dpi)
                    plt.close(figd)

                else:
                    figx = _paper_triptych(
                        cell_polys=cell_polys, perimeter_xy=perimeter_xy, extent=extent,
                        A=_clean(vxP, min_show=None),
                        B=_clean(vxG, min_show=None),
                        C=_clean(vxD, min_show=None),
                        titles=("Predicted Value", "Truth Value", "Difference"),
                        norms=(
                            Normalize(vx_pred_min, vx_pred_max),
                            Normalize(vx_true_min, vx_true_max),
                            Normalize(-vx_diff_abs, vx_diff_abs),
                        ),
                        cmaps=(vmag_cmap, vmag_cmap, diff_cmap),
                        cbar_labels=("Vx (m/s)", "Vx (m/s)", "ΔVx (m/s)"),
                        draw_bg=draw_bg, dpi=dpi, figsize=fig_triptych,
                        boundary_lw=boundary_lw, boundary_color=boundary_color
                    )
                    _save(figx, os.path.join(sample_out, f"{sid}_t{t:02d}Vx"), formats, dpi)
                    plt.close(figx)
                vyP = vy_pred[t].astype(np.float32).copy()
                vyG = vy_true[t].astype(np.float32).copy()
                vyP[~wet_p] = np.nan
                vyG[~wet_g] = np.nan
                vyP[np.abs(vyP) <= min_show_vcomp] = np.nan  # ✅避免发白
                vyG[np.abs(vyG) <= min_show_vcomp] = np.nan

                vyD = (vy_pred[t] - vy_true[t]).astype(np.float32)
                vyD[~wet_u] = np.nan

                # Vy
                vy_p_raw = vy_pred[t].astype(np.float32)
                vy_t_raw = vy_true[t].astype(np.float32)

                # Diff：仅在“pred 和 gt 都为 0”时隐藏 0（否则保留差异）
                vy_d = (vy_p_raw - vy_t_raw).astype(np.float32)
                mask_both_zero = (np.abs(vy_p_raw) <= eps0) & (np.abs(vy_t_raw) <= eps0)
                vy_d[mask_both_zero] = np.nan
                if separate_panels:
                    figp = _paper_single_map(
                        cell_polys=cell_polys, perimeter_xy=perimeter_xy, extent=extent,
                        data=_clean(vyP, min_show=None),
                        title="Predicted Value",
                        cmap_name=vmag_cmap,
                        norm=Normalize(vy_pred_min, vy_pred_max),
                        cbar_label="Vy (m/s)",
                        draw_bg=draw_bg, dpi=dpi, figsize=(6.2, 5.8),
                        boundary_lw=boundary_lw, boundary_color=boundary_color
                    )
                    _save(figp, os.path.join(sample_out, f"{sid}_t{t:02d}Vy_预测"), formats, dpi)
                    plt.close(figp)

                    figt = _paper_single_map(
                        cell_polys=cell_polys, perimeter_xy=perimeter_xy, extent=extent,
                        data=_clean(vyG, min_show=None),
                        title="Truth Value",
                        cmap_name=vmag_cmap,
                        norm=Normalize(vy_true_min, vy_true_max),
                        cbar_label="Vy (m/s)",
                        draw_bg=draw_bg, dpi=dpi, figsize=(6.2, 5.8),
                        boundary_lw=boundary_lw, boundary_color=boundary_color
                    )
                    _save(figt, os.path.join(sample_out, f"{sid}_t{t:02d}Vy_真值"), formats, dpi)
                    plt.close(figt)

                    figd = _paper_single_map(
                        cell_polys=cell_polys, perimeter_xy=perimeter_xy, extent=extent,
                        data=_clean(vyD, min_show=None),
                        title="Difference",
                        cmap_name=diff_cmap,
                        norm=Normalize(-vy_diff_abs, vy_diff_abs),
                        cbar_label="ΔVy (m/s)",
                        draw_bg=draw_bg, dpi=dpi, figsize=(6.2, 5.8),
                        boundary_lw=boundary_lw, boundary_color=boundary_color
                    )
                    _save(figd, os.path.join(sample_out, f"{sid}_t{t:02d}Vy_差值"), formats, dpi)
                    plt.close(figd)

                else:
                    figy = _paper_triptych(
                        cell_polys=cell_polys, perimeter_xy=perimeter_xy, extent=extent,
                        A=_clean(vyP, min_show=None),
                        B=_clean(vyG, min_show=None),
                        C=_clean(vyD, min_show=None),
                        titles=("Predicted Value", "Truth Value", "Difference"),
                        cmaps=(vmag_cmap, vmag_cmap, diff_cmap),
                        norms=(
                            Normalize(vy_pred_min, vy_pred_max),
                            Normalize(vy_true_min, vy_true_max),
                            Normalize(-vy_diff_abs, vy_diff_abs),
                        ),
                        cbar_labels=("Vy (m/s)", "Vy (m/s)", "ΔVy (m/s)"),
                        draw_bg=draw_bg, dpi=dpi, figsize=fig_triptych,
                        boundary_lw=boundary_lw, boundary_color=boundary_color
                    )
                    _save(figy, os.path.join(sample_out, f"{sid}_t{t:02d}Vy"), formats, dpi)
                    plt.close(figy)
        # metrics
        if save_metrics and (not pred_only):
            df = metrics_per_timestep(depth_pred, depth_true, thr=threshold)
            df.to_csv(os.path.join(sample_out, f"{sid}_metrics_per_timestep.csv"), index=False, encoding="utf-8-sig")
            # ======================
            # 整场洪水汇总指标（T×H）
            # ======================
            if (not pred_only):
                ev = metrics_event_flood(
                    depth_pred, depth_true,
                    thr=threshold,
                    vmag_pred=vmag_pred,
                    vmag_true=vmag_true,
                    vx_pred=vx_pred,
                    vx_true=vx_true,
                    vy_pred=vy_pred,
                    vy_true=vy_true,
                    wet_union_all=wet_union_all
                )
                ev.update(dict(
                    sid=str(sid),
                    T=int(depth_pred.shape[0]),
                    H=int(depth_pred.shape[1]),
                    threshold=float(threshold),
                ))

                # 每个样本一个汇总文件（单行）
                pd.DataFrame([ev]).to_csv(
                    os.path.join(sample_out, f"{sid}_event_metrics.csv"),
                    index=False, encoding="utf-8-sig"
                )

                # 汇总到全局
                all_event_rows.append(ev)
        # ======================
        # 全样本整场洪水指标汇总
        # ======================
        if all_event_rows:
            df_all = pd.DataFrame(all_event_rows)
            df_all.to_csv(
                os.path.join(out_dir, "event_metrics_all_samples.csv"),
                index=False, encoding="utf-8-sig"
            )

    logger.info(f"DONE. Output => {out_dir}")

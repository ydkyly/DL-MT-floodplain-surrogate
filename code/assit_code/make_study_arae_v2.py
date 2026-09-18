# -*- coding: utf-8 -*-
"""
make_study_area_mask.py

根据 HEC-RAS HDF 中二维流动区域 Perimeter，叠加 GeoTIFF 遥感底图，
生成“研究区内正常显示、研究区外半透明灰白淡化/透明化”的研究区范围图。

依赖：
    pip install h5py rasterio matplotlib scipy

示例：
    python make_study_area_mask.py \
        --hdf "D:/Work/qyb/ResNet-18/data/test/hdf/1.hdf" \
        --tif "D:/Work/qyb/Hec_Ras/YD/YD.tif" \
        --out "StudyArea/study_area.png" \
        --area "Perimeter 1" \
        --max_size 3500 \
        --feather 6
"""
from __future__ import annotations

import os
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path

try:
    import rasterio
    from rasterio.enums import Resampling
except Exception as e:
    raise RuntimeError("需要安装 rasterio：pip install rasterio") from e

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None


# -----------------------------
# HDF geometry path
# -----------------------------
def _geom_perimeter_path(area_name: str) -> str:
    return "/".join(["Geometry", "2D Flow Areas", area_name, "Perimeter"])


def _detect_area_name(h5: h5py.File, area_name: str | None) -> str:
    root = "Geometry/2D Flow Areas"
    if root not in h5:
        raise KeyError(f"HDF 中找不到二维流动区域根路径：{root}")

    area_names = list(h5[root].keys())
    if not area_names:
        raise KeyError(f"{root} 下没有任何 2D Flow Area")

    if area_name and area_name in area_names:
        return area_name

    if area_name and area_name not in area_names:
        print(f"[Warn] 指定 area={area_name!r} 不存在，HDF 中可用区域为：{area_names}")

    if len(area_names) == 1:
        print(f"[Info] 自动使用唯一二维区域：{area_names[0]}")
        return area_names[0]

    raise ValueError(f"HDF 中存在多个 2D Flow Area，请用 --area 指定：{area_names}")


def read_perimeter_xy(hdf_path: str, area_name: str | None = "Perimeter 1") -> np.ndarray:
    """读取 HEC-RAS HDF 中的二维流动区域外边界 Perimeter，返回 [N,2]。"""
    if not os.path.isfile(hdf_path):
        raise FileNotFoundError(f"HDF 文件不存在：{hdf_path}")

    with h5py.File(hdf_path, "r") as h5:
        area = _detect_area_name(h5, area_name)
        per_path = _geom_perimeter_path(area)
        if per_path not in h5:
            raise KeyError(f"HDF 中找不到研究区边界数据集：{per_path}")
        per = h5[per_path][:].astype(np.float64)

    if per.ndim != 2 or per.shape[1] != 2 or per.shape[0] < 3:
        raise ValueError(f"Perimeter 坐标形状异常：{per.shape}，期望 [N,2]")

    # 闭合边界
    if not np.allclose(per[0], per[-1]):
        per = np.vstack([per, per[0]])
    return per


# -----------------------------
# GeoTIFF read + display stretch
# -----------------------------
def percentile_stretch_rgb(rgb: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """将 RGB 影像按分位数拉伸到 [0,1]，用于增强遥感底图显示。"""
    rgb = rgb.astype(np.float32)
    out = np.zeros_like(rgb, dtype=np.float32)
    for i in range(3):
        band = rgb[..., i]
        finite = np.isfinite(band)
        if not np.any(finite):
            continue
        lo, hi = np.nanpercentile(band[finite], [p_low, p_high])
        if hi <= lo:
            out[..., i] = np.clip(band, 0, 1)
        else:
            out[..., i] = np.clip((band - lo) / (hi - lo), 0, 1)
    return out


def read_tif_rgb(tif_path: str, max_size: int = 3500, p_low: float = 2.0, p_high: float = 98.0):
    """读取 GeoTIFF 为 RGB 图像，并返回 extent=[xmin,xmax,ymin,ymax]。"""
    if not os.path.isfile(tif_path):
        raise FileNotFoundError(f"TIF 文件不存在：{tif_path}")

    with rasterio.open(tif_path) as ds:
        scale = min(1.0, float(max_size) / float(max(ds.width, ds.height)))
        out_w = max(1, int(round(ds.width * scale)))
        out_h = max(1, int(round(ds.height * scale)))

        if ds.count >= 3:
            arr = ds.read([1, 2, 3], out_shape=(3, out_h, out_w), resampling=Resampling.bilinear)
            rgb = np.transpose(arr, (1, 2, 0)).astype(np.float32)
        else:
            arr = ds.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype(np.float32)
            rgb = np.repeat(arr[..., None], 3, axis=2)

        left, bottom, right, top = ds.bounds
        extent = [float(left), float(right), float(bottom), float(top)]
        crs = str(ds.crs) if ds.crs is not None else None

    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    rgb = percentile_stretch_rgb(rgb, p_low=p_low, p_high=p_high)
    return rgb, extent, crs


# -----------------------------
# Mask + compose
# -----------------------------
def make_inside_mask(perimeter_xy: np.ndarray, extent: list[float], shape_hw: tuple[int, int]) -> np.ndarray:
    """根据边界 polygon 和影像 extent，生成栅格级 inside mask。"""
    h, w = shape_hw
    xmin, xmax, ymin, ymax = extent

    xs = xmin + (np.arange(w, dtype=np.float64) + 0.5) * (xmax - xmin) / w
    ys = ymax - (np.arange(h, dtype=np.float64) + 0.5) * (ymax - ymin) / h
    X, Y = np.meshgrid(xs, ys)

    path = Path(perimeter_xy)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    mask = path.contains_points(pts).reshape(h, w)
    return mask


def compose_effect(
    rgb: np.ndarray,
    inside_mask: np.ndarray,
    *,
    feather: float = 6.0,
    outside_fade: float = 0.65,
    outside_alpha: float = 0.55,
    transparent_outside: bool = False,
) -> np.ndarray:
    """
    生成类似示例图的效果：
    - 研究区内保持正常彩色；
    - 研究区外做灰白淡化；
    - 边界通过 feather 做柔化过渡。
    """
    rgb = np.clip(rgb.astype(np.float32), 0, 1)
    mask = inside_mask.astype(np.float32)

    if feather > 0 and gaussian_filter is not None:
        alpha = gaussian_filter(mask, sigma=float(feather))
        alpha = np.clip(alpha, 0.0, 1.0)
    else:
        alpha = mask

    # 区域外：先生成“灰白淡色层”，再按 outside_alpha 与原始 RGB 混合。
    # outside_fade  : 灰白层的泛白程度，越大越白；
    # outside_alpha : 灰白层的不透明度，越大研究区外越淡，越小保留越多原始色彩。
    gray = np.mean(rgb, axis=2, keepdims=True)
    gray_rgb = np.repeat(gray, 3, axis=2)
    pale_layer = gray_rgb * (1.0 - outside_fade) + np.ones_like(gray_rgb) * outside_fade
    outside = rgb * (1.0 - outside_alpha) + pale_layer * outside_alpha

    composed = outside * (1.0 - alpha[..., None]) + rgb * alpha[..., None]

    if transparent_outside:
        rgba = np.dstack([composed, alpha])
        return np.clip(rgba, 0, 1)
    return np.clip(composed, 0, 1)


def plot_study_area(
    *,
    hdf_path: str,
    tif_path: str,
    out_path: str,
    area_name: str | None = "Perimeter 1",
    max_size: int = 3500,
    feather: float = 6.0,
    outside_fade: float = 0.65,
    outside_alpha: float = 0.55,
    boundary_color: str = "black",
    boundary_lw: float = 1.4,
    boundary_ls: str = "--",
    halo_lw: float = 5.0,
    crop_to_perimeter: bool = False,
    crop_margin: float = 0.06,
    crop_margin_left: float | None = None,
    crop_margin_right: float | None = None,
    crop_margin_top: float | None = None,
    crop_margin_bottom: float | None = None,
    dpi: int = 300,
    transparent_outside: bool = False,
):
    per = read_perimeter_xy(hdf_path, area_name=area_name)
    rgb, extent, crs = read_tif_rgb(tif_path, max_size=max_size)
    mask = make_inside_mask(per, extent, rgb.shape[:2])
    img = compose_effect(
        rgb,
        mask,
        feather=feather,
        outside_fade=outside_fade,
        outside_alpha=outside_alpha,
        transparent_outside=transparent_outside,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    h, w = rgb.shape[:2]
    fig_w = 8.0
    fig_h = fig_w * h / max(w, 1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(img, extent=extent, origin="upper")

    # 边界光晕 + 主边界线，尽量接近示例图的柔和边界
    ax.plot(per[:, 0], per[:, 1], color="white", lw=halo_lw, alpha=0.30, solid_capstyle="round", zorder=3)
    ax.plot(per[:, 0], per[:, 1], color=boundary_color, lw=boundary_lw, ls=boundary_ls, alpha=0.95, solid_capstyle="round", zorder=4)

    if crop_to_perimeter:
        xmin, ymin = np.min(per[:, 0]), np.min(per[:, 1])
        xmax, ymax = np.max(per[:, 0]), np.max(per[:, 1])
        width = xmax - xmin
        height = ymax - ymin

        # 支持四个方向单独设置 margin。
        # 这样就可以解决“下方少一点、上方多一点”的问题：
        # 适当增大 bottom，减小 top。
        ml = crop_margin if crop_margin_left is None else crop_margin_left
        mr = crop_margin if crop_margin_right is None else crop_margin_right
        mt = crop_margin if crop_margin_top is None else crop_margin_top
        mb = crop_margin if crop_margin_bottom is None else crop_margin_bottom

        ax.set_xlim(xmin - width * ml, xmax + width * mr)
        ax.set_ylim(ymin - height * mb, ymax + height * mt)
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    ax.set_aspect("equal")
    ax.set_anchor("C")
    ax.margins(0)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # 这里不要再用 bbox_inches='tight'，否则 Matplotlib 会二次裁边，
    # 容易出现“底部少一点、顶部多一点”或边缘被吃掉的问题。
    fig.savefig(out_path, dpi=dpi, pad_inches=0, transparent=transparent_outside)
    plt.close(fig)

    print(f"[OK] 输出完成：{out_path}")
    print(f"[Info] TIF CRS: {crs}")
    print(f"[Info] TIF extent: {extent}")
    print(f"[Info] Perimeter extent: [{per[:,0].min():.3f}, {per[:,0].max():.3f}, {per[:,1].min():.3f}, {per[:,1].max():.3f}]")


def main():
    p = argparse.ArgumentParser("Generate study-area mask from HEC-RAS HDF perimeter and GeoTIFF")
    p.add_argument("--hdf", default='D:/Work/qyb/ResNet-18/data/test/hdf/01.hdf',help="包含 Geometry/2D Flow Areas/.../Perimeter 的 HEC-RAS HDF 文件")
    p.add_argument("--tif", default=r'D:\Work\qyb\ResNet-18\YD2.tif', help="GeoTIFF 遥感底图")
    p.add_argument("--out", default="Study_area6.png", help="输出 PNG 路径")
    p.add_argument("--area", default="Perimeter 1", help="2D Flow Area 名称；不确定时可先用默认值")
    p.add_argument("--max_size", type=int, default=3500, help="输出图最长边像素，越大越清晰但越慢")
    p.add_argument("--feather", type=float, default=6.0, help="边界柔化像素，0 表示不柔化")
    p.add_argument("--outside_fade", type=float, default=0.75, help="研究区外灰白层的泛白程度，0-1，越大越白")
    p.add_argument("--outside_alpha", type=float, default=0.55, help="研究区外灰白层的不透明度，0-1；越大越淡，越小保留越多底图色彩")
    p.add_argument("--boundary_color", default="black", help="边界线颜色")
    p.add_argument("--boundary_lw", type=float, default=1.4, help="边界线宽")
    p.add_argument("--boundary_ls", default="--", help="边界线样式，例如 --、-、:、-.")
    p.add_argument("--halo_lw", type=float, default=5.0, help="边界外发光宽度")
    p.add_argument("--crop_to_perimeter", action="store_true", help="只显示研究区附近范围，而不是完整 TIF 范围")
    p.add_argument("--crop_margin", type=float, default=0.06, help="统一外扩比例；若单独指定上下左右，则此值作为默认值")
    p.add_argument("--crop_margin_left", type=float, default=None, help="左侧单独外扩比例")
    p.add_argument("--crop_margin_right", type=float, default=None, help="右侧单独外扩比例")
    p.add_argument("--crop_margin_top", type=float, default=None, help="上侧单独外扩比例")
    p.add_argument("--crop_margin_bottom", type=float, default=None, help="下侧单独外扩比例")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--transparent_outside", action="store_true", help="研究区外透明，适合后续排版叠图")
    args = p.parse_args()

    plot_study_area(
        hdf_path=args.hdf,
        tif_path=args.tif,
        out_path=args.out,
        area_name=args.area,
        max_size=args.max_size,
        feather=args.feather,
        outside_fade=args.outside_fade,
        outside_alpha=args.outside_alpha,
        boundary_color=args.boundary_color,
        boundary_lw=args.boundary_lw,
        boundary_ls=args.boundary_ls,
        halo_lw=args.halo_lw,
        crop_to_perimeter=args.crop_to_perimeter,
        crop_margin=args.crop_margin,
        crop_margin_left=args.crop_margin_left,
        crop_margin_right=args.crop_margin_right,
        crop_margin_top=args.crop_margin_top,
        crop_margin_bottom=args.crop_margin_bottom,
        dpi=args.dpi,
        transparent_outside=args.transparent_outside,
    )


if __name__ == "__main__":
    main()

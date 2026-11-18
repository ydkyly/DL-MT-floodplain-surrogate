# -*- coding: utf-8 -*-
"""Shared flood visualisation utilities for training and inference. save_depth_maps writes prediction/reference/difference/mask panels per time step. Includes HDF coordinate loading, reference-file lookup, and Web Mercator conversion. Mask zero and nonfinite reference values to avoid black-background artefacts."""
import os, math, numpy as np, h5py, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .paths import XY_PATH
from .plotstyle import enforce_times_new_roman
enforce_times_new_roman()

R_EARTH = 6378137.0  # Web Mercator sphere radius.

# Basic utilities.

def find_gt_file(gt_dir: str, sid: str) -> str:
    for ext in ('.hdf', '.h5', '.hdf5'):
        p = os.path.join(gt_dir, f"{sid}{ext}")
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"未找到真值HDF：{os.path.join(gt_dir, sid)}.hdf(.h5/.hdf5)")

def load_xy_from_hdf(h5_path: str, expect_H: int | None = None, *, to_lonlat: bool = False):
    if not os.path.isfile(h5_path):
        return None
    with h5py.File(h5_path, "r") as h5:
        if XY_PATH not in h5:
            return None
        xy = h5[XY_PATH][:].astype(np.float32)  # [H,2]
    if expect_H is not None and xy.shape != (expect_H, 2):
        return None
    if to_lonlat:
        xy = webmerc_to_lonlat(xy)
    return xy

def webmerc_to_lonlat(xy_m: np.ndarray) -> np.ndarray:
    x = xy_m[:, 0].astype(np.float64)
    y = xy_m[:, 1].astype(np.float64)
    lon = x * 180.0 / (math.pi * R_EARTH)
    lat = np.degrees(2.0 * np.arctan(np.exp(y / R_EARTH)) - math.pi / 2.0)
    return np.stack([lon, lat], axis=1).astype(np.float32)

# Core plotting functions.

def _scatter_or_imshow(ax, data: np.ndarray, xy: np.ndarray | None, *,
                       cmap: str, vmin=None, vmax=None, cbar_label: str = "",
                       skip_zeros: bool = False):
    """Draw a point-cloud [H] or gridded [Ny,Nx] map with its colour bar. skip_zeros hides zero-valued points, useful for reference maps."""
    if xy is not None and data.ndim == 1:
        # Point cloud: mask nonfinite values and optionally zeros.
        mask = np.isfinite(data)
        if skip_zeros:
            mask &= (data != 0)
        sc = ax.scatter(xy[mask, 0], xy[mask, 1], c=data[mask], s=2,
                        cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        # Grid.
        sc = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(cbar_label)

def save_depth_maps(pred_frames: np.ndarray,
                    gt_frames: np.ndarray,
                    xy: np.ndarray | None = None,
                    *,
                    threshold: float = 0.1,
                    out_dir: str,
                    prefix: str = "",
                    combine: bool = True,
                    to_lonlat: bool = False,
                    cmap: str = "Blues",
                    diff_cmap: str = "RdBu_r",
                    mask_cmap: str = "gray_r",
                    dpi: int = 150):
    """Write four-panel or separate depth maps from [T,H] or [T,Ny,Nx] frames and optional [H,2] coordinates. Mask zero reference depths and colour differences only in the union of reference-wet and prediction-wet cells."""
    os.makedirs(out_dir, exist_ok=True)
    if xy is not None and to_lonlat:
        xy = webmerc_to_lonlat(xy)

    T = int(pred_frames.shape[0])
    label_x = "Longitude" if (xy is not None and to_lonlat) else ("X" if xy is not None else "Column")
    label_y = "Latitude"  if (xy is not None and to_lonlat) else ("Y" if xy is not None else "Row")

    for t in range(T):
        pred = np.asarray(pred_frames[t])
        gt   = np.asarray(gt_frames[t])
        # Ensure numeric values.
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        gt   = np.nan_to_num(gt,   nan=0.0, posinf=0.0, neginf=0.0)

        diff = pred - gt
        mask = (pred >= threshold).astype(np.uint8)

        # Display differences only in relevant regions so the background does not dominate the colour scale.
        if diff.ndim == 1:
            show_mask = (gt >= threshold) | (pred >= threshold)
            diff_show = diff.copy()
            diff_show[~show_mask] = np.nan
            # Robust colour limits based on valid-region quantiles.
            if np.any(np.isfinite(diff_show)):
                vmax = float(np.nanpercentile(np.abs(diff_show), 99))
                vmax = max(vmax, 1e-3)
            else:
                vmax = 1.0
        else:
            # Simplified gridded rendering.
            show_mask = None
            vmax = max(1e-3, float(np.percentile(np.abs(diff), 99)))

        if combine:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=dpi)
            (ax1, ax2), (ax3, ax4) = axes
            # Prediction: retain zeros to show predicted inundation.
            _scatter_or_imshow(ax1, pred, xy, cmap=cmap, cbar_label="Depth (m)")
            ax1.set_title(f"t={t} Pred")
            # Reference: omit zeros to avoid black-background artefacts.
            _scatter_or_imshow(ax2, gt, xy, cmap=cmap, cbar_label="Depth (m)", skip_zeros=True)
            ax2.set_title(f"t={t} GT")
            # Difference: colour only show_mask and set other cells to NaN.
            _scatter_or_imshow(ax3, (diff_show if diff.ndim==1 else diff),
                               xy, cmap=diff_cmap, vmin=-vmax, vmax=vmax, cbar_label="Pred - GT (m)")
            ax3.set_title(f"t={t} Diff (±{vmax:.2f} m)")
            # Mask
            _scatter_or_imshow(ax4, mask, xy, cmap=mask_cmap, cbar_label=f"Mask (>= {threshold} m)")
            ax4.set_title(f"t={t} Mask")
            for ax in (ax1, ax2, ax3, ax4):
                ax.set_xlabel(label_x); ax.set_ylabel(label_y)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{prefix}_t{t:02d}_4in1.png"))
            plt.close(fig)
        else:
            # Write four separate figures.
            maps = [
                ("pred", pred, cmap, "Depth (m)"),
                ("gt",   gt,   cmap, "Depth (m)"),
                ("diff", (diff_show if diff.ndim==1 else diff), diff_cmap, "Pred - GT (m)"),
                ("mask", mask, mask_cmap, f"Mask (>= {threshold} m)")
            ]
            for kind, data, cmap_use, cbar in maps:
                fig = plt.figure(figsize=(6,5), dpi=dpi)
                ax = fig.add_subplot(111)
                vmin = None; vmax_use = None
                if kind == "diff":
                    vmin, vmax_use = -vmax, vmax
                _scatter_or_imshow(ax, data, xy, cmap=cmap_use, vmin=vmin, vmax=vmax_use,
                                   cbar_label=cbar, skip_zeros=(kind=="gt"))
                ax.set_title(f"t={t} {kind.upper()}")
                ax.set_xlabel(label_x); ax.set_ylabel(label_y)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"{prefix}_t{t:02d}_{kind}.png"))
                plt.close(fig)

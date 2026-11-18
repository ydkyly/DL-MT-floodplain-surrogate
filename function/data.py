# -*- coding: utf-8 -*-
import os, numpy as np, pandas as pd, h5py, torch
from torch.utils.data import Dataset
from .paths import DEPTH_PATH, VX_PATH, VY_PATH


class FloodDataset(Dataset):
    """Load N input workbooks of shape [24,26] and matching HDF/NPY labels. Return x:[26,24], normalised using training-set global min/max; y:[T,H] for depth-only labels or a depth/vx/vy dictionary of [T,H] tensors when all HDF variables are present; and sid as a string sample ID."""

    def __init__(self, in_dir, tar_dir=None, *, dtype=np.float32,
                 timesteps=24, is_train=True, ref_min=None, ref_max=None):
        self.dtype, self.timesteps = dtype, timesteps
        self.in_dir, self.tar_dir = in_dir, tar_dir
        self.ids = sorted(
            [f for f in os.listdir(in_dir) if f.endswith('.xlsx')],
            key=lambda s: int(os.path.splitext(s)[0])
        )
        assert self.ids, f"{in_dir} 中找不到 .xlsx"

        # Normalisation statistics.
        if is_train:
            rows = []
            for fname in self.ids:
                arr = pd.read_excel(os.path.join(in_dir, fname),
                                    header=None).to_numpy(dtype)
                rows.append(arr)
            big = np.concatenate(rows, axis=0)   # [N*24, 26]
            self.x_min = big.min(0)
            self.x_max = big.max(0)
        else:
            assert ref_min is not None and ref_max is not None, \
                "验证/推理集必须提供 ref_min/ref_max"
            self.x_min = np.asarray(ref_min, dtype=dtype)
            self.x_max = np.asarray(ref_max, dtype=dtype)

        # These flags suppress repeated missing-vx/vy warnings.
        self._warned_no_vx_vy = False

    # -----------------------------
    # Read and normalise Excel inputs.
    # -----------------------------
    def _read_excel(self, path):
        raw = pd.read_excel(path, header=None).to_numpy(dtype=self.dtype)  # [24,26]
        eps = 1e-8
        norm = (raw - self.x_min) / (self.x_max - self.x_min + eps)
        return norm.T  # [26,24]

    # -----------------------------
    # Generic HDF reader.
    # -----------------------------
    def _read_hdf_field(self, h5_path, data_path, round_decimals=None):
        """Read a [T,H] array from an HDF dataset, replacing NaN/Inf with zero. data_path selects DEPTH_PATH, VX_PATH, or VY_PATH; round_decimals optionally controls rounding."""
        with h5py.File(h5_path, "r") as h5:
            if data_path not in h5:
                raise KeyError(f"{h5_path} 中缺少数据路径：{data_path}")
            arr = h5[data_path][:self.timesteps].astype(self.dtype)  # [T,H]

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if round_decimals is not None:
            arr = np.round(arr, round_decimals)
        return arr

    def _read_hdf_depth_only(self, h5_path):
        """Backward-compatible reader for depth only."""
        return self._read_hdf_field(h5_path, DEPTH_PATH, round_decimals=2)

    # -----------------------------
    # Legacy format with one .npy file per time step.
    # -----------------------------
    def _read_dir_npy(self, d):
        files = sorted(
            [f for f in os.listdir(d) if f.endswith('.npy')],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        vecs = [np.load(os.path.join(d, f)).astype(self.dtype)
                for f in files[:self.timesteps]]
        return np.stack(vecs, 0)  # [T,H]

    # -----------------------------
    # Dataset interface.
    # -----------------------------
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = os.path.splitext(self.ids[idx])[0]

        # x: [26,24]
        x_np = self._read_excel(os.path.join(self.in_dir, f"{sid}.xlsx"))
        x = torch.from_numpy(x_np).float()

        y = None
        if self.tar_dir is not None:
            h5_path  = os.path.join(self.tar_dir, f"{sid}.hdf")
            npy_path = os.path.join(self.tar_dir, f"{sid}.npy")
            dir_path = os.path.join(self.tar_dir, sid)

            # Prefer HDF, supporting multi-task labels.
            if os.path.isfile(h5_path):
                # Read required water-depth labels first.
                depth_np = self._read_hdf_field(
                    h5_path, DEPTH_PATH, round_decimals=2
                )  # [T,H]

                vx_np = vy_np = None
                # Attempt to read optional vx/vy labels.
                try:
                    vx_np = self._read_hdf_field(h5_path, VX_PATH)
                    vy_np = self._read_hdf_field(h5_path, VY_PATH)
                except KeyError:
                    # Fall back to depth-only labels if either velocity component is missing; warn once.
                    if not self._warned_no_vx_vy:
                        print(f"[FloodDataset][Warn] HDF {h5_path} 中缺少 VX_PATH 或 VY_PATH，"
                              f"本次数据集将仅使用水深标签。")
                        self._warned_no_vx_vy = True
                    vx_np = vy_np = None

                # Assemble y.
                if (vx_np is not None) and (vy_np is not None):
                    # Multi-task output is a dictionary.
                    y = {
                        "depth": torch.from_numpy(depth_np).float(),   # [T,H]
                        "vx":    torch.from_numpy(vx_np).float(),      # [T,H]
                        "vy":    torch.from_numpy(vy_np).float(),      # [T,H]
                    }
                else:
                    # Single-task output contains only depth.
                    y = torch.from_numpy(depth_np).float()  # [T,H]

            # Legacy single-.npy format stores depth only.
            elif os.path.isfile(npy_path):
                y_np = np.load(npy_path).astype(self.dtype)[:self.timesteps]
                y_np = np.nan_to_num(y_np, nan=0.0, posinf=0.0, neginf=0.0)
                y = torch.from_numpy(y_np).float()  # [T,H]

            # Legacy per-time-step .npy directory stores depth only.
            elif os.path.isdir(dir_path):
                y_np = self._read_dir_npy(dir_path)  # [T,H]
                y_np = np.nan_to_num(y_np, nan=0.0, posinf=0.0, neginf=0.0)
                y = torch.from_numpy(y_np).float()  # [T,H]

            else:
                raise FileNotFoundError(f"{sid} 无对应标签(.hdf/.npy/子目录)")

        return x, y, sid

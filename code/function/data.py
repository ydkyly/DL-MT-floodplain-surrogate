# -*- coding: utf-8 -*-
import os, numpy as np, pandas as pd, h5py, torch
from torch.utils.data import Dataset
from .paths import DEPTH_PATH, VX_PATH, VY_PATH


class FloodDataset(Dataset):
    """
    输入 in_dir  : N 个 24×26 的 .xlsx
    标签 tar_dir: 同名 .hdf / .npy / 子目录（每步一npy）

    产出:
      x : [26, 24]   (C=26, L=24)，已经按训练集全局 min/max 做了 [0,1] 归一化
      y :
        - 若仅有水深标签：Tensor [T, H]（原有单任务形式）
        - 若 HDF 中同时存在 DEPTH_PATH / VX_PATH / VY_PATH：
            dict {
              "depth": Tensor [T, H],
              "vx":    Tensor [T, H],
              "vy":    Tensor [T, H],
            }
      sid : 样本ID（文件名前缀，字符串）
    """

    def __init__(self, in_dir, tar_dir=None, *, dtype=np.float32,
                 timesteps=24, is_train=True, ref_min=None, ref_max=None):
        self.dtype, self.timesteps = dtype, timesteps
        self.in_dir, self.tar_dir = in_dir, tar_dir
        self.ids = sorted(
            [f for f in os.listdir(in_dir) if f.endswith('.xlsx')],
            key=lambda s: int(os.path.splitext(s)[0])
        )
        assert self.ids, f"{in_dir} 中找不到 .xlsx"

        # 归一化统计
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

        # 这两个标志仅用于打印一次缺少 vx/vy 的提醒
        self._warned_no_vx_vy = False

    # -----------------------------
    # 输入 Excel 读取 + 归一化
    # -----------------------------
    def _read_excel(self, path):
        raw = pd.read_excel(path, header=None).to_numpy(dtype=self.dtype)  # [24,26]
        eps = 1e-8
        norm = (raw - self.x_min) / (self.x_max - self.x_min + eps)
        return norm.T  # [26,24]

    # -----------------------------
    # 通用 HDF 读取函数
    # -----------------------------
    def _read_hdf_field(self, h5_path, data_path, round_decimals=None):
        """
        从 HDF 文件中按指定路径读取 [T,H] 数组，并做 NaN/Inf→0 处理。

        data_path      : HDF 内数据集路径（DEPTH_PATH / VX_PATH / VY_PATH）
        round_decimals : 若不为 None，对结果做 np.round(..., round_decimals)
        """
        with h5py.File(h5_path, "r") as h5:
            if data_path not in h5:
                raise KeyError(f"{h5_path} 中缺少数据路径：{data_path}")
            arr = h5[data_path][:self.timesteps].astype(self.dtype)  # [T,H]

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if round_decimals is not None:
            arr = np.round(arr, round_decimals)
        return arr

    def _read_hdf_depth_only(self, h5_path):
        """向后兼容：仅读取水深字段。"""
        return self._read_hdf_field(h5_path, DEPTH_PATH, round_decimals=2)

    # -----------------------------
    # 目录中每步一个 npy 的老格式
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
    # Dataset 接口
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

            # 优先：HDF（支持多任务标签）
            if os.path.isfile(h5_path):
                # 1) 先读水深（必需）
                depth_np = self._read_hdf_field(
                    h5_path, DEPTH_PATH, round_decimals=2
                )  # [T,H]

                vx_np = vy_np = None
                # 2) 尝试读取 vx / vy（可选）
                try:
                    vx_np = self._read_hdf_field(h5_path, VX_PATH)
                    vy_np = self._read_hdf_field(h5_path, VY_PATH)
                except KeyError:
                    # 如果缺少 vx 或 vy，则退回单任务水深，并只打印一次警告
                    if not self._warned_no_vx_vy:
                        print(f"[FloodDataset][Warn] HDF {h5_path} 中缺少 VX_PATH 或 VY_PATH，"
                              f"本次数据集将仅使用水深标签。")
                        self._warned_no_vx_vy = True
                    vx_np = vy_np = None

                # 3) 组装 y
                if (vx_np is not None) and (vy_np is not None):
                    # 多任务：返回 dict
                    y = {
                        "depth": torch.from_numpy(depth_np).float(),   # [T,H]
                        "vx":    torch.from_numpy(vx_np).float(),      # [T,H]
                        "vy":    torch.from_numpy(vy_np).float(),      # [T,H]
                    }
                else:
                    # 单任务：只用水深
                    y = torch.from_numpy(depth_np).float()  # [T,H]

            # 旧格式：单个 .npy 只存水深
            elif os.path.isfile(npy_path):
                y_np = np.load(npy_path).astype(self.dtype)[:self.timesteps]
                y_np = np.nan_to_num(y_np, nan=0.0, posinf=0.0, neginf=0.0)
                y = torch.from_numpy(y_np).float()  # [T,H]

            # 旧格式：子目录中每步一份 .npy，只存水深
            elif os.path.isdir(dir_path):
                y_np = self._read_dir_npy(dir_path)  # [T,H]
                y_np = np.nan_to_num(y_np, nan=0.0, posinf=0.0, neginf=0.0)
                y = torch.from_numpy(y_np).float()  # [T,H]

            else:
                raise FileNotFoundError(f"{sid} 无对应标签(.hdf/.npy/子目录)")

        return x, y, sid

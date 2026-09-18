# -*- coding: utf-8 -*-
"""
sanity.py
数据与输出的健壮性工具：
 - check_nan_inf: 统计/修复 NaN 与 Inf
 - enforce_nonnegative: 施加非负约束（<0→0）
 - round_tensor/round_array: 四舍五入到指定小数位
 - sanitize_pred_for_save: 预测结果保存前的一键处理（修复NaN/Inf→非负→三位小数）
"""
from __future__ import annotations
import numpy as np
import torch

def check_nan_inf(x, name: str = "", *, raise_on_nan: bool = False, fix: bool = True):
    """
    统计 & （可选）修复 NaN/Inf。
    - Tensor: 用 torch.nan_to_num(0)
    - ndarray: 用 np.nan_to_num(0)
    返回与传入类型一致的对象（若 fix=True 则为修复后的对象）。
    """
    if isinstance(x, torch.Tensor):
        nan = torch.isnan(x).sum().item()
        inf = torch.isinf(x).sum().item()
        if nan or inf:
            msg = f"[NaN/Inf] {name}: nan={nan}, inf={inf}, shape={tuple(x.shape)}"
            if fix:
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                print(msg + "  -> fixed by torch.nan_to_num(0)")
            elif raise_on_nan:
                raise ValueError(msg)
            else:
                print(msg)
        return x
    else:
        arr = np.asarray(x)
        nan = np.isnan(arr).sum()
        inf = np.isinf(arr).sum()
        if nan or inf:
            msg = f"[NaN/Inf] {name}: nan={nan}, inf={inf}, shape={arr.shape}"
            if fix:
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                print(msg + "  -> fixed by np.nan_to_num(0)")
            elif raise_on_nan:
                raise ValueError(msg)
            else:
                print(msg)
        return arr

# sanity.py
def enforce_nonnegative(x, tol: float = 1e-6, *, return_count: bool = False):
    """
    深度 >= 0 约束；允许极小负值（|x|<tol）视作 0。
    return_count=True 时返回 (clamped_x, neg_count)，否则只返回 clamped_x。
    """
    if isinstance(x, torch.Tensor):
        if return_count:
            neg_cnt = torch.count_nonzero(x < -tol).item()
            x = torch.clamp(x, min=0.0)
            return x, neg_cnt
        else:
            return torch.clamp(x, min=0.0)
    else:
        arr = np.asarray(x)
        if return_count:
            neg_cnt = int((arr < -tol).sum())
            arr = np.maximum(arr, 0.0)
            return arr, neg_cnt
        else:
            return np.maximum(arr, 0.0)


def round_tensor(x: torch.Tensor, decimals: int = 3) -> torch.Tensor:
    if decimals <= 0: return torch.round(x)
    factor = 10.0 ** decimals
    return torch.round(x * factor) / factor

def round_array(x: np.ndarray, decimals: int = 3) -> np.ndarray:
    return np.round(np.asarray(x), decimals).astype(np.float32)

def sanitize_pred_for_save(pred, *, decimals: int = 3, tol: float = 1e-6):
    """
    保存/出图前的一键处理：修复 NaN/Inf → 非负约束 → 三位小数
    返回 numpy.float32 数组。
    """
    pred = check_nan_inf(pred, "pred", raise_on_nan=False, fix=True)
    pred = enforce_nonnegative(pred, tol=tol)
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    pred = round_array(pred, decimals=decimals)
    return pred

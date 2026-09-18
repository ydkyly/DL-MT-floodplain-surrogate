# -*- coding: utf-8 -*-
import os
import glob
import csv
import h5py
import numpy as np

# ====== 直接内嵌路径（请按需改成你的实际路径）======

# 方案1：精确指定某个 diff.npy（可留空）
DIFF_PATH = r""

# 预测文件（必填）
Pred_PATH = r'D:\Work\qyb\ResNet-18\data_dw\Infer_SVD_resnet\01_Y_pred.npy'

# 真值优先文件：
# 1) 如果这里填的是存在的 .npy / .hdf / .h5 / .hdf5，就直接读取它
# 2) 如果这里为空，或文件不存在，则自动按 Pred_PATH 的 sid 查找 true.npy
GT_PATH = r''

# 当 GT_PATH 不存在时，可在这个目录里按 sid 自动查找：
#   05_Y_true.npy / 05_true.npy / 05_gt.npy
# 若还找不到，则继续查找：
#   05.hdf / 05.h5 / 05.hdf5
GT_DIR = r'D:\Work\qyb\ResNet-18\data_dw\test\hdf'

# 方案2：如果上面 DIFF_PATH 不存在，则在这个目录里自动找最新的 *_diff.npy
BASE_DIR = r""

# HDF 中水深数据集路径（与你当前项目一致）
HDF_DATA_PATH = (
    "Results/Unsteady/Output/Output Blocks/Base Output/"
    "Unsteady Time Series/2D Flow Areas/Perimeter 1/Cell Hydraulic Depth"
)

# 误差阈值（单位：米）
THR_LOW = 0.1
THR_HIGH = 1
# ===============================================


# -----------------------------
# 基础工具
# -----------------------------
def find_latest_diff(base_dir: str):
    if not base_dir or not os.path.isdir(base_dir):
        return None
    pattern = os.path.join(base_dir, "**", "*_diff.npy")
    cand = glob.glob(pattern, recursive=True)
    if not cand:
        return None
    cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cand[0]


def to_percent(x, total):
    return 100.0 * x / max(1, total)


def fmt_thr(x: float) -> str:
    """阈值打印更紧凑：0.01 -> 0.01，0.1 -> 0.1"""
    return f"{x:g}"


def strip_pred_suffix(stem: str) -> str:
    """
    从预测文件名中提取 sid
    例如：
      05_Y_pred.npy              -> 05
      05_Y_pred_raw.npy          -> 05
      05_Y_pred_sanitized.npy    -> 05
      05_pred.npy                -> 05
    """
    suffixes = [
        "_Y_pred_raw",
        "_Y_pred_sanitized",
        "_Y_pred",
        "_pred_raw",
        "_pred_sanitized",
        "_pred",
    ]
    for suf in suffixes:
        if stem.endswith(suf):
            return stem[:-len(suf)]
    return stem


def possible_true_npy_paths(sid: str, dirs):
    names = [
        f"{sid}_Y_true.npy",
        f"{sid}_true.npy",
        f"{sid}_gt.npy",
        f"{sid}.npy",  # 兼容极简命名
    ]
    out = []
    for d in dirs:
        if d and os.path.isdir(d):
            out.extend([os.path.join(d, n) for n in names])
    return out


def possible_hdf_paths(sid: str, dirs):
    exts = [".hdf", ".h5", ".hdf5"]
    out = []
    for d in dirs:
        if d and os.path.isdir(d):
            for ext in exts:
                out.append(os.path.join(d, sid + ext))
    return out


def load_hdf_array(h5_path: str, data_path: str, dtype=np.float32):
    with h5py.File(h5_path, "r") as h5:
        if data_path not in h5:
            raise KeyError(f"HDF 文件中找不到数据集路径：{data_path}\n文件：{h5_path}")
        arr = h5[data_path][:]
    arr = np.asarray(arr, dtype=dtype)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def load_array_auto(path: str, *, hdf_data_path: str, dtype=np.float32):
    """
    自动读取 .npy / .hdf / .h5 / .hdf5
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path).astype(dtype)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr
    if ext in (".hdf", ".h5", ".hdf5"):
        return load_hdf_array(path, hdf_data_path, dtype=dtype)
    raise ValueError(f"暂不支持的文件类型：{path}")


def resolve_gt_path(pred_path: str, gt_path: str, gt_dir: str, hdf_data_path: str):
    """
    真值解析逻辑：
    1) 如果 GT_PATH 存在，则直接使用 GT_PATH（支持 npy/hdf）
    2) 否则根据 pred 的 sid，优先找 true.npy
    3) 若 true.npy 不存在，则找对应 sid 的 hdf/h5/hdf5
    """
    # 1) GT_PATH 直接可用
    if gt_path and os.path.isfile(gt_path):
        print(f"[Info] 使用显式指定 GT_PATH：{gt_path}")
        return gt_path

    pred_stem = os.path.splitext(os.path.basename(pred_path))[0]
    sid = strip_pred_suffix(pred_stem)

    # 搜索目录优先级
    search_dirs = []
    if gt_path:
        gt_parent = os.path.dirname(gt_path)
        if gt_parent:
            search_dirs.append(gt_parent)
    if gt_dir:
        search_dirs.append(gt_dir)
    pred_dir = os.path.dirname(pred_path)
    if pred_dir:
        search_dirs.append(pred_dir)

    # 去重但保序
    uniq_dirs = []
    seen = set()
    for d in search_dirs:
        if d and d not in seen:
            uniq_dirs.append(d)
            seen.add(d)

    # 2) 优先找 true.npy
    for p in possible_true_npy_paths(sid, uniq_dirs):
        if os.path.isfile(p):
            print(f"[Info] 未找到显式 GT_PATH，自动匹配到 true.npy：{p}")
            return p

    # 3) true.npy 不存在，则找对应 hdf
    for p in possible_hdf_paths(sid, uniq_dirs):
        if os.path.isfile(p):
            print(f"[Info] 未找到 true.npy，自动回退到 HDF：{p}")
            return p

    raise FileNotFoundError(
        f"未找到与预测文件对应的真值文件。\n"
        f"Pred_PATH = {pred_path}\n"
        f"解析得到 sid = {sid}\n"
        f"已搜索目录 = {uniq_dirs}\n"
        f"尝试过 *_Y_true.npy / *_true.npy / *_gt.npy 以及 .hdf/.h5/.hdf5"
    )


def load_diff():
    path = DIFF_PATH
    if not path or not os.path.isfile(path):
        if path:
            print(f"[Info] 指定 DIFF_PATH 不存在：{path}")
        latest = find_latest_diff(BASE_DIR)
        if latest is None:
            raise FileNotFoundError(f"在 {BASE_DIR} 下未找到任何 *_diff.npy")
        print(f"[Info] 使用最新的 diff 文件：{latest}")
        path = latest
    else:
        print(f"[Info] 使用指定 DIFF_PATH：{path}")

    diff = np.load(path)
    if diff.ndim != 2:
        raise ValueError(f"期望 diff 为二维 (T,H)，实际 {diff.shape}")
    return path, diff


# -----------------------------
# 主流程
# -----------------------------
def main():
    if not os.path.isfile(Pred_PATH):
        raise FileNotFoundError(f"预测文件不存在：{Pred_PATH}")

    # 读取 pred
    pred = load_array_auto(Pred_PATH, hdf_data_path=HDF_DATA_PATH, dtype=np.float32)

    # 读取 GT：优先 true.npy，不存在时自动回退到对应 hdf
    gt_real_path = resolve_gt_path(
        pred_path=Pred_PATH,
        gt_path=GT_PATH,
        gt_dir=GT_DIR,
        hdf_data_path=HDF_DATA_PATH,
    )
    GT = load_array_auto(gt_real_path, hdf_data_path=HDF_DATA_PATH, dtype=np.float32)

    # 形状检查
    if pred.ndim != 2:
        raise ValueError(f"Pred 期望为二维 (T,H)，实际 {pred.shape}")
    if GT.ndim != 2:
        raise ValueError(f"GT 期望为二维 (T,H)，实际 {GT.shape}")
    if pred.shape != GT.shape:
        raise ValueError(f"Pred 与 GT 形状不一致：pred={pred.shape}, GT={GT.shape}")

    diff = np.subtract(pred, GT, dtype=np.float32)
    err = np.abs(np.nan_to_num(diff, nan=np.inf, posinf=np.inf, neginf=np.inf))

    T, H = err.shape
    b0, b1 = float(THR_LOW), float(THR_HIGH)

    # —— 逐时间步统计 ——
    per_step = []
    for t in range(T):
        e = err[t]
        c_lt = int(np.sum(e < b0))
        c_mid = int(np.sum((e >= b0) & (e < b1)))
        c_gt = int(np.sum(e >= b1))
        per_step.append((t, c_lt, c_mid, c_gt, H))

    # —— 全时段汇总 ——
    e_all = err.reshape(-1)
    total = int(e_all.size)
    all_lt = int(np.sum(e_all < b0))
    all_mid = int(np.sum((e_all >= b0) & (e_all < b1)))
    all_gt = int(np.sum(e_all >= b1))

    # —— 打印结果 ——
    print(f"\nPred file : {Pred_PATH}")
    print(f"GT file   : {gt_real_path}")
    print(f"Shape     : T={T}, H={H}")
    print(f"Thresholds: <{fmt_thr(b0)} m, [{fmt_thr(b0)},{fmt_thr(b1)}) m, ≥{fmt_thr(b1)} m")

    header = (
        f"{'t':>3} | "
        f"{('<' + fmt_thr(b0) + ' m'):>10} "
        f"{('[' + fmt_thr(b0) + ',' + fmt_thr(b1) + ') m'):>14} "
        f"{('≥' + fmt_thr(b1) + ' m'):>10} | "
        f"{'H':>7} | "
        f"{'<thr %':>7} {'mid %':>7} {'>thr %':>7}"
    )
    print(header)
    print("-" * len(header))

    for t, c1, c2, c3, h in per_step:
        print(
            f"{t:3d} | {c1:10d} {c2:14d} {c3:10d} | {h:7d} | "
            f"{to_percent(c1, h):7.2f} {to_percent(c2, h):7.2f} {to_percent(c3, h):7.2f}"
        )

    p1 = to_percent(all_lt, total)
    p2 = to_percent(all_mid, total)
    p3 = to_percent(all_gt, total)

    print("\nOverall across all T×H:")
    print(f"  total cells           : {total}")
    print(f"  < {fmt_thr(b0)} m           : {all_lt} ({p1:.2f} %)")
    print(f"  [{fmt_thr(b0)},{fmt_thr(b1)}) m      : {all_mid} ({p2:.2f} %)")
    print(f"  ≥ {fmt_thr(b1)} m           : {all_gt} ({p3:.2f} %)")

    # —— 导出 CSV ——
    out_csv = os.path.splitext(Pred_PATH)[0] + f"_stat_{b0:.2f}_{b1:.2f}.csv"
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pred_file", Pred_PATH])
        w.writerow(["gt_file", gt_real_path])
        w.writerow(["shape", f"T={T},H={H}"])
        w.writerow([])
        w.writerow([
            "t",
            f"<{b0}m_count",
            f"[{b0},{b1})m_count",
            f">={b1}m_count",
            "H",
            f"<{b0}m_pct(%)",
            f"[{b0},{b1})m_pct(%)",
            f">={b1}m_pct(%)"
        ])
        for t, c1, c2, c3, h in per_step:
            w.writerow([
                t, c1, c2, c3, h,
                round(to_percent(c1, h), 4),
                round(to_percent(c2, h), 4),
                round(to_percent(c3, h), 4)
            ])
        w.writerow([])
        w.writerow([
            "ALL(T×H)",
            all_lt, all_mid, all_gt, total,
            round(p1, 4), round(p2, 4), round(p3, 4)
        ])

    print(f"\nSaved CSV → {out_csv}")


if __name__ == "__main__":
    main()
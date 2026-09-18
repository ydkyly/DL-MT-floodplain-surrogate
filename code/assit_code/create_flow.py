# -*- coding: utf-8 -*-
"""
平滑 24h 洪水流量过程生成器
- 每小时一个点，共 24 点
- 洪峰值在 [3000, 15000] 内，互不重复
- 曲线为单峰(高斯)或双峰(两高斯叠加)且自然平滑
- 起点不一致：随机基流占峰值 5%~30%
- 前 20 条曲线绘图为 PNG（中文宋体，英文 Times New Roman）
- 全部曲线保存为 Excel（第一列为 0~23 小时）
- 提供任意曲线倍比放大函数
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ================= 基本参数 =================
N_CURVES = 100
PEAK_MIN, PEAK_MAX = 3000, 12000
PNG_PATH = "first20_curves1.png"
XLSX_PATH = "flood_curves1.xlsx"
RANDOM_SEED = None          # 可设为 int 以复现实验，如 42

# ================ 字体设置（Times + 宋体） ================
rcParams['font.family'] = ['Times New Roman', 'SimSun']  # 英文/数字优先 Times；中文回退宋体
rcParams['axes.unicode_minus'] = False
rcParams['mathtext.fontset'] = 'stix'
rcParams['mathtext.rm'] = 'Times New Roman'

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

# ================ 曲线生成 ================
def _gauss(t, mu, sigma):
    """标准高斯核（峰值=1）。"""
    return np.exp(-0.5 * ((t - mu) / sigma) ** 2)

def generate_smooth_curve(peak_value: float, double_peak: bool = False) -> np.ndarray:
    """
    生成一条平滑的 24h 洪水过程。
    - peak_value: 目标洪峰（最终曲线最大值会被归一至该峰值）
    - double_peak: 是否双峰（两个高斯叠加）
    返回：长度 24 的 numpy 数组
    """
    hours = np.arange(24)

    # 随机基流：峰值的 5% ~ 30%，确保起点不一致、曲线不从零开始
    base_flow = random.uniform(0.05 * peak_value, 0.15 * peak_value)

    if not double_peak:
        # 单峰：一个高斯，宽度 2~5 小时，峰现 6~18 时
        t_peak = random.randint(6, 18)
        sigma = random.uniform(2.0, 5.0)
        g = _gauss(hours, t_peak, sigma)
        # 初曲线：基流 + 振幅 * g，后续统一归一到 peak_value
        curve = base_flow + (peak_value - base_flow) * g
    else:
        # 双峰：两个高斯叠加，峰位错开 ≥3 小时
        t1 = random.randint(4, 12)
        t2 = random.randint(t1 + 3, 21)
        sigma1 = random.uniform(2.0, 4.5)
        sigma2 = random.uniform(2.0, 4.5)

        g1 = _gauss(hours, t1, sigma1)
        g2 = _gauss(hours, t2, sigma2)

        # 两个峰的相对权重（先随机构造，再整体归一到目标峰值）
        w1 = random.uniform(0.6, 1.0)   # 主峰权
        w2 = random.uniform(0.4, 0.9)   # 次峰权
        raw = w1 * g1 + w2 * g2
        # 防止两个峰太近导致过尖：做一次轻微平滑卷积
        kern = np.array([0.2, 0.6, 0.2])
        raw = np.convolve(raw, kern, mode='same')

        # 归一化到 [0, 1] 再拉到目标峰值
        raw_max = raw.max()
        if raw_max < 1e-8:
            raw = g1  # 极小概率防守
            raw_max = raw.max()
        curve = base_flow + (peak_value - base_flow) * (raw / raw_max)

    # 数值清理：不为负，确保最大值精确等于 peak_value
    curve = np.clip(curve, 0, None)
    # 归一一次，确保 max 为 peak_value（尤其双峰叠加后数值稳定）
    cmax = curve.max()
    if cmax > 0:
        curve = base_flow + (peak_value - base_flow) * (curve - base_flow) / (cmax - base_flow + 1e-9)
    # 起止小时不强行相等，保持自然
    return curve

# ================ 生成全集 ================
# 在 [PEAK_MIN, PEAK_MAX] 取 N_CURVES 个互不重复的整数洪峰
peak_values = random.sample(range(PEAK_MIN, PEAK_MAX + 1), N_CURVES)

curves = []
for pv in peak_values:
    is_double = (random.random() < 0.5)  # 约 50% 概率双峰
    curves.append(generate_smooth_curve(pv, double_peak=is_double))
curves = np.vstack(curves)   # 形状 (N_CURVES, 24)

# ================ 绘图（前 20 条） ================
hours = np.arange(24)
fig, ax = plt.subplots(figsize=(8.2, 6.0), dpi=120)
for i in range(min(20, N_CURVES)):
    ax.plot(hours, curves[i, :], linewidth=2.0, alpha=0.95, label=f"Curve {i+1}")

ax.set_xlabel("Time(h)", fontsize=12)
ax.set_ylabel("Flow($m^3/s$)", fontsize=12)
ax.set_title("前20条洪水过程曲线（平滑）", fontsize=14)
ax.grid(True, linestyle="--", alpha=0.35)
# 如需图例可开启（曲线多时可关闭避免遮挡）
# ax.legend(ncol=2, fontsize=8, frameon=False)
plt.tight_layout()
fig.savefig(PNG_PATH, dpi=300)

# ================ 写 Excel（每列一条曲线） ================
df = pd.DataFrame({"Hour": hours})
for i in range(N_CURVES):
    df[f"Curve{i+1}"] = curves[i, :].round(2)  # 保留两位小数，也可改为整数 .round(0)
df.to_excel(XLSX_PATH, index=False)

# ================ 倍比放大功能 ================
def scale_curve(curves_matrix: np.ndarray, curve_index: int, factor: float) -> np.ndarray:
    """
    对第 curve_index 条曲线做倍比放大（不改变其他曲线）。
    返回一条新的 24 长度数组。
    """
    assert 0 <= curve_index < curves_matrix.shape[0], "curve_index 超界"
    return curves_matrix[curve_index, :] * float(factor)

# ====== 使用示例：将第 1 条曲线放大 1.5 倍并另存为图 ======
if __name__ == "__main__":
    idx = 0
    scaled = scale_curve(curves, idx, 1.5)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(hours, curves[idx, :], label="原曲线", linewidth=2.2)
    ax2.plot(hours, scaled, label="放大 1.5×", linewidth=2.2, linestyle="--")
    ax2.set_xlabel("Time(h)", fontsize=12)
    ax2.set_ylabel("Flow($m^3/s$)", fontsize=12)
    ax2.set_title("单条曲线倍比放大示例", fontsize=13)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(frameon=False)
    plt.tight_layout()
    fig2.savefig("scaled_example.png", dpi=300)

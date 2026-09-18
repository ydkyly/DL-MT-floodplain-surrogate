import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# -----------------------------
# 1) 读入你给的 24 时刻数据
# -----------------------------
txt = r"""
t	RMSE_depth	MAE_depth	PCC_depth
0	0.951 	0.390 	0.881
1	0.758 	0.306 	0.891
2	0.591 	0.225 	0.930
3	0.438 	0.162 	0.963
4	0.327 	0.124 	0.984
5	0.304 	0.118 	0.992
6	0.308 	0.131 	0.995
7	0.321 	0.144 	0.996
8	0.331 	0.150 	0.996
9	0.335 	0.153 	0.995
10	0.333 	0.152 	0.995
11	0.320 	0.146 	0.995
12	0.303 	0.138 	0.995
13	0.285 	0.128 	0.995
14	0.264 	0.118 	0.996
15	0.244 	0.109 	0.996
16	0.224 	0.099 	0.996
17	0.205 	0.090 	0.997
18	0.188 	0.082 	0.997
19	0.173 	0.074 	0.997
20	0.159 	0.067 	0.997
21	0.148 	0.061 	0.997
22	0.140 	0.055 	0.997
23	0.134 	0.049 	0.997
"""
df = pd.read_csv(StringIO(txt), sep=r"\s+", engine="python")

rmse = df["RMSE_depth"].to_numpy()
mae  = df["MAE_depth"].to_numpy()
pcc  = df["PCC_depth"].to_numpy()

# -----------------------------
# 2) 画箱线图：左轴 RMSE/MAE，右轴 PCC
# -----------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

fig, ax1 = plt.subplots(figsize=(8.0, 5.0), dpi=300)
ax2 = ax1.twinx()

pos_rmse, pos_mae, pos_pcc = 1, 2, 3
width = 0.55

# 左轴：RMSE & MAE
bp_left = ax1.boxplot(
    [rmse, mae],
    positions=[pos_rmse, pos_mae],
    widths=width,
    patch_artist=True,
    showfliers=False
)

# 右轴：PCC
bp_right = ax2.boxplot(
    [pcc],
    positions=[pos_pcc],
    widths=width,
    patch_artist=True,
    showfliers=False
)

# （可选）叠加散点，展示 24 个时刻的分布（不影响箱线图）
rng = np.random.default_rng(0)
jitter = 0.05
ax1.scatter(rng.normal(pos_rmse, jitter, size=rmse.size), rmse, s=10, alpha=0.6)
ax1.scatter(rng.normal(pos_mae,  jitter, size=mae.size),  mae,  s=10, alpha=0.6)
ax2.scatter(rng.normal(pos_pcc,  jitter, size=pcc.size),  pcc,  s=10, alpha=0.6)

# 轴与标签
ax1.set_ylabel("RMSE / MAE")
ax2.set_ylabel("PCC")

ax1.set_xticks([pos_rmse, pos_mae, pos_pcc])
ax1.set_xticklabels(["RMSE", "MAE", "PCC"])

ax1.grid(True, axis="y", alpha=0.3)

# 让右轴更“贴合”PCC范围（可按需注释掉）
ax2.set_ylim(max(0.0, pcc.min() - 0.02), min(1.0, pcc.max() + 0.02))

ax1.set_title("Boxplots over 24 Timesteps (Left: RMSE/MAE, Right: PCC)")
fig.tight_layout()

out_path = "boxplot_rmse_mae_pcc_dual_axis.png"
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)

print("Saved:", out_path)

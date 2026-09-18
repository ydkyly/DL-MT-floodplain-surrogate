import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties
from io import StringIO

# ===== 字体：英文 Times New Roman，中文宋体 =====
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

def _get_chinese_font():
    candidates = ["SimSun", "宋体", "SimHei", "黑体", "Microsoft YaHei", "微软雅黑", "Arial Unicode MS"]
    available = set(f.name for f in fm.fontManager.ttflist)
    for name in candidates:
        if name in available:
            return FontProperties(family=name)
    return FontProperties()

CN = _get_chinese_font()

# ==============================
# ✅ 你只需要改下面这一段“原始表格文本”
# ==============================
RAW = r"""
sid	RMSE_depth	MAE_depth	PCC_depth	F1_depth	POD_depth	FAR_depth	RMSE_vmag	MAE_vmag	PCC_vmag
1	0.601 	0.264 	0.965 	0.885 	0.936 	0.161 	0.099 	0.033 	0.978 
2	0.458 	0.161 	0.964 	0.817 	0.994 	0.306 	0.063 	0.019 	0.979 
3	0.527 	0.185 	0.951 	0.810 	0.999 	0.318 	0.059 	0.018 	0.976 
4	0.761 	0.328 	0.911 	0.809 	0.932 	0.285 	0.103 	0.030 	0.952 
5	0.489 	0.195 	0.962 	0.890 	0.958 	0.169 	0.078 	0.024 	0.975 
6	0.590 	0.229 	0.947 	0.869 	0.943 	0.195 	0.082 	0.025 	0.977 
7	0.793 	0.323 	0.918 	0.794 	0.908 	0.294 	0.120 	0.040 	0.958 
8	0.709 	0.316 	0.940 	0.877 	0.928 	0.169 	0.134 	0.046 	0.948 
9	0.460 	0.168 	0.961 	0.782 	1.000 	0.358 	0.084 	0.026 	0.947 
10	0.810 	0.380 	0.933 	0.865 	0.896 	0.165 	0.123 	0.040 	0.967 
"""

def parse_table_text(raw: str) -> pd.DataFrame:
    # 1) 去掉多余空格，统一分隔符：连续空白 -> 单个制表符
    raw = raw.strip()
    lines = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        ln = re.sub(r"[ \t]+", "\t", ln)  # 空格/Tab统一
        lines.append(ln)
    clean = "\n".join(lines)
    df = pd.read_csv(StringIO(clean), sep="\t")
    df = df.apply(pd.to_numeric, errors="ignore")
    df = df.sort_values("sid").reset_index(drop=True)
    return df

df = parse_table_text(RAW)

# ===== 画图：A2箱线图（总体分布+散点sid），A3权衡散点（带大小图例）=====
OUT_DIR = "./SVD_Res_1"
os.makedirs(OUT_DIR, exist_ok=True)

def plot_metric_box_with_points(df, col, title_cn, ylab_cn, out_name, jitter=0.06):
    y = df[col].values
    sid = df["sid"].values
    fig = plt.figure(figsize=(6.0, 4.6))
    ax = plt.gca()
    ax.boxplot([y], positions=[1], widths=0.35, showfliers=True)

    rng = np.random.default_rng(42)
    x = 1 + rng.uniform(-jitter, jitter, size=len(y))
    ax.scatter(x, y, s=80, alpha=0.9)
    for xi, yi, si in zip(x, y, sid):
        ax.text(xi, yi, str(int(si)), ha="center", va="bottom", fontsize=10)

    ax.set_xticks([1])
    ax.set_xticklabels(["验证集10场洪水"], fontproperties=CN, fontsize=12)
    ax.set_xlabel("样本集合", fontproperties=CN, fontsize=12)
    ax.set_ylabel(ylab_cn, fontproperties=CN, fontsize=12)
    ax.set_title(title_cn, fontproperties=CN, fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, out_name), dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_A3_tradeoff(df, out_path, CN,
                     s_min=60, s_max=520,
                     legend_scale=0.45):
    """
    A3：F1_depth vs RMSE_depth，圆圈大小=RMSE_vmag
    - 点中心标注洪水编号 sid
    - 图例自动选位置（不指定loc）
    - 图例无边框线（frameon=False）
    """
    x = df["F1_depth"].values
    y = df["RMSE_depth"].values
    v = df["RMSE_vmag"].values
    sid = df["sid"].values.astype(int)

    # 点大小映射
    v_min = float(v.min())
    v_max = float(v.max())
    v_norm = (v - v_min) / (v_max - v_min + 1e-12)
    sizes = s_min + v_norm * (s_max - s_min)

    # 字号随点大小变化（与半径~sqrt(size)近似成正比）
    font_sizes = np.clip(np.sqrt(sizes) * 0.38, 8, 14)

    fig = plt.figure(figsize=(7.2, 5.2))
    ax = plt.gca()

    # 主散点（用默认颜色），接住对象以便图例颜色一致
    sc = ax.scatter(x, y, s=sizes, marker="o", alpha=0.75)

    # ✅ 洪水编号：放在圆心
    for xi, yi, si, fs in zip(x, y, sid, font_sizes):
        ax.text(xi, yi, str(si),
                ha="center", va="center",
                fontsize=float(fs),
                color="black", fontweight="bold")

    ax.set_xlabel("F1", fontproperties="Times New Roman", fontsize=12)
    ax.set_ylabel("Depth_RMSE(m)", fontproperties="Times New Roman", fontsize=12)
    ax.margins(x=0.08, y=0.12)

    # =========================
    # ✅ 图例：圆圈大小含义（自动位置 + 无边框）
    # =========================
    v_med = float(np.median(v))

    def _size(val):
        t = (val - v_min) / (v_max - v_min + 1e-12)
        return s_min + t * (s_max - s_min)

    # 颜色与主图一致（从sc中取）
    face = sc.get_facecolor()[0]
    edge = sc.get_edgecolor()[0] if len(sc.get_edgecolor()) > 0 else face

    handles = [
        ax.scatter([], [], s=_size(v_min) * legend_scale, marker="o",
                   facecolor=face, edgecolor=edge, alpha=0.75),
        ax.scatter([], [], s=_size(v_med) * legend_scale, marker="o",
                   facecolor=face, edgecolor=edge, alpha=0.75),
        ax.scatter([], [], s=_size(v_max) * legend_scale, marker="o",
                   facecolor=face, edgecolor=edge, alpha=0.75),
    ]
    labels = [f"最小：{v_min:.3f}", f"中位：{v_med:.3f}", f"最大：{v_max:.3f}"]

    leg = ax.legend(handles, labels,
                    title="RMSE_V",
                    loc=3,  # ✅ 自动选最合适位置
                    frameon=True,  # ✅ 打开外框
                    fancybox=False,  # ✅ 直角框（需要圆角就改 True）
                    framealpha=1.0,  # ✅ 不透明
                    borderpad=0.9,  # ✅ 外边框“内边距” -> 影响盒子长宽（越大越大）
                    labelspacing=1.15,  # ✅ 行间距 -> 影响高度
                    handletextpad=0.9,  # ✅ 圆圈与文字距离 -> 影响宽度
                    handlelength=1.6,  # ✅ handle预留长度 -> 影响宽度
                    borderaxespad=0.6  # ✅ 图例与坐标轴的距离（不影响盒子大小)
                    )
    # 图例：数字/英文 Times New Roman，中文回退宋体（或其他中文字体）
    LEG_FP = FontProperties(family=["Times New Roman", "SimSun"])

    legend_title_size = 12
    legend_text_size = 11

    # 标题
    leg.get_title().set_fontproperties(LEG_FP)
    leg.get_title().set_fontsize(legend_title_size)

    # 内容每一行
    for t in leg.get_texts():
        t.set_fontproperties(LEG_FP)
        t.set_fontsize(legend_text_size)

    # ✅ 调整外框颜色/线宽/背景
    frame = leg.get_frame()
    frame.set_edgecolor("black")  # 外框线颜色：0~1灰度；也可用 "black"
    frame.set_linewidth(0.5)  # 外框线宽
    frame.set_facecolor("white")  # 背景色（可选）

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# A2箱线图
# plot_metric_box_with_points(df, "F1_depth", "A2：验证集10场洪水的F1分布", "F1（无量纲）", "A2_box_F1.png")
# plot_metric_box_with_points(df, "RMSE_depth", "A2：验证集10场洪水的水深RMSE分布", "水深 RMSE（m）", "A2_box_RMSEdepth.png")
# plot_metric_box_with_points(df, "RMSE_vmag", "A2：验证集10场洪水的流速模长RMSE分布", "流速模长 RMSE（m/s）", "A2_box_RMSEvmag.png")

# A3散点图
plot_A3_tradeoff(
    df,
    out_path=os.path.join(OUT_DIR, "A3_tradeoff_autoLegend_noFrame.png"),
    CN=CN,
    s_min=60, s_max=520, legend_scale=0.45
)

print("图已保存到：", os.path.abspath(OUT_DIR))

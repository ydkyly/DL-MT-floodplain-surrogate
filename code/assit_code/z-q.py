#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高水位段 水位-流量 线性拟合（两份Excel：stage.xlsx & flow.xlsx）
- 训练：合并多个年份，筛选 Q>阈值 后做线性回归 Q = a*H + b（H 为自变量）
- 验证：对指定年份（默认2019）里 Q>阈值 的样本，用 H 代入公式预测 Q，并计算 RMSE
- 可视化：训练散点 + 拟合直线（H横轴，Q纵轴）
- 预测接口：predict_flow(H_value) / predict_flow_array(list/ndarray) —— 返回整数（四舍五入）

依赖：pandas numpy matplotlib openpyxl
安装：pip install pandas numpy matplotlib openpyxl
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 如需弹窗可注释掉本行
import matplotlib.pyplot as plt

# ===================== 配置区（在这里改） =====================
STAGE_XLSX = "stage.xlsx"     # 水位工作簿（sheet名=年份；12列=1~12月逐日）
FLOW_XLSX  = "flow.xlsx"      # 流量工作簿（结构同上）
STAGE_UNIT = "m"              # 水位单位：m / cm / mm
TRAIN_YEARS = [2019, 2022]    # 训练年份（可多个）
VALID_YEAR  = 2021            # 验证年份——改为 2021 就写 VALID_YEAR = 2021
Q_HIGH_TH   = 900.0           # 高流量阈值（仅用 Q>阈值 的样本拟合/验证）
OUT_PNG     = "high_stage_linear_fit.png"
OUT_TXT     = "high_stage_linear_fit.txt"
# ===========================================================

# 字体：中文宋体 + Times New Roman
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun', 'SimHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False

# --------- 工具：读取“12列月份、表名=年份”的日序列为 Series(date->value) ---------
MONTH_MAP = {
    "1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"10":10,"11":11,"12":12,
    "jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,"may":5,
    "jun":6,"june":6,"jul":7,"july":7,"aug":8,"august":8,"sep":9,"sept":9,"september":9,
    "oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12,
    "一月":1,"1月":1,"二月":2,"2月":2,"三月":3,"3月":3,"四月":4,"4月":4,"五月":5,"5月":5,
    "六月":6,"6月":6,"七月":7,"7月":7,"八月":8,"8月":8,"九月":9,"9月":9,"十月":10,"10月":10,
    "十一月":11,"11月":11,"十二月":12,"12月":12,
}

def _guess_month_cols(df: pd.DataFrame):
    """从列名识别12个月列；若列名不是月份，则直接取前12列。"""
    hits = []
    for c in df.columns:
        key = str(c).strip().lower()
        if key in MONTH_MAP and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0:
            hits.append((MONTH_MAP[key], c))
    if hits:
        uniq = {}
        for m, c in hits:
            if m not in uniq: uniq[m] = c
        pairs = sorted(uniq.items(), key=lambda x: x[0])
        return [col for _, col in pairs][:12]
    else:
        return list(df.columns[:12])

def _sheet_to_daily_series(df: pd.DataFrame, year: int) -> pd.Series:
    """将某年表的 12列(月)×日 矩阵展开为按日期的 Series"""
    month_cols = _guess_month_cols(df)
    df_m = df[month_cols].copy()

    # 若存在“日/日期”列，则优先用；否则按行号 1..N 推断天数
    day_col = None
    for k in df.columns:
        ks = str(k).strip().lower()
        if ks in ("day","日","日期","date"):
            day_col = k; break
    if day_col is not None:
        days = pd.to_numeric(df[day_col], errors="coerce").to_numpy()
    else:
        days = np.arange(1, len(df_m) + 1, dtype=int)

    out_list = []
    for j, col in enumerate(month_cols, start=1):
        vals = pd.to_numeric(df_m[col], errors="coerce").to_numpy()
        n = min(len(days), len(vals))
        d = pd.DataFrame({"day": days[:n], "val": vals[:n]}).dropna(subset=["val"])
        # 合法日期
        def mkdate(row):
            dd = int(row["day"])
            try:
                return pd.Timestamp(year=int(year), month=int(j), day=int(dd))
            except:
                return pd.NaT
        d["date"] = d.apply(mkdate, axis=1)
        d = d.dropna(subset=["date"]).set_index("date")["val"].astype(float)
        out_list.append(d)
    if not out_list:
        return pd.Series(dtype=float)
    return pd.concat(out_list).sort_index()

def read_yearbook_12x_months(xlsx_path: str, kind="H") -> pd.Series:
    """读取整个工作簿为一个按日期索引的 Series"""
    xls = pd.ExcelFile(xlsx_path)
    ser_list = []
    for sheet in xls.sheet_names:
        # 表名识别年份
        year = None
        try:
            year = int(sheet)
        except:
            for token in str(sheet).split():
                t = ''.join(ch for ch in token if ch.isdigit())
                if len(t) == 4 and t.startswith(('19','20')):
                    year = int(t); break
        if year is None:
            continue
        df = xls.parse(sheet)
        ser = _sheet_to_daily_series(df, year=year)
        if len(ser) > 0:
            ser_list.append(ser)
        else:
            print(f"[WARN] 跳过 sheet={sheet}（未识别出月份列或数据为空）")
    if not ser_list:
        raise RuntimeError(f"[ERROR] {xlsx_path} 未读出任何数据；请检查格式。")
    out = pd.concat(ser_list).sort_index()
    cnts = out.groupby(out.index.year).size().to_dict()
    print(f"[INFO] {('水位' if kind=='H' else '流量')} 各年有效日数：{cnts}")
    return out

# ---------------- 主流程 ----------------
def main():
    if not os.path.exists(STAGE_XLSX): raise FileNotFoundError(f"找不到水位文件：{STAGE_XLSX}")
    if not os.path.exists(FLOW_XLSX):  raise FileNotFoundError(f"找不到流量文件：{FLOW_XLSX}")

    print("[INFO] 读取Excel...")
    ser_H = read_yearbook_12x_months(STAGE_XLSX, kind="H").rename("H")
    ser_Q = read_yearbook_12x_months(FLOW_XLSX,  kind="Q").rename("Q")

    # 单位换算
    scale = {"m":1.0, "cm":0.01, "mm":0.001}[STAGE_UNIT.lower()]
    if abs(scale-1.0) > 1e-12:
        print(f"[INFO] 水位单位换算 ×{scale}")
    ser_H = ser_H * scale

    # 对齐 & 清洗
    df = pd.concat([ser_H, ser_Q], axis=1).sort_index().replace([np.inf,-np.inf], np.nan).dropna()
    df = df[df["Q"] > 0]
    if df.empty: raise RuntimeError("[ERROR] 合并后数据为空或全部Q<=0")

    # ========== 训练：合并指定年份 + Q>阈值 ==========
    df_tr = df[df.index.year.isin(TRAIN_YEARS)].copy()
    df_tr = df_tr[df_tr["Q"] > Q_HIGH_TH]
    if df_tr.shape[0] < 5:
        raise RuntimeError(f"[ERROR] 训练集中 Q>{Q_HIGH_TH} 的样本太少（{df_tr.shape[0]}），无法线性拟合。")
    print(f"[INFO] 训练年份: {TRAIN_YEARS}")
    print(f"[INFO] 训练样本数: {len(df_tr)}, 水位H范围: {df_tr['H'].min():.3f} ~ {df_tr['H'].max():.3f}, "
          f"流量Q范围: {df_tr['Q'].min():.1f} ~ {df_tr['Q'].max():.1f}")

    # 线性拟合：Q = a*H + b（H 为自变量）
    H_tr = df_tr["H"].to_numpy()
    Q_tr = df_tr["Q"].to_numpy()
    a, b = np.polyfit(H_tr, Q_tr, 1)
    formula = f"Q = {a:.6f} * H + {b:.6f}"
    print("========== 高水位段线性拟合 ==========")
    print(f"[公式] {formula}")

    # 训练R²
    Q_tr_hat = a * H_tr + b
    ss_res = float(np.sum((Q_tr - Q_tr_hat) ** 2))
    ss_tot = float(np.sum((Q_tr - np.mean(Q_tr)) ** 2))
    R2_tr  = 1.0 - ss_res/ss_tot if ss_tot > 1e-12 else 0.0
    print(f"[训练R²] {R2_tr:.4f}")

    # ========== 验证：仅取指定年份 + Q>阈值，然后用 H 输入预测 Q 并计算 RMSE ==========
    df_va_all = df[df.index.year == VALID_YEAR].copy()
    df_va = df_va_all[df_va_all["Q"] > Q_HIGH_TH]
    print(f"[INFO] 验证年份={VALID_YEAR}, 样本数: {len(df_va)}, 水位H范围: "
          f"{df_va['H'].min() if not df_va.empty else np.nan:.3f} ~ "
          f"{df_va['H'].max() if not df_va.empty else np.nan:.3f}, "
          f"流量Q范围: {df_va['Q'].min() if not df_va.empty else np.nan:.1f} ~ "
          f"{df_va['Q'].max() if not df_va.empty else np.nan:.1f}")

    if df_va.empty:
        RMSE_va = np.nan
        print(f"[WARN] 验证年 {VALID_YEAR} 中 Q>{Q_HIGH_TH} 的样本为空，跳过RMSE计算。")
    else:
        H_va = df_va["H"].to_numpy()
        Q_va = df_va["Q"].to_numpy()
        Q_va_hat = a * H_va + b
        RMSE_va = float(np.sqrt(np.mean((Q_va_hat - Q_va) ** 2)))
        print(f"[验证RMSE] 年份={VALID_YEAR}, RMSE={RMSE_va:.1f} m³/s")
        # —— 误差解释（相对于均值/最大值的占比）——
        q_mean = float(df_va["Q"].mean())
        q_max  = float(df_va["Q"].max())
        rel_to_mean = RMSE_va / q_mean * 100 if q_mean > 0 else np.nan
        rel_to_max  = RMSE_va / q_max  * 100 if q_max  > 0 else np.nan
        print(f"[解释] RMSE占平均流量: {rel_to_mean:.1f}%，占最大流量: {rel_to_max:.1f}%。")
        if rel_to_mean < 10:
            print("[结论] 对高流量预测而言，误差很小。")
        elif rel_to_mean < 20:
            print("[结论] 对高流量预测而言，误差可接受。")
        else:
            print("[结论] 对高流量预测而言，误差偏大。")

    # ========== 可视化（训练散点 + 拟合直线；H横轴，Q纵轴） ==========
    plt.figure(figsize=(7,6))
    plt.scatter(H_tr, Q_tr, s=18, alpha=0.75, edgecolors="k", linewidths=0.3,
                label=f"训练(Q>{Q_HIGH_TH:g})")
    # 直线画在训练H范围内
    H_min, H_max = float(np.min(H_tr)), float(np.max(H_tr))
    H_line = np.linspace(H_min, H_max, 200)
    Q_line = a * H_line + b
    plt.plot(H_line, Q_line, "r-", lw=2.0, label="线性拟合")
    plt.text(0.02, 0.98, f"{formula}\n$R^2$ = {R2_tr:.3f}",
             transform=plt.gca().transAxes, va="top",
             bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    plt.xlabel("水位 H")
    plt.ylabel("流量 Q")
    plt.title("高水位段 水位-流量 线性拟合（训练散点 & 拟合直线）")
    plt.grid(alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"[OK] 已输出图像：{OUT_PNG}")

    # —— 预测接口（返回整数） —— #
    def predict_flow(H_value: float) -> int:
        """输入单个水位值H，返回预测流量Q（整数，四舍五入）。"""
        return int(round(a * H_value + b))

    def predict_flow_array(H_values) -> np.ndarray:
        """输入list/ndarray的水位序列，返回预测流量数组（整数，四舍五入）。"""
        H_arr = np.asarray(H_values, dtype=float)
        return np.rint(a * H_arr + b).astype(int)

    # 示例：用训练集中位水位做一次预测
    demo_H = float(np.median(H_tr))
    demo_Q = predict_flow(demo_H)
    print(f"[示例预测] H={demo_H:.3f} -> Q_pred(int)={demo_Q:d}")

    # 保存公式与指标
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(f"公式: {formula}\n")
        f.write(f"训练年份: {TRAIN_YEARS}\n")
        f.write(f"训练样本数: {len(df_tr)}, 水位H范围: {H_min:.6f} ~ {H_max:.6f}, "
                f"流量Q范围: {df_tr['Q'].min():.1f} ~ {df_tr['Q'].max():.1f}\n")
        f.write(f"训练R2: {R2_tr:.6f}\n")
        f.write(f"验证年份: {VALID_YEAR}, 样本数: {len(df_va)}, 水位H范围: "
                f"{(df_va['H'].min() if not df_va.empty else np.nan):.6f} ~ "
                f"{(df_va['H'].max() if not df_va.empty else np.nan):.6f}, "
                f"流量Q范围: "
                f"{(df_va['Q'].min() if not df_va.empty else np.nan):.1f} ~ "
                f"{(df_va['Q'].max() if not df_va.empty else np.nan):.1f}\n")
        if not np.isnan(RMSE_va):
            f.write(f"验证RMSE({VALID_YEAR}): {RMSE_va:.6f} m^3/s\n")
            f.write(f"RMSE占平均流量: {rel_to_mean:.2f}%，占最大流量: {rel_to_max:.2f}%\n")
    print(f"[OK] 已保存：{OUT_TXT}")

if __name__ == "__main__":
    main()

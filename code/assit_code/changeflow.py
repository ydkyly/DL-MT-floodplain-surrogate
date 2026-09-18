#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从一个Excel读取（默认第一个工作表），每列是一条流量序列。
为每列生成“row + current + 全序列”的表，并按从 start 开始的号码命名：
30.xlsx, 31.xlsx, ...（不做零填充）。严格读取每列前 length 个数（默认 24）。
Excel 输出不含表头。
"""

import argparse
import os
from typing import List, Literal, Optional

import pandas as pd

FillType = Optional[Literal["zero", "ffill"]]

def load_series_exact(df: pd.DataFrame, col, length: int, fill: FillType) -> List[int]:
    """
    严格读取该列前 length 个单元格（不丢弃空值）。
    - 若存在非数字/空值：默认报错；若 fill 指定：
        * "zero"  — 用 0 填充
        * "ffill" — 前向填充（第一个若为空则为 0）
    - 最终四舍五入为 int（若需保留小数，替换为 float）
    """
    s = df[col].iloc[:length]
    # 转数值（非数值转为 NaN）
    s_num = pd.to_numeric(s, errors="coerce")

    if s_num.isna().any():
        if fill is None:
            # 定位问题点
            bad_idx = [i+1 for i, v in enumerate(s_num.tolist()) if pd.isna(v)]
            raise SystemExit(
                f"列[{col}] 前 {length} 个单元格中存在非数字/空值，位置（1基）：{bad_idx}。\n"
                "可修正Excel，或加参数 --fill zero / --fill ffill 自动填补。"
            )
        else:
            if fill == "ffill":
                s_num = s_num.ffill().fillna(0)
            elif fill == "zero":
                s_num = s_num.fillna(0)

    if len(s_num) != length:
        raise SystemExit(f"列[{col}] 实际可读长度为 {len(s_num)}，与要求的 {length} 不一致。请检查数据或调整 --length。")

    return [int(round(float(v))) for v in s_num.tolist()]

def build_table(flows: List[int]) -> pd.DataFrame:
    n = len(flows)
    all_str = [str(x) for x in flows]
    rows = []
    for i, val in enumerate(flows, start=1):
        rows.append([str(i), str(val)] + all_str)
    # 列名仅用于内部；写出时 header=False
    columns = ["row", "current"] + [f"c{i}" for i in range(1, n + 1)]
    return pd.DataFrame(rows, columns=columns)

def main():
    parser = argparse.ArgumentParser(description="按列批量生成流量表（严格取每列前 length 个值），Excel不含表头")
    parser.add_argument("--input", "-i", default=r'D:\Work\qyb\ResNet-18\data\Data.xlsx', help="输入Excel路径")
    parser.add_argument("--sheet", "-s", default=None, help="工作表名（缺省读取第一个工作表）")
    parser.add_argument("--outdir", "-o", default="flow_outputs", help="输出目录（默认 flow_outputs）")
    parser.add_argument("--prefix", default="", help="文件名前缀（可选）")
    parser.add_argument("--suffix", default="", help="文件名后缀（可选，不含扩展名）")
    parser.add_argument("--start", type=int, default=30, help="起始编号（第一列对应的文件号，默认30）")
    parser.add_argument("--length", type=int, default=24, help="每列应读取的点数（默认24）")
    parser.add_argument("--fill", choices=["zero", "ffill"], default=None,
                        help="缺失/非数字的填充方式：zero=用0填；ffill=前向填充；默认不填充而报错")
    args = parser.parse_args()

    # 读取Excel（header=0 表示首行作为列名）
    df = pd.read_excel(args.input, sheet_name=(args.sheet if args.sheet else 0), header=0)

    os.makedirs(args.outdir, exist_ok=True)

    total = 0
    for idx, col in enumerate(df.columns, start=1):
        try:
            flows = load_series_exact(df, col, length=args.length, fill=args.fill)
        except SystemExit as e:
            print(f"[错误] 第{idx}列（列名：{col}）→ {e}")
            continue

        out_df = build_table(flows)

        file_number = args.start + (idx - 1)   # 30, 31, ...
        fname = f"{args.prefix}{file_number}{args.suffix}.xlsx"
        fpath = os.path.join(args.outdir, fname)

        # 不导出表头
        out_df.to_excel(fpath, index=False, header=False)
        total += 1
        print(f"[完成] 第{idx}列 → {fpath}（{len(flows)} 个点）")

    print(f"完成：生成 {total} 个文件；目录：{os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()

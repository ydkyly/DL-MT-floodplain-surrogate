# -*- coding: utf-8 -*-
"""
plot_style.py
统一设置 Matplotlib 英文字体为 'Times New Roman'，并做合理回退。
"""
from __future__ import annotations
from matplotlib import rcParams, font_manager

def enforce_times_new_roman():
    target = "Times New Roman"
    # 检查系统是否有这个字体；没有则友好回退
    has_tnr = any(f.name == target for f in font_manager.fontManager.ttflist)
    if not has_tnr:
        print(f"[PlotStyle][WARN] '{target}' not found, fallback to STIXGeneral/DejaVu Serif.")
        rcParams["font.family"] = ["STIXGeneral", "DejaVu Serif", "serif"]
        rcParams["mathtext.fontset"] = "stix"
    else:
        rcParams["font.family"] = [target]
        rcParams["mathtext.fontset"] = "stix"   # 数学字体风格与 Times 更协调
    rcParams["axes.unicode_minus"] = False      # 负号正常显示

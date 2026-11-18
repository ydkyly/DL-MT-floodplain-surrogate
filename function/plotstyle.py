# -*- coding: utf-8 -*-
"""Apply a consistent Times New Roman Matplotlib style with graceful font fallback."""
from __future__ import annotations
from matplotlib import rcParams, font_manager

def enforce_times_new_roman():
    target = "Times New Roman"
    # Check font availability and fall back gracefully.
    has_tnr = any(f.name == target for f in font_manager.fontManager.ttflist)
    if not has_tnr:
        print(f"[PlotStyle][WARN] '{target}' not found, fallback to STIXGeneral/DejaVu Serif.")
        rcParams["font.family"] = ["STIXGeneral", "DejaVu Serif", "serif"]
        rcParams["mathtext.fontset"] = "stix"
    else:
        rcParams["font.family"] = [target]
        rcParams["mathtext.fontset"] = "stix"   # Choose mathematical fonts compatible with Times.
    rcParams["axes.unicode_minus"] = False      # Render minus signs correctly.

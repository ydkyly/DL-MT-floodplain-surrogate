#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEM 垂直基准校正：WGS84 椭球高 h  →  EGM2008 正常高 H = h - N

用法示例：
python dem_egm2008_vshift.py \
  --dem "E109D5_N20D0_CopyRaster.tif" \
  --geoid "us_nga_egm08_25.tif" \
  --out "DEM_H_EGM2008.tif"

可选：
  --resampling bilinear|cubic|nearest
  --nodata -9999
  --dtype float32
  --write-report  校正后打印采样对比（需加 --lon --lat）
  --lon 109.75 --lat 20.25
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.enums import Compression
from rasterio.transform import Affine


def parse_args():
    p = argparse.ArgumentParser(description="Apply EGM2008 geoid to DEM (H = h + N).")
    p.add_argument("--dem", required=True, help="输入 DEM（椭球高，GeoTIFF）")
    p.add_argument("--geoid", required=True, help="EGM2008 geoid 栅格（GeoTIFF/GTX）")
    p.add_argument("--out", required=True, help="输出 正常高 DEM（GeoTIFF）")
    p.add_argument("--resampling", default="bilinear",
                   choices=["nearest", "bilinear", "cubic"], help="重采样方法（对 geoid → DEM 网格）")
    p.add_argument("--nodata", type=float, default=-9999.0, help="输出 NoData 值")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"], help="输出数据类型")
    p.add_argument("--lon", type=float, help="可选：验证点 经度")
    p.add_argument("--lat", type=float, help="可选：验证点 纬度")
    p.add_argument("--write-report", action="store_true", help="打印验证点 h、N、H 值")
    return p.parse_args()


def resampling_enum(name: str) -> Resampling:
    return {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic
    }[name]


def open_raster(path: str):
    try:
        ds = rasterio.open(path)
    except Exception as e:
        sys.exit(f"[ERROR] 打开栅格失败：{path}\n{e}")
    return ds


def sample_point(ds: rasterio.io.DatasetReader, lon: float, lat: float):
    # 假设 DEM 与 Geoid 都在地理坐标（WGS84，经纬度）。若不是，rasterio.index 同样适用。
    row, col = ds.index(lon, lat)
    arr = ds.read(1, masked=True)
    if 0 <= row < ds.height and 0 <= col < ds.width:
        v = arr[row, col]
        return None if np.ma.is_masked(v) else float(v)
    return None


def main():
    args = parse_args()

    dem_ds = open_raster(args.dem)
    geoid_ds = open_raster(args.geoid)

    print("[INFO] DEM CRS:", dem_ds.crs)
    print("[INFO] Geoid CRS:", geoid_ds.crs)

    # 读取 DEM（h）
    h = dem_ds.read(1, masked=True).astype(np.float32)
    dem_meta = dem_ds.meta.copy()

    # 准备一个与 DEM 同尺寸/同地理参考的 geoid 栅格 N_resamp
    N_resamp = np.zeros((dem_ds.height, dem_ds.width), dtype=np.float32)

    # 重采样 geoid → DEM 网格
    reproject(
        source=rasterio.band(geoid_ds, 1),
        destination=N_resamp,
        src_transform=geoid_ds.transform,
        src_crs=geoid_ds.crs,
        dst_transform=dem_ds.transform,
        dst_crs=dem_ds.crs,
        dst_width=dem_ds.width,
        dst_height=dem_ds.height,
        resampling=resampling_enum(args.resampling),
    )

    # 计算 H = h - N
    H = h.filled(np.nan) + N_resamp  # 保持 float32
    # 将 DEM 掩膜或 geoid 无值的像元统一设为 NoData
    mask = np.ma.getmaskarray(h) | ~np.isfinite(H)
    H_out = np.where(mask, args.nodata, H).astype(args.dtype)

    # 写出
    out_meta = dem_meta.copy()
    out_meta.update(
        dtype=args.dtype,
        nodata=args.nodata,
        compress="lzw",
        tiled=True,
        BIGTIFF="IF_SAFER",
        driver="GTiff",
        count=1,
    )
    with rasterio.open(args.out, "w", **out_meta) as dst:
        dst.write(H_out, 1)
        # 写入少量元数据，注明垂直基准
        dst.update_tags(
            1,
            LONG_NAME="Orthometric height (EGM2008)",
            DESCRIPTION="H = h + N(EGM2008)"
        )
        dst.update_tags(
            VERTICAL_DATUM="EGM2008",
            NOTE="Converted from WGS84 ellipsoidal heights using EGM2008 geoid (undulation N).",
        )

    print(f"[OK] 输出完成：{args.out}")

    # 可选：在指定点打印 h, N, H 的数值核查
    if args.write_report and args.lon is not None and args.lat is not None:
        h_val = sample_point(dem_ds, args.lon, args.lat)
        N_val = sample_point(geoid_ds, args.lon, args.lat)
        # 若 geoid 与 DEM 网格不同，直接采样 geoid_ds 是近邻像元值，仅用于快速核查；
        # 更严格可改为采样 N_resamp（即与 DEM 对齐后的 geoid 值）。
        # 这里我们也打印对齐后的 N_resamp：
        row, col = dem_ds.index(args.lon, args.lat)
        N_val_resamp = float(N_resamp[row, col]) if (0 <= row < dem_ds.height and 0 <= col < dem_ds.width) else None
        H_val = sample_point(open_raster(args.out), args.lon, args.lat)
        report = {
            "lon": args.lon, "lat": args.lat,
            "h_DEM": h_val,
            "N_geoid_native": N_val,
            "N_geoid_resamp": N_val_resamp,
            "H_out": H_val,
            "relation": "H ≈ h - N_resamp"
        }
        print("[REPORT]", json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
import argparse, os, sys
from function.train import train_and_validate
from function.infer import batch_infer
from function.basis import build_basis as build_basis_ipca_cli
from function.basis_builder import build_basis_from_args as build_basis_svd_cli
from function.basis_checker import check_basis_from_args
from function.postviz import viz_pred as post_viz_cli
from function.tools_dump_xy import dump_xy as dump_xy_cli

def str2bool(v):
    """兼容：--flag / --flag true / --flag false"""
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool: {v}")

def _add_common_data_args(p: argparse.ArgumentParser):
    p.add_argument('--inflow', default='D:/Work/qyb/ResNet-18/data/train/inflow', help="训练集输入目录 (.xlsx)")
    p.add_argument('--HDF5', default='D:/Work/qyb/ResNet-18/data/train/hdf', help="训练集标签目录 (.hdf/子目录)")
    p.add_argument("--timesteps", type=int, default=24, help="时间步数 T")
    p.add_argument("--flood_threshold", type=float, default=0.2, help="淹没阈值（F1/POD/FAR）")
    p.add_argument('--val_inflow', default='D:/Work/qyb/ResNet-18/data/val/inflow', help="验证集输入目录 (.xlsx)")
    p.add_argument('--val_hdf5', default='D:/Work/qyb/ResNet-18/data/val/hdf', help="验证集标签目录")

def _add_model_args(p: argparse.ArgumentParser):
    # 兼容旧参数 block，同时新增 backbone
    p.add_argument("--backbone", choices=["resnet_basic","resnet_bottleneck","tcn"],
                   default="tcn", help="主干网络")
    p.add_argument("--block", choices=["basic","bottleneck"], default="basic",
                   help="仅当 backbone 为 resnet_* 时有效")
    p.add_argument("--tasks", type=str, default='depth, flow', help="训练/推理任务列表，例如depth只训练水深，depth，flow水深与流速")
    p.add_argument("--rank_k", type=int, default=256, help="低秩基底秩 K,水深，流速，默认相同")
    # p.add_argument("--basis", default=r'D:\Work\qyb\ResNet-18\V2.3\Basis0\basis.npy', help="basis.npy 路径")
    p.add_argument("--depth_basis", default=r'D:\Work\qyb\ResNet-18\V2.4\Basis_WD\basis.npy', help="水深基底")
    p.add_argument("--vx_basis", default=r'D:\Work\qyb\ResNet-18\V2.4\Basis_vx\basis.npy', help="流速x基底")
    p.add_argument("--vy_basis", default=r'D:\Work\qyb\ResNet-18\V2.4\Basis_vy\basis.npy', help="流速y基底")

    # TCN 细节
    p.add_argument("--tcn_kernel", type=int, default=3)
    p.add_argument("--tcn_dropout", type=float, default=0.2)
    p.add_argument("--tcn_channels", type=int, nargs="+", default=[64,64,64],
                   help="TCN 每层通道数，如 64 64 64")

def _add_train_args(p: argparse.ArgumentParser):
    _add_common_data_args(p)
    _add_model_args(p)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--shuffle_train", action="store_true")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--plateau_patience", type=int, default=20)
    p.add_argument("--plateau_factor", type=float, default=0.5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--val_interval", type=int, default=1)
    p.add_argument("--save_dir", default="Result_20try")
    p.add_argument("--preload", action="store_true")
    p.add_argument("--accum_steps", type=int, default=1)
    # 采样/列块
    p.add_argument("--col_mode", choices=["full","chunk_in_batch","random","cycle_step"], default="full")
    p.add_argument("--sample_h", type=int, default=200000, help="每步采样列数（H 子块）")
    # loss / 正则
    p.add_argument("--alpha_depth_loss", type=float, default=0)
    p.add_argument("--alpha_target", type=float, default=0)
    p.add_argument("--depth_loss_gamma", type=float, default=0)
    p.add_argument("--gamma_target", type=float, default=0)
    p.add_argument("--warmup_ratio", type=float, default=0)
    p.add_argument("--smooth_c_lambda", type=float, default=0)
    p.add_argument("--huber_delta", type=float, default=0.1)
    p.add_argument("--w_clip_q", type=float, default=0.999)
    p.add_argument("--beta_dry", type=float, default=0.5)  # 干区惩罚权重
    # 多任务损失权重（仅在 tasks 包含 vx/vy 时生效）
    p.add_argument("--w_depth", type=float, default=1.0, help="水深任务在总损失中的权重")
    p.add_argument("--w_vx",    type=float, default=1.0, help="流速 x 任务在总损失中的权重")
    p.add_argument("--w_vy",    type=float, default=1.0, help="流速 y 任务在总损失中的权重")
    # 早停（可选）
    p.add_argument("--early_stop_patience", type=int, default=30)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-3)
    p.add_argument("--train_stop_threshold", type=float, default=None)
    # 训练结束后后处理
    p.add_argument("--post_eval", default=False)
    # p.add_argument("--save_depth_maps", action="store_true")
    # p.add_argument("--save_flow_components", action="store_true")
    # p.add_argument("--save_velocity_maps", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument('--xy_npy', default='D:/Work/qyb/ResNet-18/data/xy.npy', help="坐标")
    p.add_argument("--to_lonlat", default=True, help="xy_npy 是否需要转经纬度")
    p.add_argument("--bg_image", type=str, default=r'D:\Work\qyb\Hec_Ras\YD\YD.tif', help="(可选) 遥感影像底图 (tif/png)，仅在 --save_depth_maps 时叠加")
    p.add_argument("--bg_alpha", type=float, default=0.5,help="遥感影像透明度(0-1)，用于叠加底图")

def _add_infer_args(p: argparse.ArgumentParser):
    _add_model_args(p)
    p.add_argument('--inflow', default='D:/Work/qyb/ResNet-18/data/test/inflow', help="测试集输入目录 (.xlsx)")
    p.add_argument("--timesteps", type=int, default=24, help="时间步数 T")
    p.add_argument("--flood_threshold", type=float, default=0.1, help="淹没阈值（F1/POD/FAR）")
    p.add_argument("--checkpoint", default=r'D:\Work\qyb\ResNet-18\V2.4\Result\20251212_210230_0.5_k256_depth, flow_resnet_basic\best_20251212_210230_0.5_k256_depth, flow_resnet_basic.pth', help="best_*.pth 路径")
    p.add_argument("--norm_stats", default=r'D:\Work\qyb\ResNet-18\V2.4\Result\20251212_210230_0.5_k256_depth, flow_resnet_basic\best_20251212_210230_0.5_k256_depth, flow_resnet_basic.npz', help="可选：norm_stats_*.npz（否则自动匹配）")
    p.add_argument("--output_dir", default="Infer_PCA_0727", help="预测输出目录")
    p.add_argument("--val_block", type=int, default=200000, help="推理分块重建列宽")
    p.add_argument("--mean_depth", type=str, default=r'D:\Work\qyb\ResNet-18\V2.4\Basis_WD\mean.npy', help="depth 的 mean.npy 路径（ckpt 无 mu 时用于重建：Y=C@B+mu）")
    p.add_argument("--mean_vx", type=str, default=r'D:\Work\qyb\ResNet-18\V2.4\Basis_vx\mean.npy', help="vx 的 mean.npy 路径（ckpt 无 mu 时用于重建）")
    p.add_argument("--mean_vy", type=str, default=r'D:\Work\qyb\ResNet-18\V2.4\Basis_vy\mean.npy', help="vy 的 mean.npy 路径（ckpt 无 mu 时用于重建）")

def _add_basis_ipca_args(p: argparse.ArgumentParser):
    p.add_argument("--chunk_rows", type=int, default=8192, help="增量PCA行块大小(样本*时间)")
    p.add_argument('--label_dir', default='D:/Work/qyb/ResNet-18/data/train/hdf80', help="标签目录(.hdf/.npy/子目录)")
    p.add_argument('--timesteps', type=int, default=24, help="使用前 T 个时间步参与PCA")
    p.add_argument('--n_components', type=int, default=256, help="PCA 主成分数量 K")
    p.add_argument('--append_mean', action='store_true', help="把列均值也作为一条基(输出K+1行)")
    p.add_argument('--out_dir',     type=str, default="Basis_80vy", help="输出目录")
    p.add_argument('--data_path', type=str, default="Results/Unsteady/Output/Output Blocks/Base Output/"
                  "Unsteady Time Series/2D Flow Areas/Perimeter 1/Cell Velocity - Velocity Y", help="(可选) HDF 内数据集路径。")

def _add_basis_svd_args(p: argparse.ArgumentParser):
    p.add_argument("--hdf_dir", default='D:/Work/qyb/ResNet-18/data/train/hdf', help="标签目录(.hdf/.npy/子目录)")
    p.add_argument("--pattern", default="*.hdf")
    p.add_argument("--rank_k", type=int, default=256)
    p.add_argument("--save", default="Basis_SVD_vy", help="保存路径（文件或目录）")
    p.add_argument("--data_path", default="Results/Unsteady/Output/Output Blocks/Base Output/"
              "Unsteady Time Series/2D Flow Areas/Perimeter 1/Cell Velocity - Velocity Y")
    p.add_argument("--svd_iter", type=int, default=5)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--wet_threshold", type=float, default=None)
    p.add_argument("--center", default=True, help="是否对列做均值中心化并保存 mean.npy")
    p.add_argument("--append_mean", action="store_true", help="将 μ 作为最后一行拼入 B（兼容旧流程，不推荐）")

def _add_check_basis_args(p: argparse.ArgumentParser):
    p.add_argument("--basis", default=r'D:\Work\qyb\ResNet-18\V2.4\Basis_vy\basis.npy')
    p.add_argument("--hdf_dir", default='D:/Work/qyb/ResNet-18/data/train/hdf')
    p.add_argument("--pattern", default="*.hdf")
    p.add_argument("--data_path", default="Results/Unsteady/Output/Output Blocks/Base Output/"
              "Unsteady Time Series/2D Flow Areas/Perimeter 1/Cell Velocity - Velocity Y")
    p.add_argument("--sample_n", type=int, default=None)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--save_dir", default=None)
    p.add_argument("--save_fig", action="store_true")


def _add_viz_args(p: argparse.ArgumentParser):
    # 基本输入输出
    p.add_argument("--pred_dir", type=str,
                   default=r"D:\Work\qyb\ResNet-18\data_dw\Infer_SVD_resnet",
                   help="包含预测/真值 npy 的目录")

    p.add_argument("--gt_dir", type=str,
                   default=r"D:\Work\qyb\ResNet-18\data_dw\test\hdf",
                   help="(可选) 真值 HDF 目录；当同目录无 true.npy 时才回退读取")

    p.add_argument("--output_dir", type=str, default=r"D:\Work\qyb\ResNet-18\data_dw\Infer_SVD_resnet\infer_SVD_resnet", help="输出目录")

    # 识别/绘图开关
    p.add_argument('--separate_panels', default=True, help='分开输出 Pred/Truth/Diff 三张图（每时刻），不输出三联图')
    p.add_argument("--threshold", type=float, default=0.2,
                   help="depth 淹没阈值（m），用于 mask 与 F1/POD/FAR")
    p.add_argument("--combine", type=str2bool, nargs="?", const=True, default=False,
                   help="是否输出 Pred/GT/Diff 合并图")
    p.add_argument("--pred_only", type=str2bool, nargs="?", const=True, default=False,
                   help="仅输出 Pred（不画 GT/Diff/confusion）")
    p.add_argument("--save_depth_maps", type=str2bool, nargs="?", const=True, default=True,
                   help="是否输出水深图")
    p.add_argument("--save_vel_maps", type=str2bool, nargs="?", const=True, default=True,
                   help="是否输出流速模长|v|图（同目录存在 Vx/Vy 或 Vmag 时）")
    p.add_argument("--save_vel_component_maps", type=str2bool, nargs="?", const=True, default=True,
                   help="是否输出 Vx/Vy 分量 Pred/GT/Diff（各自单独画布）")
    p.add_argument("--save_metrics_per_timestep", type=str2bool, nargs="?", const=True, default=True,
                   help="输出逐时刻 depth(+vmag) 指标 CSV")

    # 显示/阈值（你当前 postviz 真实会用到）
    p.add_argument("--min_show_depth", type=float, default=0.1,
                   help="绘图显示最小水深，小于该值置为 NaN（减少浅水噪声）")
    p.add_argument("--min_show_speed", type=float, default=0.001,
                   help="绘图显示最小流速，小于该值置为 NaN（减少噪声/幽灵外溢）")
    p.add_argument("--outside_mask_alpha", type=float, default=0.0,
                   help="研究区外遮罩透明度(0表示不遮罩)")
    p.add_argument("--vmax_percentile", type=float, default=99,
                   help="色阶上限采用分位数（depth默认98.5，速度一般99可在命令行改）")

    # 栅格化与显示（grid 模式会用；poly 模式可忽略但保留兼容）
    p.add_argument("--nx", type=int, default=1200, help="(grid) 栅格化 x 方向网格数")
    p.add_argument("--smooth_sigma", type=float, default=0.8, help="(grid) NaN-aware 高斯平滑 sigma（0 表示不平滑）")
    p.add_argument("--data_alpha", type=float, default=0.85, help="数据层透明度")
    p.add_argument("--interpolation", type=str, default="bilinear",
                   choices=["nearest", "bilinear", "bicubic"], help="imshow 插值")
    p.add_argument("--boundary_lw", type=float, default=1.5, help="研究区边界线宽")
    p.add_argument("--vmax_mode", type=str, default="global",
                   choices=["global", "frame"], help="色阶：全局一致 or 每帧自适应")

    # ✅几何/研究区外边界/淹没confusion（poly 渲染会用）
    p.add_argument("--geom_hdf", type=str, default=r"D:\Work\qyb\ResNet-18\data_dw\test\hdf\01.hdf",
                   help="(推荐) 指定含 Geometry 的 HDF；为空则每个样本用 gt_dir/{sid}.hdf 读取几何")
    p.add_argument("--area_name", type=str, default="Perimeter 1",
                   help="2D Flow Area 名称（HEC-RAS 常见为 'Perimeter 1'）")

    # 淹没confusion 渲染模式：auto(优先poly)，poly(强制单元格填充)，raster(回退栅格投票)
    p.add_argument("--confusion_render", type=str, default="auto",
                   choices=["auto", "poly", "raster"],
                   help="淹没混淆图渲染模式")
    # 输出格式/画布
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--formats", type=str, default="png", help="输出格式，如 'png' 或 'png,pdf,svg'")

    # 坐标轴
    p.add_argument("--show_axes", type=str2bool, nargs="?", const=True, default=False,
                   help="是否显示主坐标轴刻度")
    p.add_argument("--show_secondary_lonlat", type=str2bool, nargs="?", const=True, default=True,
                   help="是否添加经纬度副坐标轴")
    p.add_argument("--to_lonlat", type=str2bool, nargs="?", const=True, default=False,
                   help="(兼容旧参数) 等价于 --show_secondary_lonlat")
    p.add_argument("--assume_xy_crs", type=str, default="EPSG:3857", help="xy 坐标 CRS，用于经纬度副轴")

    # 背景影像
    p.add_argument("--bg_image", type=str, default=r"D:\Work\qyb\ResNet-18\data_dw\Remote_mengwa.tif",
                   help="GeoTIFF/PNG/JPG 背景影像路径")
    p.add_argument("--bg_alpha", type=float, default=0.5)
    p.add_argument("--bg_stretch", type=str2bool, nargs="?", const=True, default=True,
                   help="是否对背景做百分位拉伸")
    p.add_argument("--bg_p_low", type=float, default=2.0)
    p.add_argument("--bg_p_high", type=float, default=98.0)


def _add_dump_xy_args(p: argparse.ArgumentParser):
    p.add_argument('--hdf_dir', default='D:/Work/qyb/ResNet-18/data/test/hdf', help="包含 <sid>.hdf/.h5/.hdf5 的目录")
    p.add_argument('--out', default='D:/Work/qyb/ResNet-18/data/xy_try.npy', help="输出 xy.npy 路径（形状 [H,2]）")
    p.add_argument('--tolerance', type=float, default=1e-6, help="跨文件坐标一致性容差（欧氏距离）")
    p.add_argument('--max_check', type=int, default=20, help="抽查验证的最大文件数（0=检查全部）")

def main():
    ap = argparse.ArgumentParser("Flood Runner")
    sp = ap.add_subparsers(dest="cmd", required=True)

    # 训练 / 推理
    p_train = sp.add_parser("train", help="训练")
    _add_train_args(p_train)

    p_infer = sp.add_parser("infer", help="批量推理")
    _add_infer_args(p_infer)

    # 基底构建（IPCA / SVD）
    p_bi = sp.add_parser("build-basis-ipca", help="增量 PCA 构建 basis（旧流程，兼容）")
    _add_basis_ipca_args(p_bi)

    p_bs = sp.add_parser("build-basis-svd", help="TruncatedSVD 构建 basis（支持 center / wet_mask）")
    _add_basis_svd_args(p_bs)

    # basis 自检
    p_chk = sp.add_parser("check-basis", help="对 basis.npy 做重建误差自检")
    _add_check_basis_args(p_chk)

    # 统一可视化
    p_viz = sp.add_parser("viz", help="对 *_pred.npy 进行绘图与指标汇总")
    _add_viz_args(p_viz)

    # 坐标导出
    p_xy = sp.add_parser("dump-xy", help="从若干 HDF 抽查并导出 XY 坐标到 .npy")
    _add_dump_xy_args(p_xy)

    args = ap.parse_args()

    if args.cmd == "train":
        return train_and_validate(args)
    elif args.cmd == "infer":
        return batch_infer(args)
    elif args.cmd == "build-basis-ipca":
        return build_basis_ipca_cli(args)
    elif args.cmd == "build-basis-svd":
        return build_basis_svd_cli(args)
    elif args.cmd == "check-basis":
        return check_basis_from_args(args)
    elif args.cmd == "viz":
        return post_viz_cli(args)
    elif args.cmd == "dump-xy":
        return dump_xy_cli(args)
    else:
        raise SystemExit(f"未知子命令：{args.cmd}")

if __name__ == "__main__":
    main()

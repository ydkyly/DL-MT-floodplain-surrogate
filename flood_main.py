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
    """Accept --flag, --flag true, and --flag false."""
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
    p.add_argument('--inflow', default='D:/Work/qyb/ResNet-18/data/train/inflow', help="Directory containing input hydrographs and derived features (.xlsx).")
    p.add_argument('--HDF5', default='D:/Work/qyb/ResNet-18/data/train/hdf', help="Directory containing training HEC-RAS reference fields.")
    p.add_argument("--timesteps", type=int, default=24, help="Timesteps.")
    p.add_argument("--flood_threshold", type=float, default=0.1, help="Wet/dry depth threshold in metres (default: 0.1).")
    p.add_argument('--val_inflow', default='D:/Work/qyb/ResNet-18/data/val/inflow', help="Validation input directory (.xlsx).")
    p.add_argument('--val_hdf5', default='D:/Work/qyb/ResNet-18/data/val/hdf', help="Validation HEC-RAS reference directory.")

def _add_model_args(p: argparse.ArgumentParser):
    # Preserve the legacy block argument while adding backbone selection.
    p.add_argument("--backbone", choices=["resnet_basic","resnet_bottleneck","tcn"],
                   default="tcn", help="Backbone.")
    p.add_argument("--block", choices=["basic","bottleneck"], default="basic",
                   help="Block.")
    p.add_argument("--tasks", type=str, default='depth, flow', help="Tasks: depth for depth only; depth,flow for depth and vx/vy.")
    p.add_argument("--rank_k", type=int, default=256, help="Low-rank basis size K, shared across variables unless overridden.")
    # Legacy basis.npy argument; use the variable-specific basis paths below.
    p.add_argument("--depth_basis", default=r'D:\Work\qyb\ResNet-18\V2.4\Basis_WD\basis.npy', help="Depth basis.npy path.")
    p.add_argument("--vx_basis", default=r'D:\Work\qyb\ResNet-18\V2.4\Basis_vx\basis.npy', help="Velocity-x basis.npy path.")
    p.add_argument("--vy_basis", default=r'D:\Work\qyb\ResNet-18\V2.4\Basis_vy\basis.npy', help="Velocity-y basis.npy path.")

    # TCN configuration.
    p.add_argument("--tcn_kernel", type=int, default=3)
    p.add_argument("--tcn_dropout", type=float, default=0.2)
    p.add_argument("--tcn_channels", type=int, nargs="+", default=[64,64,64],
                   help="Channels per temporal block; default 64 64 64 gives dilations 1, 2, 4.")

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
    # Sampling and column blocks.
    p.add_argument("--col_mode", choices=["full","chunk_in_batch","random","cycle_step"], default="full")
    p.add_argument("--sample_h", type=int, default=200000, help="Number of spatial columns sampled per step.")
    # Loss and regularisation.
    p.add_argument("--alpha_depth_loss", type=float, default=0.5)
    p.add_argument("--alpha_target", type=float, default=0.5)
    p.add_argument("--depth_loss_gamma", type=float, default=1.0)
    p.add_argument("--gamma_target", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0)
    p.add_argument("--smooth_c_lambda", type=float, default=0)
    p.add_argument("--huber_delta", type=float, default=0.1)
    p.add_argument("--w_clip_q", type=float, default=0.999)
    p.add_argument("--beta_dry", type=float, default=0.5)  # Dry-region penalty weight.
    # Multi-task loss weights, used when tasks include vx/vy.
    p.add_argument("--w_depth", type=float, default=1.0, help="W depth.")
    p.add_argument("--w_vx",    type=float, default=1.0, help="W vx.")
    p.add_argument("--w_vy",    type=float, default=1.0, help="W vy.")
    # Optional early stopping.
    p.add_argument("--early_stop_patience", type=int, default=30)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-3)
    p.add_argument("--train_stop_threshold", type=float, default=None)
    # Post-processing after training.
    p.add_argument("--post_eval", default=False)
    # p.add_argument("--save_depth_maps", action="store_true")
    # p.add_argument("--save_flow_components", action="store_true")
    # p.add_argument("--save_velocity_maps", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument('--xy_npy', default='D:/Work/qyb/ResNet-18/data/xy.npy', help="Xy npy.")
    p.add_argument("--to_lonlat", default=True, help="To lonlat.")
    p.add_argument("--bg_image", type=str, default=r'D:\Work\qyb\Hec_Ras\YD\YD.tif', help="Bg image.")
    p.add_argument("--bg_alpha", type=float, default=0.5,help="Bg alpha.")

def _add_infer_args(p: argparse.ArgumentParser):
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu",
                   help="Inference device; cpu is the default for CPU timing comparisons.")
    _add_model_args(p)
    p.add_argument('--inflow', default='D:/Work/qyb/ResNet-18/data/test/inflow', help="Directory containing input hydrographs and derived features (.xlsx).")
    p.add_argument("--timesteps", type=int, default=24, help="Timesteps.")
    p.add_argument("--flood_threshold", type=float, default=0.1, help="Wet/dry depth threshold in metres (default: 0.1).")
    p.add_argument("--checkpoint", default=r'D:\Work\qyb\ResNet-18\V2.4\Result\20251212_210230_0.5_k256_depth, flow_resnet_basic\best_20251212_210230_0.5_k256_depth, flow_resnet_basic.pth', help="Trained checkpoint path.")
    p.add_argument("--norm_stats", default=r'D:\Work\qyb\ResNet-18\V2.4\Result\20251212_210230_0.5_k256_depth, flow_resnet_basic\best_20251212_210230_0.5_k256_depth, flow_resnet_basic.npz', help="Training normalisation NPZ; inferred from checkpoint if omitted.")
    p.add_argument("--output_dir", default="Infer_PCA_0727", help="Output dir.")
    p.add_argument("--val_block", type=int, default=200000, help="Number of spatial columns reconstructed per inference block.")
    p.add_argument("--mean_depth", type=str, default=r'D:\Work\qyb\ResNet-18\V2.4\Basis_WD\mean.npy', help="Depth mean.npy fallback when absent from checkpoint.")
    p.add_argument("--mean_vx", type=str, default=r'D:\Work\qyb\ResNet-18\V2.4\Basis_vx\mean.npy', help="Velocity-x mean.npy fallback when absent from checkpoint.")
    p.add_argument("--mean_vy", type=str, default=r'D:\Work\qyb\ResNet-18\V2.4\Basis_vy\mean.npy', help="Velocity-y mean.npy fallback when absent from checkpoint.")

def _add_basis_ipca_args(p: argparse.ArgumentParser):
    p.add_argument("--chunk_rows", type=int, default=8192, help="Chunk rows.")
    p.add_argument('--label_dir', default='D:/Work/qyb/ResNet-18/data/train/hdf80', help="Label dir.")
    p.add_argument('--timesteps', type=int, default=24, help="Timesteps.")
    p.add_argument('--n_components', type=int, default=256, help="N components.")
    p.add_argument('--append_mean', action='store_true', help="Append mean.")
    p.add_argument('--out_dir',     type=str, default="Basis_80vy", help="Out dir.")
    p.add_argument('--data_path', type=str, default="Results/Unsteady/Output/Output Blocks/Base Output/"
                  "Unsteady Time Series/2D Flow Areas/Perimeter 1/Cell Velocity - Velocity Y", help="Data path.")

def _add_basis_svd_args(p: argparse.ArgumentParser):
    p.add_argument("--hdf_dir", default='D:/Work/qyb/ResNet-18/data/train/hdf', help="Hdf dir.")
    p.add_argument("--pattern", default="*.hdf")
    p.add_argument("--rank_k", type=int, default=256)
    p.add_argument("--save", default="Basis_SVD_vy", help="Save.")
    p.add_argument("--data_path", default="Results/Unsteady/Output/Output Blocks/Base Output/"
              "Unsteady Time Series/2D Flow Areas/Perimeter 1/Cell Velocity - Velocity Y")
    p.add_argument("--svd_iter", type=int, default=5)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--wet_threshold", type=float, default=None)
    p.add_argument("--center", default=True, help="Center.")
    p.add_argument("--append_mean", action="store_true", help="Append mean.")

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
    # Input and output paths.
    p.add_argument("--pred_dir", type=str,
                   default=r"D:\Work\qyb\ResNet-18\data_dw\Infer_SVD_resnet",
                   help="Pred dir.")

    p.add_argument("--gt_dir", type=str,
                   default=r"D:\Work\qyb\ResNet-18\data_dw\test\hdf",
                   help="Gt dir.")

    p.add_argument("--output_dir", type=str, default=r"D:\Work\qyb\ResNet-18\data_dw\Infer_SVD_resnet\infer_SVD_resnet", help="Output dir.")

    # Evaluation and plotting switches.
    p.add_argument('--separate_panels', default=True, help="Separate panels.")
    p.add_argument("--threshold", type=float, default=0.1,
                   help="Wet/dry depth threshold in metres for masks and F1/POD/FAR.")
    p.add_argument("--combine", type=str2bool, nargs="?", const=True, default=False,
                   help="Combine.")
    p.add_argument("--pred_only", type=str2bool, nargs="?", const=True, default=False,
                   help="Pred only.")
    p.add_argument("--save_depth_maps", type=str2bool, nargs="?", const=True, default=True,
                   help="Save depth maps.")
    p.add_argument("--save_vel_maps", type=str2bool, nargs="?", const=True, default=True,
                   help="Save vel maps.")
    p.add_argument("--save_vel_component_maps", type=str2bool, nargs="?", const=True, default=True,
                   help="Save vel component maps.")
    p.add_argument("--save_metrics_per_timestep", type=str2bool, nargs="?", const=True, default=True,
                   help="Save metrics per timestep.")

    # Display thresholds used by postviz.
    p.add_argument("--min_show_depth", type=float, default=0.1,
                   help="Minimum plotted depth in metres; display setting only.")
    p.add_argument("--min_show_speed", type=float, default=0.001,
                   help="Minimum plotted speed in m/s; not a classification threshold.")
    p.add_argument("--outside_mask_alpha", type=float, default=0.0,
                   help="Outside mask alpha.")
    p.add_argument("--vmax_percentile", type=float, default=99,
                   help="Vmax percentile.")

    # Rasterisation and display settings; retained for compatibility with polygon mode.
    p.add_argument("--nx", type=int, default=1200, help="Nx.")
    p.add_argument("--smooth_sigma", type=float, default=0.8, help="Smooth sigma.")
    p.add_argument("--data_alpha", type=float, default=0.85, help="Data alpha.")
    p.add_argument("--interpolation", type=str, default="bilinear",
                   choices=["nearest", "bilinear", "bicubic"], help="Interpolation.")
    p.add_argument("--boundary_lw", type=float, default=1.5, help="Boundary lw.")
    p.add_argument("--vmax_mode", type=str, default="global",
                   choices=["global", "frame"], help="Vmax mode.")

    # Geometry, domain perimeter, and inundation confusion maps for polygon rendering.
    p.add_argument("--geom_hdf", type=str, default=r"D:\Work\qyb\ResNet-18\data_dw\test\hdf\01.hdf",
                   help="Geom hdf.")
    p.add_argument("--area_name", type=str, default="Perimeter 1",
                   help="Area name.")

    # Confusion rendering: auto prefers polygons, poly forces cell fills, raster uses raster voting.
    p.add_argument("--confusion_render", type=str, default="auto",
                   choices=["auto", "poly", "raster"],
                   help="Confusion render.")
    # Output format and canvas.
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--formats", type=str, default="png", help="Formats.")

    # Coordinate axes.
    p.add_argument("--show_axes", type=str2bool, nargs="?", const=True, default=False,
                   help="Show axes.")
    p.add_argument("--show_secondary_lonlat", type=str2bool, nargs="?", const=True, default=True,
                   help="Show secondary lonlat.")
    p.add_argument("--to_lonlat", type=str2bool, nargs="?", const=True, default=False,
                   help="To lonlat.")
    p.add_argument("--assume_xy_crs", type=str, default="EPSG:3857", help="Assume xy crs.")

    # Background imagery.
    p.add_argument("--bg_image", type=str, default=r"D:\Work\qyb\ResNet-18\data_dw\Remote_mengwa.tif",
                   help="Bg image.")
    p.add_argument("--bg_alpha", type=float, default=0.5)
    p.add_argument("--bg_stretch", type=str2bool, nargs="?", const=True, default=True,
                   help="Bg stretch.")
    p.add_argument("--bg_p_low", type=float, default=2.0)
    p.add_argument("--bg_p_high", type=float, default=98.0)


def _add_dump_xy_args(p: argparse.ArgumentParser):
    p.add_argument('--hdf_dir', default='D:/Work/qyb/ResNet-18/data/test/hdf', help="Hdf dir.")
    p.add_argument('--out', default='D:/Work/qyb/ResNet-18/data/xy_try.npy', help="Out.")
    p.add_argument('--tolerance', type=float, default=1e-6, help="Tolerance.")
    p.add_argument('--max_check', type=int, default=20, help="Max check.")

def main():
    ap = argparse.ArgumentParser("Flood Runner")
    sp = ap.add_subparsers(dest="cmd", required=True)

    # Training and inference.
    p_train = sp.add_parser("train", help="Train and validate a surrogate.")
    _add_train_args(p_train)

    p_infer = sp.add_parser("infer", help="Reconstruct flood events.")
    _add_infer_args(p_infer)

    # Basis construction using IPCA or SVD.
    p_bi = sp.add_parser("build-basis-ipca", help="Construct an incremental PCA basis.")
    _add_basis_ipca_args(p_bi)

    p_bs = sp.add_parser("build-basis-svd", help="Construct a truncated-SVD basis with optional centering and wet mask.")
    _add_basis_svd_args(p_bs)

    # Basis validation.
    p_chk = sp.add_parser("check-basis", help="Evaluate basis reconstruction errors.")
    _add_check_basis_args(p_chk)

    # Unified visualisation.
    p_viz = sp.add_parser("viz", help="Plot predictions and calculate metrics.")
    _add_viz_args(p_viz)

    # Coordinate export.
    p_xy = sp.add_parser("dump-xy", help="Check and export HEC-RAS cell coordinates to NPY.")
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
        raise SystemExit(f"Unknown subcommand: {args.cmd}")

if __name__ == "__main__":
    main()

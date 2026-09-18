# -*- coding: utf-8 -*-
import os, json, platform,time, hashlib, math, numpy as np, torch
from datetime import datetime
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import pandas as pd
from .losses import dry_region_penalty

from .data import FloodDataset
from .model import build_model
from .losses import rmse, mae, pcc, depth_weighted_loss
from .metrics import f1_pod_far
from .colsampler import make_epoch_chunks, StepCycler
from . import postviz  # 新增的可视化模块
import h5py
from .paths import DEPTH_PATH, VX_PATH, VY_PATH
from .sanity import check_nan_inf, enforce_nonnegative, sanitize_pred_for_save
from .plotstyle import enforce_times_new_roman
import matplotlib.pyplot as plt
enforce_times_new_roman()

def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _fmt_sec(sec: float) -> str:
    sec = float(sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"

def compute_velocity_mag_dir_torch(
    vx: torch.Tensor,
    vy: torch.Tensor,
    depth: torch.Tensor | None = None,
    depth_threshold: float | None = None,
    mask_value: float = float("nan"),
    eps: float = 1e-8,
):
    """
    根据 vx, vy 计算流速模长与流向。
    参数：
      vx, vy         : [B, T, H] 或 [T, H] 的张量（单位：m/s）
      depth          : 同形状水深张量（可选，用来做掩膜）
      depth_threshold: 水深阈值，小于此值的点视为“干地”，速度置 0，方向置为 NaN 或给定常数
      mask_value     : 干地上方向的填充值，默认 NaN
      eps            : 避免 sqrt(0) 的小常数

    返回：
      v_mag     : 速度模长，同形状
      v_dir_rad : 方向（弧度，[-π, π]）
      v_dir_deg : 方向（角度，[0, 360)）
    """
    # 统一到至少 3 维 [B, T, H]
    if vx.ndim == 2:
        vx = vx.unsqueeze(0)
        vy = vy.unsqueeze(0)
        if depth is not None and depth.ndim == 2:
            depth = depth.unsqueeze(0)

    v_mag = torch.sqrt(vx * vx + vy * vy + eps)
    v_dir_rad = torch.atan2(vy, vx)                         # [-π, π]
    v_dir_deg = v_dir_rad * 180.0 / math.pi                 # [-180, 180]
    v_dir_deg = (v_dir_deg + 360.0) % 360.0                 # [0, 360)

    if (depth is not None) and (depth_threshold is not None):
        mask = (depth < depth_threshold)
        v_mag = v_mag.masked_fill(mask, 0.0)
        if math.isnan(mask_value):
            fill = float("nan")
        else:
            fill = float(mask_value)
        v_dir_deg = v_dir_deg.masked_fill(mask, fill)
        v_dir_rad = v_dir_rad.masked_fill(mask, float("nan"))

    # 若最开始是 [T,H]，最后也还原成 [T,H]
    if v_mag.size(0) == 1:
        v_mag = v_mag.squeeze(0)
        v_dir_rad = v_dir_rad.squeeze(0)
        v_dir_deg = v_dir_deg.squeeze(0)

    return v_mag, v_dir_rad, v_dir_deg

def compute_velocity_mag_dir_np(
    vx_np: np.ndarray,
    vy_np: np.ndarray,
    depth_np: np.ndarray | None = None,
    depth_threshold: float | None = None,
    mask_value: float = np.nan,
    eps: float = 1e-12,
):
    """
    Numpy 版本：根据 vx, vy 计算速度模长与流向（方便保存/画图）。
    参数与返回值含义与 Torch 版一致，但全部是 numpy.ndarray。
    """
    vx_np = np.asarray(vx_np, dtype=np.float32)
    vy_np = np.asarray(vy_np, dtype=np.float32)

    v_mag = np.sqrt(vx_np * vx_np + vy_np * vy_np + eps)
    v_dir_rad = np.arctan2(vy_np, vx_np)        # [-π, π]
    v_dir_deg = np.degrees(v_dir_rad)           # [-180, 180]
    v_dir_deg = (v_dir_deg + 360.0) % 360.0     # [0, 360)

    if (depth_np is not None) and (depth_threshold is not None):
        depth_np = np.asarray(depth_np, dtype=np.float32)
        mask = depth_np < depth_threshold
        v_mag[mask] = 0.0
        if np.isnan(mask_value):
            v_dir_deg[mask] = np.nan
        else:
            v_dir_deg[mask] = mask_value
        v_dir_rad[mask] = np.nan

    return v_mag.astype(np.float32), v_dir_rad.astype(np.float32), v_dir_deg.astype(np.float32)

# --------- Multi-task helper: safely move y to device & split targets ---------
def _move_y_to_device(y, device):
    """
    将标签 y 移动到指定 device。
    支持:
      - Tensor: 直接 y.to(device)
      - dict   : 对每个 value 调用 to(device)，返回新的 dict
    """
    if isinstance(y, dict):
        return {
            k: (v.to(device, non_blocking=True) if v is not None else None)
            for k, v in y.items()
        }
    return y.to(device, non_blocking=True)


def _split_targets(y_dev):
    """
    从 y_dev 中拆出 depth / vx / vy 三个分量。

    返回:
      y_depth, y_vx, y_vy
    """
    if isinstance(y_dev, dict):
        y_depth = y_dev.get("depth", None)
        y_vx    = y_dev.get("vx", None)
        y_vy    = y_dev.get("vy", None)
    else:
        y_depth = y_dev
        y_vx = y_vy = None
    return y_depth, y_vx, y_vy


# ============== alpha/gamma 退火调度 ==============
def schedule_alpha_gamma(epoch, epochs,
                         alpha_base=0.5, alpha_max=1.0,
                         gamma_base=1.0, gamma_max=1.5,
                         warmup_ratio=0.3):
    ramp = int(warmup_ratio * epochs)
    if epoch <= ramp:
        # 前期弱加权
        return 0.25, 0.8
    p = (epoch - ramp) / max(1, (epochs - ramp))
    alpha = alpha_base + p * (alpha_max - alpha_base)
    gamma = gamma_base + p * (gamma_max - gamma_base)
    return float(alpha), float(gamma)

# ============== 后处理：保存 C / Y_pred / diff / 可视化 ==============
@torch.no_grad()
def post_evaluate_best_model(args, model, val_loader, device, H: int):
    """
    训练结束后：
      1) 用 best_* 权重逐样本推理，计算水深指标（RMSE/MAE/PCC/F1/POD/FAR），汇总到 CSV；
      2) 保存每个样本的水深预测 / 真值（Y_pred.npy / Y_true.npy）；
      3) 若为多任务并且存在 vx, vy：
           - 同时重建并保存 Vx_pred / Vy_pred 及其真值；
           - 计算并保存流速模长 |v| 和流向角度（度）；
           - 使用 postviz.save_flow_components 输出 vx/vy 分量 2×3 对比图；
           - 使用 postviz.save_velocity_maps 输出 |v| + 流向 2×2 对比图。
      4) 调用 postviz.save_depth_maps 对水深做四合一出图（若提供 XY）。
    注意：
      - 仅使用固定的 XY npy，不再从 HDF 读取 XY；
      - 多任务下 best_* 仍然是“水深验证 RMSE 最小”的最佳权重，流速是附加任务。
    """
    import matplotlib
    matplotlib.use("Agg")

    run_dir = args.run_dir
    tag     = args.tag
    out_dir = os.path.join(run_dir, "PostEval")
    os.makedirs(out_dir, exist_ok=True)

    # ----------------- 加载 best 权重 -----------------
    best_path = os.path.join(run_dir, f"best_{tag}.pth")
    state = torch.load(best_path, map_location=device)
    if isinstance(state, dict) and any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[PostEval] Loaded best model: {best_path}")

    # 多任务判定：是否存在 decoders / forward_coeffs
    is_multi = hasattr(model, "decoders") and hasattr(model, "forward_coeffs")

    # ---- 仅一次性加载全局 XY（来自 args.xy_npy），不再从 HDF 读取 ----
    global_xy = None
    if getattr(args, "xy_npy", None):
        if os.path.isfile(args.xy_npy):
            global_xy = np.load(args.xy_npy).astype(np.float32)
            if getattr(args, "to_lonlat", False):
                global_xy = postviz.webmerc_to_lonlat(global_xy)
            if global_xy.ndim != 2 or global_xy.shape[1] != 2:
                print(f"[PostEval][Warn] xy_npy 形状异常：{global_xy.shape}，忽略坐标绘图。")
                global_xy = None
            elif global_xy.shape[0] != H:
                print(f"[PostEval][Warn] xy_npy 的 H={global_xy.shape[0]} 与数据 H={H} 不一致，忽略坐标绘图。")
                global_xy = None
        else:
            print(f"[PostEval][Warn] 未找到 xy_npy 文件：{args.xy_npy}，将仅计算指标不绘图。")
    else:
        print("[PostEval][Note] 未提供 xy_npy，将仅计算指标不绘图。")

    # 若需要背景影像，在这里统一设置一次
    if getattr(args, "bg_image", None):
        postviz.set_background_image(args.bg_image, getattr(args, "bg_alpha", 0.5))
    else:
        postviz.set_background_image(None)

    # 可选：仅保留某些 sid
    def _maybe_parse_sid_list(s: str):
        if not s or s.strip().lower() == 'all':
            return None
        return set(t.strip() for t in s.split(',') if t.strip())

    sid_allow = _maybe_parse_sid_list(getattr(args, 'post_eval_sids', 'all'))

    save_depth_maps        = bool(getattr(args, 'save_depth_maps', False))
    save_flow_components   = bool(getattr(args, 'save_flow_components', False))
    save_velocity_maps_flag = bool(getattr(args, 'save_velocity_maps', False))
    if (save_depth_maps or save_flow_components or save_velocity_maps_flag) and (global_xy is None):
        print("[PostEval][Note] 未能使用 XY 坐标；将仅计算指标，不绘制深度/流速空间图。")

    block = max(1, min(int(getattr(args, 'post_block', 20000)), H))
    num_blocks = math.ceil(H / block)

    metrics_rows = []  # 汇总每个样本的指标（深度 + （可选）流速）

    flood_threshold = float(getattr(args, "flood_threshold", 0.1))
    dpi = int(getattr(args, "dpi", 150))

    def _to_device_y(y):
        """兼容 Tensor / dict 标签"""
        if isinstance(y, dict):
            return {k: v.to(device, non_blocking=True) for k, v in y.items()}
        else:
            return y.to(device, non_blocking=True)

    def _split_targets(y_dev):
        """
        - 单任务：y_dev 为 [B,T,H] Tensor
        - 多任务：y_dev 为 dict{'depth':..., 'vx':..., 'vy':...}
        """
        if isinstance(y_dev, dict):
            y_depth = y_dev.get("depth", None)
            y_vx    = y_dev.get("vx", None)
            y_vy    = y_dev.get("vy", None)
        else:
            y_depth = y_dev
            y_vx = y_vy = None
        if y_depth is None:
            raise RuntimeError("[PostEval] 需要提供 'depth' 标签。")
        return y_depth, y_vx, y_vy

    # ================== 主循环：遍历验证集样本 ==================
    for x, y, sid_batch in val_loader:
        x = x.to(device, non_blocking=True)
        y_dev = _to_device_y(y)

        # 拆标签（batch 级别）
        y_depth_all, y_vx_all, y_vy_all = _split_targets(y_dev)

        with autocast('cuda', enabled=torch.cuda.is_available()):
            if is_multi:
                # 多任务：先拿到各任务的 C
                C_dict   = model.forward_coeffs(x)   # dict: task -> [B,T,K]
                C_depth  = C_dict["depth"]
                C_vx     = C_dict.get("vx", None)
                C_vy     = C_dict.get("vy", None)
                Bsz, T, _ = C_depth.shape
            else:
                # 单任务：行为与原来一致
                C        = model(x)                  # [B,T,K]
                Bsz, T, _ = C.shape

        # 逐样本重建 / 计算指标
        for i in range(Bsz):
            sid = str(sid_batch[i])
            if sid_allow and sid not in sid_allow:
                continue

            # 深度预测/真值缓冲
            pred_i = np.zeros((T, H), dtype=np.float32)
            tgt_i  = np.zeros((T, H), dtype=np.float32)

            # vx / vy 缓冲（仅在多任务 + 有对应标签时启用）
            has_vx = is_multi and (C_vx is not None) and isinstance(y_dev, dict) and (y_vx_all is not None)
            has_vy = is_multi and (C_vy is not None) and isinstance(y_dev, dict) and (y_vy_all is not None)

            if has_vx:
                vx_pred_i = np.zeros((T, H), dtype=np.float32)
                vx_tgt_i  = np.zeros((T, H), dtype=np.float32)
            else:
                vx_pred_i = vx_tgt_i = None

            if has_vy:
                vy_pred_i = np.zeros((T, H), dtype=np.float32)
                vy_tgt_i  = np.zeros((T, H), dtype=np.float32)
            else:
                vy_pred_i = vy_tgt_i = None

            # -------- 分块重建 --------
            for b in range(num_blocks):
                s_idx = b * block
                e_idx = min(H, s_idx + block)
                cols = torch.arange(s_idx, e_idx, device=device)

                with autocast('cuda', enabled=torch.cuda.is_available()):
                    if is_multi:
                        # depth
                        piece_d = model.decoders["depth"].reconstruct(C_depth[i:i + 1], cols=cols)  # [1,T,S]
                        piece_d = check_nan_inf(piece_d, f"{sid}-d-piece", raise_on_nan=False, fix=True)
                        piece_d = enforce_nonnegative(piece_d)
                        tgt_d   = y_depth_all[i:i+1, :, s_idx:e_idx]

                        # vx
                        if has_vx:
                            piece_vx = model.decoders["vx"].reconstruct(C_vx[i:i + 1], cols=cols)
                            piece_vx = check_nan_inf(piece_vx, f"{sid}-vx-piece", raise_on_nan=False, fix=True)
                            tgt_vx   = y_vx_all[i:i+1, :, s_idx:e_idx]
                        # vy
                        if has_vy:
                            piece_vy = model.decoders["vy"].reconstruct(C_vy[i:i + 1], cols=cols)
                            piece_vy = check_nan_inf(piece_vy, f"{sid}-vy-piece", raise_on_nan=False, fix=True)
                            tgt_vy   = y_vy_all[i:i+1, :, s_idx:e_idx]
                    else:
                        piece_d = model.lowrank.reconstruct(C[i:i+1], cols=cols)  # [1,T,S]
                        piece_d = check_nan_inf(piece_d, f"{sid}-d-piece", raise_on_nan=False, fix=True)
                        piece_d = enforce_nonnegative(piece_d)
                        tgt_d   = y_depth_all[i:i+1, :, s_idx:e_idx]

                piece_d = torch.nan_to_num(piece_d, nan=0.0, posinf=0.0, neginf=0.0)
                pred_i[:, s_idx:e_idx] = piece_d.squeeze(0).detach().cpu().numpy()
                tgt_i[:,  s_idx:e_idx] = tgt_d.squeeze(0).detach().cpu().numpy()

                if has_vx:
                    piece_vx = torch.nan_to_num(piece_vx, nan=0.0, posinf=0.0, neginf=0.0)
                    vx_pred_i[:, s_idx:e_idx] = piece_vx.squeeze(0).detach().cpu().numpy()
                    vx_tgt_i[:,  s_idx:e_idx] = tgt_vx.squeeze(0).detach().cpu().numpy()
                if has_vy:
                    piece_vy = torch.nan_to_num(piece_vy, nan=0.0, posinf=0.0, neginf=0.0)
                    vy_pred_i[:, s_idx:e_idx] = piece_vy.squeeze(0).detach().cpu().numpy()
                    vy_tgt_i[:,  s_idx:e_idx] = tgt_vy.squeeze(0).detach().cpu().numpy()

                # 显式释放临时张量
                del piece_d, tgt_d
                if has_vx:
                    del piece_vx, tgt_vx
                if has_vy:
                    del piece_vy, tgt_vy
                torch.cuda.empty_cache()

            # ---- 水深指标（先算再保存）----
            p = torch.from_numpy(pred_i)
            g = torch.from_numpy(np.nan_to_num(tgt_i, nan=0.0))

            pred_i_save = sanitize_pred_for_save(pred_i, decimals=3)  # NaN→0, >=0, round(3)
            tgt_i_save  = np.nan_to_num(tgt_i.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

            np.save(os.path.join(out_dir, f"{sid}_Y_pred.npy"), pred_i_save)
            np.save(os.path.join(out_dir, f"{sid}_Y_true.npy"), tgt_i_save)

            RMSE = rmse(p, g).item()
            MAE  = mae(p, g).item()
            PCC  = pcc(p, g).item()
            F1, POD, FAR = f1_pod_far(p, g, threshold=flood_threshold)

            row = {
                "sid":       sid,
                "RMSE_depth": RMSE,
                "MAE_depth":  MAE,
                "PCC_depth":  PCC,
                "F1_depth":   F1,
                "POD_depth":  POD,
                "FAR_depth":  FAR
            }

            # ---- 若有 vx/vy，则保存流速文件 + 计算 |v| ----
            if has_vx:
                np.save(os.path.join(out_dir, f"{sid}_Vx_pred.npy"), vx_pred_i.astype(np.float32))
                np.save(os.path.join(out_dir, f"{sid}_Vx_true.npy"), vx_tgt_i.astype(np.float32))
                row["RMSE_vx"] = rmse(torch.from_numpy(vx_pred_i),
                                      torch.from_numpy(vx_tgt_i)).item()
                row["MAE_vx"]  = mae(torch.from_numpy(vx_pred_i),
                                     torch.from_numpy(vx_tgt_i)).item()
                row["PCC_vx"]  = pcc(torch.from_numpy(vx_pred_i),
                                     torch.from_numpy(vx_tgt_i)).item()

            if has_vy:
                np.save(os.path.join(out_dir, f"{sid}_Vy_pred.npy"), vy_pred_i.astype(np.float32))
                np.save(os.path.join(out_dir, f"{sid}_Vy_true.npy"), vy_tgt_i.astype(np.float32))
                row["RMSE_vy"] = rmse(torch.from_numpy(vy_pred_i),
                                      torch.from_numpy(vy_tgt_i)).item()
                row["MAE_vy"]  = mae(torch.from_numpy(vy_pred_i),
                                     torch.from_numpy(vy_tgt_i)).item()
                row["PCC_vy"]  = pcc(torch.from_numpy(vy_pred_i),
                                     torch.from_numpy(vy_tgt_i)).item()

            # 流速模长和流向：只有 vx 和 vy 都有时才计算
            if has_vx and has_vy:
                vmag_pred_i, vdir_rad_i, vdir_deg_i = compute_velocity_mag_dir_np(
                    vx_pred_i, vy_pred_i,
                    depth_np=pred_i,
                    depth_threshold=flood_threshold,
                    mask_value=np.nan
                )
                vmag_true_i, _, _ = compute_velocity_mag_dir_np(
                    vx_tgt_i, vy_tgt_i,
                    depth_np=tgt_i,
                    depth_threshold=flood_threshold,
                    mask_value=np.nan
                )
                np.save(os.path.join(out_dir, f"{sid}_Vmag_pred.npy"), vmag_pred_i)
                np.save(os.path.join(out_dir, f"{sid}_Vmag_true.npy"), vmag_true_i)
                np.save(os.path.join(out_dir, f"{sid}_Vdir_deg_pred.npy"), vdir_deg_i)

                row["RMSE_vmag"] = rmse(torch.from_numpy(vmag_pred_i),
                                        torch.from_numpy(vmag_true_i)).item()
                row["MAE_vmag"]  = mae(torch.from_numpy(vmag_pred_i),
                                       torch.from_numpy(vmag_true_i)).item()
                row["PCC_vmag"]  = pcc(torch.from_numpy(vmag_pred_i),
                                       torch.from_numpy(vmag_true_i)).item()

                # ✅ 流速可视化（仅当提供 XY 且用户开启选项）
                if global_xy is not None:
                    sample_dir = os.path.join(out_dir, sid)
                    os.makedirs(sample_dir, exist_ok=True)

                    # 2×3 vx/vy 分量对比图
                    if save_flow_components:
                        postviz.save_flow_components(
                            sid=sid,
                            xy=global_xy,
                            vx_pred=vx_pred_i,
                            vx_true=vx_tgt_i,
                            vy_pred=vy_pred_i,
                            vy_true=vy_tgt_i,
                            out_dir_sample=sample_dir,
                            dpi=dpi,
                        )

                    # 2×2 速度模长 + 流向图
                    if save_velocity_maps_flag:
                        pred_vel_frames = np.stack([vx_pred_i, vy_pred_i], axis=-1)  # [T,H,2]
                        true_vel_frames = np.stack([vx_tgt_i, vy_tgt_i], axis=-1)   # [T,H,2]
                        postviz.save_velocity_maps(
                            pred_vel_frames,
                            true_vel_frames,
                            xy=global_xy,
                            threshold=flood_threshold,
                            out_dir=sample_dir,
                            prefix=sid,
                            combine=True,
                            to_lonlat=False,
                            cmap="Blues",
                            diff_cmap="RdBu_r",
                            dpi=dpi,
                        )

            metrics_rows.append(row)

            print(f"[PostEval] {sid}  RMSE={RMSE:.4f} MAE={MAE:.4f} PCC={PCC:.4f} "
                  f"F1={F1:.3f} POD={POD:.3f} FAR={FAR:.3f}")

            # 深度 4 合 1 图
            if save_depth_maps and (global_xy is not None):
                sample_dir = os.path.join(out_dir, sid)
                os.makedirs(sample_dir, exist_ok=True)
                postviz.save_depth_maps(
                    pred_i, tgt_i, global_xy,
                    threshold=flood_threshold,
                    out_dir=sample_dir,
                    prefix=sid,
                    combine=True,
                    to_lonlat=False,  # XY 已在上面按需转过
                    cmap="Blues",
                    diff_cmap="RdBu_r",
                    mask_cmap="gray_r",
                    dpi=dpi,
                )

    # 保存指标汇总 CSV
    if metrics_rows:
        df = pd.DataFrame(metrics_rows)
        csv_path = os.path.join(out_dir, "metrics_summary.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[PostEval] 指标汇总已保存：{csv_path}")
    else:
        print("[PostEval] 无可用样本指标（可能被 sid 过滤）。")



# ======================= 主训练流程 =========================
def train_and_validate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====== 生成 run 目录 ======
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    brief = f"{args.beta_dry}_k{args.rank_k}_{args.tasks}_{args.backbone}"
    sig_src = json.dumps(vars(args), sort_keys=True, ensure_ascii=False)
    short_sig = hashlib.md5(sig_src.encode('utf-8')).hexdigest()[:6]
    run_dir_name = f"{ts}_{brief}"
    run_dir = os.path.join(args.save_dir, run_dir_name)
    os.makedirs(run_dir, exist_ok=False)
    tag = f"{ts}_{brief}"
    print(f"[RunDir] 本次训练产物保存到：{run_dir}")
    args.run_dir = run_dir; args.tag = tag
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "latest_run.txt"), "w", encoding="utf-8") as f:
        f.write(run_dir + "\n"); f.write(tag + "\n")

    # ===== 训练样本日志设置 =====
    log_train_sids = True           # 开关：是否记录样本 sid
    log_every_n    = 50             # 控制台每隔 N 个 batch 打印一次 sid
    repeat_warn_threshold = 10      # 连续相同 sid 的阈值（batch_size=1 时更有效）

    sid_log_csv = os.path.join(run_dir, "train_sample_log.csv")
    with open(sid_log_csv, "w", encoding="utf-8") as f:
        f.write("epoch,batch,sids\n")
    print(f"[TrainSID] 将把训练样本记录到：{sid_log_csv}")

    # ===== 数据与归一化 =====
    train_set = FloodDataset(args.inflow, args.HDF5, timesteps=args.timesteps, is_train=True)
    x_min, x_max = train_set.x_min.copy(), train_set.x_max.copy()
    if args.val_inflow and args.val_hdf5:
        val_set = FloodDataset(args.val_inflow, args.val_hdf5, timesteps=args.timesteps,
                               is_train=False, ref_min=x_min, ref_max=x_max)
    else:
        total = len(train_set); val_n = max(1, int(total*0.2)); train_n = total - val_n
        train_set, val_set = torch.utils.data.random_split(train_set, [train_n, val_n])

    if args.preload:
        for _ in range(len(train_set)): _ = train_set[_]
        for _ in range(len(val_set)): _ = val_set[_]

    workers = 0 if platform.system().lower().startswith('win') else args.workers
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle_train,
                              num_workers=workers, pin_memory=True)
    val_loader  = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=workers, pin_memory=True)

    # ===== 模型与优化器/调度器 =====
    sx, sy, _ = train_set[0]
    # y 现在可能是 Tensor 或 dict(depth/vx/vy)，这里只用 depth 决定 T,H
    if isinstance(sy, dict):
        sy_depth = sy.get("depth", None)
        assert sy_depth is not None, "FloodDataset 返回的 y 为 dict 时必须包含 'depth' 键"
    else:
        sy_depth = sy
    C_in = sx.shape[0]; T = sy_depth.shape[0]; H = sy_depth.shape[1]

    # >>> [PATCH-1] 数据来源与 y 的尺度检查
    try:
        print(f"[DataPath] FloodDataset DEPTH_PATH = {DEPTH_PATH}")
        print(f"[DataPath] FloodDataset VX_PATH = {VX_PATH}")
        print(f"[DataPath] FloodDataset VY_PATH = {VY_PATH}")
    except Exception:
        print("[DataPath][WARN] 无法从 paths.py 读取 DEPTH_PATH")

    model = build_model(args, C_in, T, H, device)
    opt   = torch.optim.NAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    scheduler = ReduceLROnPlateau(
        opt, mode='min',
        patience=args.plateau_patience,
        factor=args.plateau_factor,
        min_lr=args.min_lr
    )

    # 保存归一化统计
    norm_stats_path = os.path.join(run_dir, f"norm_stats_{tag}.npz")
    np.savez(norm_stats_path, x_min=x_min, x_max=x_max)

    best_val = float('inf')
    best_path = os.path.join(run_dir, f"best_{tag}.pth")
    best_metrics = None  # 记录最佳指标
    # 早停状态
    no_improve = 0
    min_delta = float(getattr(args, 'early_stop_min_delta', 1e-4)) if hasattr(args,'early_stop_min_delta') else 1e-4
    es_patience = int(getattr(args, 'early_stop_patience', 0)) if hasattr(args,'early_stop_patience') else 0

    # ===== 训练计时：总训练耗时（不含 post_eval）=====
    _sync_cuda()
    train_time_start = time.perf_counter()

    # ===== 列块生成 =====
    def make_chunks(total_H: int, chunk_size: int, device: torch.device):
        n = (total_H + chunk_size - 1) // chunk_size
        return [torch.arange(i*chunk_size, min(total_H, (i+1)*chunk_size), device=device) for i in range(n)]

    # ---------------- 工具：记录与监控当前 batch 的 sids ----------------
    def _record_sids(epoch:int, step:int, sid_batch, pbar=None, epoch_sids=None, last_sid_seq=None):
        # 统一处理为 list[str]
        if isinstance(sid_batch, (list, tuple)):
            sids = [str(s) for s in sid_batch]
        else:
            sids = [str(sid_batch)]
        if epoch_sids is not None:
            epoch_sids.extend(sids)
        # 写 CSV
        with open(sid_log_csv, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{step},\"{';'.join(sids)}\"\n")
        # 控制台节流打印
        if (step % log_every_n == 0) or (step == 1):
            if pbar is not None:
                pbar.set_postfix_str(f"sid={','.join(sids)}")
            else:
                print(f"[TrainSID] E{epoch:03d} step{step}: sid={','.join(sids)}")
        # 连续重复检测（针对 batch_size=1 最敏感）
        if last_sid_seq is not None and len(sids) == 1:
            cur = sids[0]
            if (len(last_sid_seq)==0) or (cur == last_sid_seq[-1]):
                last_sid_seq.append(cur)
                if len(last_sid_seq) == repeat_warn_threshold:
                    print(f"[Warn][Epoch {epoch}] 连续 {repeat_warn_threshold} 个 batch 都是同一 sid={cur}。"
                          f"请检查 DataLoader 是否重复喂样本 / shuffle_train={getattr(args,'shuffle_train',False)}。")
            else:
                last_sid_seq.clear(); last_sid_seq.append(cur)

    # ================= one_epoch_train（支持 depth + vx + vy，多任务损失权重） =================
    def one_epoch_train(epoch: int):
        """
        单轮训练：
        - 保持原有 depth 训练逻辑（rmse + depth_weighted_loss + C 平滑 + 干区惩罚）
        - 若模型/数据同时提供 vx, vy 任务，则额外加入对应的 RMSE 损失，
          并按 w_depth / w_vx / w_vy 进行加权。
        """
        model.train()
        run = 0.0
        steps = 0

        accum = max(1, int(getattr(args, 'accum_steps', 1)))
        col_mode = getattr(args, 'col_mode', 'chunk_in_batch')
        use_sampling = (args.sample_h and args.sample_h < H)
        pbar = tqdm(train_loader, desc=f"Train E{epoch:03d}", unit="batch", leave=False)

        # ---------- 动态 alpha / gamma（与原逻辑完全一致） ----------
        alpha_t, gamma_t = schedule_alpha_gamma(
            epoch, args.epochs,
            alpha_base=args.alpha_depth_loss, alpha_max=args.alpha_target,
            gamma_base=args.depth_loss_gamma, gamma_max=args.gamma_target,
            warmup_ratio=args.warmup_ratio
        )
        # C 时间平滑正则
        lam_smooth  = float(getattr(args, 'smooth_c_lambda', 0.0))
        huber_delta = float(getattr(args, 'huber_delta', 0.1))
        w_clip_q    = float(getattr(args, 'w_clip_q', 0.999))

        # ---------- 多任务损失权重 ----------
        # 建议默认：depth 作为主任务，速度作为辅任务：1 : 0.5 : 0.5
        # 若想更弱一点，可以在命令行里改成 1 : 0.3 : 0.3
        w_depth = float(getattr(args, 'w_depth', 1.0))
        w_vx    = float(getattr(args, 'w_vx', 0.5))
        w_vy    = float(getattr(args, 'w_vy', 0.5))

        def _to_device_y(y):
            """兼容 Tensor / dict 两种标签"""
            if isinstance(y, dict):
                return {k: v.to(device, non_blocking=True) for k, v in y.items()}
            else:
                return y.to(device, non_blocking=True)

        def _split_targets(y_dev):
            """
            - 单任务：y_dev 为 [B,T,H] Tensor
            - 多任务：y_dev 为 dict{ 'depth':..., 'vx':..., 'vy':... }
            """
            if isinstance(y_dev, dict):
                y_depth = y_dev.get("depth", None)
                y_vx    = y_dev.get("vx", None)
                y_vy    = y_dev.get("vy", None)
            else:
                y_depth = y_dev
                y_vx = y_vy = None
            if y_depth is None:
                raise RuntimeError("训练阶段需要提供 'depth' 标签。")
            return y_depth, y_vx, y_vy

        is_multi = hasattr(model, "decoders")  # Multi-task 模型：有 decoders 字典

        # 本轮样本收集与重复检测缓冲
        epoch_sids, last_sid_seq = [], []

        # ====== col_mode = full（或不采样）→ 与原 depth 逻辑完全一致 + 额外速度损失 ======
        if (not use_sampling) or col_mode == 'full':
            opt.zero_grad(set_to_none=True)

            for step, (x, y, sid_batch) in enumerate(pbar, 1):
                if log_train_sids:
                    _record_sids(epoch, step, sid_batch, pbar, epoch_sids, last_sid_seq)

                x = x.to(device, non_blocking=True)
                y = _to_device_y(y)

                with autocast('cuda', enabled=torch.cuda.is_available()):
                    if is_multi:
                        # --- Multi-task：先算各任务的 C，然后用 decoder 重建 ---
                        C_dict = model.forward_coeffs(x)      # dict: task -> [B,T,K]
                        C_depth = C_dict["depth"]

                        pred_depth = model.decoders["depth"].reconstruct(C_depth, cols=None)
                        pred_depth = check_nan_inf(pred_depth, "pred-train", raise_on_nan=False, fix=True)
                        pred_depth = enforce_nonnegative(pred_depth)  # 只对水深施加非负约束

                        tgt_depth, tgt_vx, tgt_vy = _split_targets(y)

                        # --- depth 损失：保持原公式 ---
                        base = rmse(pred_depth, tgt_depth)
                        aux  = depth_weighted_loss(
                            pred_depth, tgt_depth,
                            gamma=gamma_t, use_huber=True,
                            delta=huber_delta, clip_q=w_clip_q
                        )
                        if lam_smooth > 0:
                            Cdiff = C_depth[:, 1:, :] - C_depth[:, :-1, :]
                            smooth = (Cdiff.pow(2).mean()) * lam_smooth
                        else:
                            smooth = C_depth.new_tensor(0.0)

                        beta_dry = float(getattr(args, "beta_dry", 0.0))
                        dry_pen = dry_region_penalty(
                            pred_depth, tgt_depth,
                            threshold=float(getattr(args, "flood_threshold", 0.1)),
                            use_huber=True, delta=0.05
                        ) if beta_dry > 0 else C_depth.new_tensor(0.0)

                        L_depth = base + alpha_t * aux + smooth + beta_dry * dry_pen

                        # --- vx, vy 损失：简单 RMSE，作为辅助任务 ---
                        total = w_depth * L_depth

                        if ("vx" in C_dict) and ("vx" in getattr(model, "decoders", {})) and (tgt_vx is not None):
                            pred_vx = model.decoders["vx"].reconstruct(C_dict["vx"], cols=None)
                            L_vx = rmse(pred_vx, tgt_vx)
                            total = total + w_vx * L_vx

                        if ("vy" in C_dict) and ("vy" in getattr(model, "decoders", {})) and (tgt_vy is not None):
                            pred_vy = model.decoders["vy"].reconstruct(C_dict["vy"], cols=None)
                            L_vy = rmse(pred_vy, tgt_vy)
                            total = total + w_vy * L_vy

                    else:
                        # --- 单任务：完全维持原先 depth 逻辑 ---
                        C = model(x)                                        # [B,T,K]
                        pred = model.lowrank.reconstruct(C, cols=None)      # [B,T,H]
                        pred = check_nan_inf(pred, "pred-train", raise_on_nan=False, fix=True)
                        pred = enforce_nonnegative(pred)                    # 物理约束：深度>=0

                        tgt_depth, _, _ = _split_targets(y)

                        base = rmse(pred, tgt_depth)
                        aux  = depth_weighted_loss(
                            pred, tgt_depth,
                            gamma=gamma_t, use_huber=True,
                            delta=huber_delta, clip_q=w_clip_q
                        )
                        if lam_smooth > 0:
                            Cdiff = C[:, 1:, :] - C[:, :-1, :]
                            smooth = (Cdiff.pow(2).mean()) * lam_smooth
                        else:
                            smooth = C.new_tensor(0.0)

                        beta_dry = float(getattr(args, "beta_dry", 0.0))
                        dry_pen = dry_region_penalty(
                            pred, tgt_depth,
                            threshold=float(getattr(args, "flood_threshold", 0.1)),
                            use_huber=True, delta=0.05
                        ) if beta_dry > 0 else C.new_tensor(0.0)

                        L_depth = base + alpha_t * aux + smooth + beta_dry * dry_pen
                        total = w_depth * L_depth  # 单任务下只包含 depth

                    loss = total / accum

                scaler.scale(loss).backward()

                if step % accum == 0:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

                run += loss.item() * accum
                steps += 1
                pbar.set_postfix(
                    avg=f"{run/max(1,steps):.6f}",
                    a=f"{alpha_t:.2f}",
                    g=f"{gamma_t:.2f}"
                )

            if steps % accum != 0:
                scaler.step(opt)
                scaler.update()

            return run / max(1, steps)

        # ====== col_mode = chunk_in_batch：列块采样，逐块重建 ======
        if col_mode == 'chunk_in_batch':
            chunks = make_chunks(H, args.sample_h, device)
            opt.zero_grad(set_to_none=True)

            for step, (x, y, sid_batch) in enumerate(pbar, 1):
                if log_train_sids:
                    _record_sids(epoch, step, sid_batch, pbar, epoch_sids, last_sid_seq)

                x = x.to(device, non_blocking=True)
                y = _to_device_y(y)

                with autocast('cuda', enabled=torch.cuda.is_available()):
                    if is_multi:
                        C_dict = model.forward_coeffs(x)
                        C_depth = C_dict["depth"]
                    else:
                        C_depth = model(x)

                tgt_depth_full, tgt_vx_full, tgt_vy_full = _split_targets(y)
                total_loss = 0.0

                for ci, cols in enumerate(chunks, 1):
                    with autocast('cuda', enabled=torch.cuda.is_available()):
                        if is_multi:
                            # depth
                            pred_depth = model.decoders["depth"].reconstruct(C_depth, cols=cols)
                            pred_depth = check_nan_inf(pred_depth, "pred-train", raise_on_nan=False, fix=True)
                            pred_depth = enforce_nonnegative(pred_depth)
                            tgt_depth = tgt_depth_full[:, :, cols]

                            base = rmse(pred_depth, tgt_depth)
                            aux  = depth_weighted_loss(
                                pred_depth, tgt_depth,
                                gamma=gamma_t, use_huber=True,
                                delta=huber_delta, clip_q=w_clip_q
                            )
                            if lam_smooth > 0:
                                Cdiff = C_depth[:, 1:, :] - C_depth[:, :-1, :]
                                smooth = (Cdiff.pow(2).mean()) * lam_smooth
                            else:
                                smooth = C_depth.new_tensor(0.0)

                            L_depth = base + alpha_t * aux + smooth
                            total = w_depth * L_depth

                            # vx
                            if ("vx" in C_dict) and ("vx" in getattr(model, "decoders", {})) and (tgt_vx_full is not None):
                                pred_vx = model.decoders["vx"].reconstruct(C_dict["vx"], cols=cols)
                                tgt_vx = tgt_vx_full[:, :, cols]
                                L_vx = rmse(pred_vx, tgt_vx)
                                total = total + w_vx * L_vx

                            # vy
                            if ("vy" in C_dict) and ("vy" in getattr(model, "decoders", {})) and (tgt_vy_full is not None):
                                pred_vy = model.decoders["vy"].reconstruct(C_dict["vy"], cols=cols)
                                tgt_vy = tgt_vy_full[:, :, cols]
                                L_vy = rmse(pred_vy, tgt_vy)
                                total = total + w_vy * L_vy

                        else:
                            pred = model.lowrank.reconstruct(C_depth, cols=cols)
                            pred = check_nan_inf(pred, "pred-train", raise_on_nan=False, fix=True)
                            pred = enforce_nonnegative(pred)
                            tgt_depth = tgt_depth_full[:, :, cols]

                            base = rmse(pred, tgt_depth)
                            aux  = depth_weighted_loss(
                                pred, tgt_depth,
                                gamma=gamma_t, use_huber=True,
                                delta=huber_delta, clip_q=w_clip_q
                            )
                            if lam_smooth > 0:
                                Cdiff = C_depth[:, 1:, :] - C_depth[:, :-1, :]
                                smooth = (Cdiff.pow(2).mean()) * lam_smooth
                            else:
                                smooth = C_depth.new_tensor(0.0)

                            L_depth = base + alpha_t * aux + smooth
                            total = w_depth * L_depth

                        loss = total / len(chunks) / accum

                    retain = (ci != len(chunks))
                    scaler.scale(loss).backward(retain_graph=retain)
                    total_loss += loss.item()

                if step % accum == 0:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

                run += total_loss * accum
                steps += 1
                pbar.set_postfix(
                    avg=f"{run/max(1,steps):.6f}",
                    a=f"{alpha_t:.2f}",
                    g=f"{gamma_t:.2f}"
                )

            if steps % accum != 0:
                scaler.step(opt)
                scaler.update()

            if log_train_sids:
                uniq = len(set(epoch_sids))
                print(
                    f"[TrainSID][Epoch {epoch}] 本轮训练样本数：{len(epoch_sids)}，去重后：{uniq} 个。"
                    f"（batch_size={args.batch_size}, shuffle_train={getattr(args,'shuffle_train',False)}）"
                )
            return run / max(1, steps)

        # ====== col_mode = random：每 step 随机采样列 ======
        if col_mode == 'random':
            opt.zero_grad(set_to_none=True)

            for step, (x, y, sid_batch) in enumerate(pbar, 1):
                if log_train_sids:
                    _record_sids(epoch, step, sid_batch, pbar, epoch_sids, last_sid_seq)

                x = x.to(device, non_blocking=True)
                y = _to_device_y(y)
                cols = torch.randint(0, H, (args.sample_h,), device=device)

                with autocast('cuda', enabled=torch.cuda.is_available()):
                    if is_multi:
                        C_dict = model.forward_coeffs(x)
                        C_depth = C_dict["depth"]

                        pred_depth = model.decoders["depth"].reconstruct(C_depth, cols=cols)
                        pred_depth = check_nan_inf(pred_depth, "pred-train", raise_on_nan=False, fix=True)
                        pred_depth = enforce_nonnegative(pred_depth)

                        tgt_depth_full, tgt_vx_full, tgt_vy_full = _split_targets(y)
                        tgt_depth = tgt_depth_full[:, :, cols]

                        base = rmse(pred_depth, tgt_depth)
                        aux  = depth_weighted_loss(
                            pred_depth, tgt_depth,
                            gamma=gamma_t, use_huber=True,
                            delta=huber_delta, clip_q=w_clip_q
                        )
                        if lam_smooth > 0:
                            Cdiff = C_depth[:, 1:, :] - C_depth[:, :-1, :]
                            smooth = (Cdiff.pow(2).mean()) * lam_smooth
                        else:
                            smooth = C_depth.new_tensor(0.0)

                        L_depth = base + alpha_t * aux + smooth
                        total = w_depth * L_depth

                        if ("vx" in C_dict) and ("vx" in getattr(model, "decoders", {})) and (tgt_vx_full is not None):
                            pred_vx = model.decoders["vx"].reconstruct(C_dict["vx"], cols=cols)
                            tgt_vx = tgt_vx_full[:, :, cols]
                            L_vx = rmse(pred_vx, tgt_vx)
                            total = total + w_vx * L_vx

                        if ("vy" in C_dict) and ("vy" in getattr(model, "decoders", {})) and (tgt_vy_full is not None):
                            pred_vy = model.decoders["vy"].reconstruct(C_dict["vy"], cols=cols)
                            tgt_vy = tgt_vy_full[:, :, cols]
                            L_vy = rmse(pred_vy, tgt_vy)
                            total = total + w_vy * L_vy

                    else:
                        C = model(x)
                        pred = model.lowrank.reconstruct(C, cols=cols)
                        pred = check_nan_inf(pred, "pred-train", raise_on_nan=False, fix=True)
                        pred = enforce_nonnegative(pred)

                        tgt_depth_full, _, _ = _split_targets(y)
                        tgt_depth = tgt_depth_full[:, :, cols]

                        base = rmse(pred, tgt_depth)
                        aux  = depth_weighted_loss(
                            pred, tgt_depth,
                            gamma=gamma_t, use_huber=True,
                            delta=huber_delta, clip_q=w_clip_q
                        )
                        if lam_smooth > 0:
                            Cdiff = C[:, 1:, :] - C[:, :-1, :]
                            smooth = (Cdiff.pow(2).mean()) * lam_smooth
                        else:
                            smooth = C.new_tensor(0.0)

                        L_depth = base + alpha_t * aux + smooth
                        total = w_depth * L_depth

                    loss = total / accum

                scaler.scale(loss).backward()

                if step % accum == 0:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

                run += loss.item() * accum
                steps += 1
                pbar.set_postfix(
                    avg=f"{run/max(1,steps):.6f}",
                    a=f"{alpha_t:.2f}",
                    g=f"{gamma_t:.2f}"
                )

            if steps % accum != 0:
                scaler.step(opt)
                scaler.update()

            if log_train_sids:
                uniq = len(set(epoch_sids))
                print(
                    f"[TrainSID][Epoch {epoch}] 本轮训练样本数：{len(epoch_sids)}，去重后：{uniq} 个。"
                    f"（batch_size={args.batch_size}, shuffle_train={getattr(args,'shuffle_train',False)}）"
                )
            return run / max(1, steps)

        # ====== col_mode = cycle_step：逐 step 轮换采样列 ======
        if col_mode == 'cycle_step':
            cycler = StepCycler(H, args.sample_h, shuffle=True, device=device)
            opt.zero_grad(set_to_none=True)

            for step, (x, y, sid_batch) in enumerate(pbar, 1):
                if log_train_sids:
                    _record_sids(epoch, step, sid_batch, pbar, epoch_sids, last_sid_seq)

                x = x.to(device, non_blocking=True)
                y = _to_device_y(y)
                cols = cycler.next()

                with autocast('cuda', enabled=torch.cuda.is_available()):
                    if is_multi:
                        C_dict = model.forward_coeffs(x)
                        C_depth = C_dict["depth"]

                        pred_depth = model.decoders["depth"].reconstruct(C_depth, cols=cols)
                        pred_depth = check_nan_inf(pred_depth, "pred-train", raise_on_nan=False, fix=True)
                        pred_depth = enforce_nonnegative(pred_depth)

                        tgt_depth_full, tgt_vx_full, tgt_vy_full = _split_targets(y)
                        tgt_depth = tgt_depth_full[:, :, cols]

                        base = rmse(pred_depth, tgt_depth)
                        aux  = depth_weighted_loss(
                            pred_depth, tgt_depth,
                            gamma=gamma_t, use_huber=True,
                            delta=huber_delta, clip_q=w_clip_q
                        )
                        if lam_smooth > 0:
                            Cdiff = C_depth[:, 1:, :] - C_depth[:, :-1, :]
                            smooth = (Cdiff.pow(2).mean()) * lam_smooth
                        else:
                            smooth = C_depth.new_tensor(0.0)

                        L_depth = base + alpha_t * aux + smooth
                        total = w_depth * L_depth

                        if ("vx" in C_dict) and ("vx" in getattr(model, "decoders", {})) and (tgt_vx_full is not None):
                            pred_vx = model.decoders["vx"].reconstruct(C_dict["vx"], cols=cols)
                            tgt_vx = tgt_vx_full[:, :, cols]
                            L_vx = rmse(pred_vx, tgt_vx)
                            total = total + w_vx * L_vx

                        if ("vy" in C_dict) and ("vy" in getattr(model, "decoders", {})) and (tgt_vy_full is not None):
                            pred_vy = model.decoders["vy"].reconstruct(C_dict["vy"], cols=cols)
                            tgt_vy = tgt_vy_full[:, :, cols]
                            L_vy = rmse(pred_vy, tgt_vy)
                            total = total + w_vy * L_vy

                    else:
                        C = model(x)
                        pred = model.lowrank.reconstruct(C, cols=cols)  # [B,T,S] 或 [B,T,H]
                        pred = check_nan_inf(pred, "pred-train", raise_on_nan=False, fix=True)
                        pred = enforce_nonnegative(pred)  # 物理约束：深度>=0

                        tgt_depth_full, _, _ = _split_targets(y)
                        tgt_depth = tgt_depth_full[:, :, cols]

                        base = rmse(pred, tgt_depth)
                        aux  = depth_weighted_loss(
                            pred, tgt_depth,
                            gamma=gamma_t, use_huber=True,
                            delta=huber_delta, clip_q=w_clip_q
                        )
                        if lam_smooth > 0:
                            Cdiff = C[:, 1:, :] - C[:, :-1, :]
                            smooth = (Cdiff.pow(2).mean()) * lam_smooth
                        else:
                            smooth = C.new_tensor(0.0)

                        L_depth = base + alpha_t * aux + smooth
                        total = w_depth * L_depth

                    loss = total / accum

                scaler.scale(loss).backward()

                if step % accum == 0:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

                run += loss.item() * accum
                steps += 1
                pbar.set_postfix(
                    avg=f"{run/max(1,steps):.6f}",
                    a=f"{alpha_t:.2f}",
                    g=f"{gamma_t:.2f}"
                )

            if steps % accum != 0:
                scaler.step(opt)
                scaler.update()

            if log_train_sids:
                uniq = len(set(epoch_sids))
                print(
                    f"[TrainSID][Epoch {epoch}] 本轮训练样本数：{len(epoch_sids)}，去重后：{uniq} 个。"
                    f"（batch_size={args.batch_size}, shuffle_train={getattr(args,'shuffle_train',False)}）"
                )
            return run / max(1, steps)

        # 理论上不会走到这里
        return run / max(1, max(steps, 1))

    # ================= one_epoch_val（Full-H，每轮必跑，多任务指标） =================
    def one_epoch_val(epoch: int):
        """
        单轮验证：
        - 深度：RMSE / MAE / PCC / F1 / POD / FAR（保持原逻辑）
        - 速度 vx, vy：只计算 RMSE / MAE / PCC，不参与 F1/POD/FAR。
        返回：(depth_metrics_tuple, flow_metrics_dict)
        depth_metrics_tuple = (rmse, mae, pcc, f1, pod, far)
        flow_metrics_dict   = { 'rmse_vx':..., 'mae_vx':..., 'pcc_vx':..., 'rmse_vy':..., ... }
        """
        model.eval()

        # 深度指标
        s_rmse = s_mae = s_pcc = 0.0
        s_f1 = s_pod = s_far = 0.0
        n = 0

        # 速度指标
        s_rmse_vx = s_mae_vx = s_pcc_vx = 0.0; n_vx = 0
        s_rmse_vy = s_mae_vy = s_pcc_vy = 0.0; n_vy = 0

        def _to_device_y(y):
            if isinstance(y, dict):
                return {k: v.to(device, non_blocking=True) for k, v in y.items()}
            else:
                return y.to(device, non_blocking=True)

        def _split_targets(y_dev):
            if isinstance(y_dev, dict):
                y_depth = y_dev.get("depth", None)
                y_vx    = y_dev.get("vx", None)
                y_vy    = y_dev.get("vy", None)
            else:
                y_depth = y_dev
                y_vx = y_vy = None
            if y_depth is None:
                raise RuntimeError("验证阶段需要提供 'depth' 标签。")
            return y_depth, y_vx, y_vy

        is_multi = hasattr(model, "decoders")

        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f" Valid E{epoch:03d}", unit="batch", leave=False)
            for x, y, _ in vbar:
                x = x.to(device, non_blocking=True)
                y = _to_device_y(y)

                with autocast('cuda', enabled=torch.cuda.is_available()):
                    if is_multi:
                        C_dict = model.forward_coeffs(x)
                        C_depth = C_dict["depth"]

                        out_depth = model.decoders["depth"].reconstruct(C_depth, cols=None)
                        out_depth = check_nan_inf(out_depth, "pred-val", raise_on_nan=False, fix=True)
                        out_depth = enforce_nonnegative(out_depth)

                        tgt_depth, tgt_vx, tgt_vy = _split_targets(y)

                        pred_vx = pred_vy = None
                        if ("vx" in C_dict) and ("vx" in getattr(model, "decoders", {})) and (tgt_vx is not None):
                            pred_vx = model.decoders["vx"].reconstruct(C_dict["vx"], cols=None)
                        if ("vy" in C_dict) and ("vy" in getattr(model, "decoders", {})) and (tgt_vy is not None):
                            pred_vy = model.decoders["vy"].reconstruct(C_dict["vy"], cols=None)

                    else:
                        C = model(x)
                        out_depth = model.lowrank.reconstruct(C, cols=None)  # Full-H
                        out_depth = check_nan_inf(out_depth, "pred-val", raise_on_nan=False, fix=True)
                        out_depth = enforce_nonnegative(out_depth)

                        tgt_depth, tgt_vx, tgt_vy = _split_targets(y)
                        pred_vx = pred_vy = None

                    # --- 水深指标（保持原逻辑） ---
                    s_rmse += rmse(out_depth, tgt_depth).item()
                    s_mae  += mae(out_depth, tgt_depth).item()
                    s_pcc  += pcc(out_depth, tgt_depth).item()
                    f1, pod, far = f1_pod_far(out_depth, tgt_depth, threshold=args.flood_threshold)
                    s_f1  += f1
                    s_pod += pod
                    s_far += far
                    n += 1

                    # --- 速度指标：只算 RMSE/MAE/PCC ---
                    if (pred_vx is not None) and (tgt_vx is not None):
                        s_rmse_vx += rmse(pred_vx, tgt_vx).item()
                        s_mae_vx  += mae(pred_vx, tgt_vx).item()
                        s_pcc_vx  += pcc(pred_vx, tgt_vx).item()
                        n_vx += 1

                    if (pred_vy is not None) and (tgt_vy is not None):
                        s_rmse_vy += rmse(pred_vy, tgt_vy).item()
                        s_mae_vy  += mae(pred_vy, tgt_vy).item()
                        s_pcc_vy  += pcc(pred_vy, tgt_vy).item()
                        n_vy += 1

                    vbar.set_postfix(RMSE=f"{s_rmse/max(1,n):.6f}", F1=f"{s_f1/max(1,n):.3f}")

        depth_metrics = (
            s_rmse / max(1, n),
            s_mae  / max(1, n),
            s_pcc  / max(1, n),
            s_f1   / max(1, n),
            s_pod  / max(1, n),
            s_far  / max(1, n),
        )

        flow_metrics = {}
        if n_vx > 0:
            flow_metrics.update(
                rmse_vx = s_rmse_vx / max(1, n_vx),
                mae_vx  = s_mae_vx  / max(1, n_vx),
                pcc_vx  = s_pcc_vx  / max(1, n_vx),
            )
        if n_vy > 0:
            flow_metrics.update(
                rmse_vy = s_rmse_vy / max(1, n_vy),
                mae_vy  = s_mae_vy  / max(1, n_vy),
                pcc_vy  = s_pcc_vy  / max(1, n_vy),
            )

        return depth_metrics, flow_metrics

    # ================= 训练循环 =================
    for epoch in range(1, args.epochs + 1):
        _sync_cuda()
        epoch_t0 = time.perf_counter()

        tr = one_epoch_train(epoch)
        log = f"Epoch {epoch:03d}/{args.epochs} | TrainLoss:{tr:.6f}"

        do_val = (epoch % max(1, args.val_interval) == 0) or (epoch == args.epochs)
        flow_metrics = {}

        if do_val:
            (vr, vm, vp, vf1, vpod, vfar), flow_metrics = one_epoch_val(epoch)
            log += (f" | Val RMSE:{vr:.6f} MAE:{vm:.6f} PCC:{vp:.4f} "
                    f"F1:{vf1:.3f} POD:{vpod:.3f} FAR:{vfar:.3f}")

            # 若存在速度指标，则简要把 RMSE 打在日志里
            if flow_metrics:
                if "rmse_vx" in flow_metrics:
                    log += f" | RMSE_vx:{flow_metrics['rmse_vx']:.6f}"
                if "mae_vx" in flow_metrics:
                    log += f" | MAE_vx:{flow_metrics['mae_vx']:.6f}"
                if "pcc_vx" in flow_metrics:
                    log += f" | PCC_vx:{flow_metrics['pcc_vx']:.6f}"
                if "rmse_vy" in flow_metrics:
                    log += f" RMSE_vy:{flow_metrics['rmse_vy']:.6f}"
                if "mae_vy" in flow_metrics:
                    log += f" | MAE_vy:{flow_metrics['mae_vy']:.6f}"
                if "pcc_vy" in flow_metrics:
                    log += f" | PCC_vy:{flow_metrics['pcc_vy']:.6f}"

            # 仍然只用 depth RMSE 作为 best 模型的判断依据
            if vr < best_val - 1e-4:
                best_val = vr
                torch.save(model.state_dict(), best_path)
                best_metrics = {
                    "epoch": epoch,
                    "train_loss": float(tr),
                    "val_rmse": float(vr),
                    "val_mae": float(vm),
                    "val_pcc": float(vp),
                    "val_f1": float(vf1),
                    "val_pod": float(vpod),
                    "val_far": float(vfar),
                    "lr": float(opt.param_groups[0]["lr"])
                }
                # 把速度指标一并记录进去（如果有）
                for k, v in flow_metrics.items():
                    best_metrics[f"val_{k}"] = float(v)

                log += f"  >> Save Best (val_rmse={vr:.6f})"
                no_improve = 0
            else:
                no_improve += 1

            scheduler.step(vr)
        else:
            log += " | (skip val this epoch)"

        _sync_cuda()
        epoch_sec = time.perf_counter() - epoch_t0
        log += f" | EpochTime:{_fmt_sec(epoch_sec)}"

        print(log)

        # 基于验证 RMSE 的早停（逻辑保持不变）
        if es_patience > 0 and no_improve >= es_patience:
            print(f"[EarlyStop] Val RMSE 在最近 {es_patience} 轮内无显著改善（min_delta={min_delta:g}）。停止训练。")
            break

        # 训练阈值早停（保持原逻辑）
        if (getattr(args, 'train_stop_threshold', None) is not None) and (tr < args.train_stop_threshold):
            print(f"[EarlyStop] TrainLoss {tr:.6f} < threshold {args.train_stop_threshold:.6f}，提前停止训练。")
            break

        # 基于验证 RMSE 的早停（可选，默认关闭，传参启用）
        if es_patience > 0 and no_improve >= es_patience:
            print(f"[EarlyStop] Val RMSE 在最近 {es_patience} 轮内无显著改善（min_delta={min_delta:g}）。停止训练。")
            break

        # 训练阈值早停（可选）
        if (getattr(args, 'train_stop_threshold', None) is not None) and (tr < args.train_stop_threshold):
            print(f"[EarlyStop] TrainLoss {tr:.6f} < threshold {args.train_stop_threshold:g} ，停止训练。")
            break

    # ===== 保存期末产物 =====
    final_path  = os.path.join(run_dir, f"final_{tag}.pth")
    torch.save(model.state_dict(), final_path)
    config_path = os.path.join(run_dir, f"config_{tag}.json")
    with open(config_path, "w", encoding="utf-8") as f:
        cfg = vars(args).copy()
        cfg.update({"run_dir": run_dir, "timestamp": ts, "short_sig": short_sig, "brief": brief})
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"[Done] best:  {best_path} (val_rmse={best_val:.6f})")
    print(f"[Done] final: {final_path}")
    print(f"[Done] norm_stats: {norm_stats_path}")
    print(f"[Done] config: {config_path}")
    _sync_cuda()
    total_train_sec = time.perf_counter() - train_time_start
    print(f"[Time] Total training time: {_fmt_sec(total_train_sec)} ({total_train_sec:.2f} s)")

    # ✨ 训练结束后：打印并保存最佳模型的验证指标
    if best_metrics is not None:
        print("\n[Best Metrics]")
        print("  Epoch      :", best_metrics['epoch'])
        print("  TrainLoss  :", f"{best_metrics['train_loss']:.6f}")
        print("  Val RMSE   :", f"{best_metrics['val_rmse']:.6f}")
        print("  Val MAE    :", f"{best_metrics['val_mae']:.6f}")
        print("  Val PCC    :", f"{best_metrics['val_pcc']:.4f}")
        print("  Val F1     :", f"{best_metrics['val_f1']:.3f}")
        print("  Val POD    :", f"{best_metrics['val_pod']:.3f}")
        print("  Val FAR    :", f"{best_metrics['val_far']:.3f}")
        print("  LR at best :", f"{best_metrics['lr']:.6g}")
        # 如果有记录速度指标，则一并打印
        for key in ["rmse_vx", "mae_vx", "pcc_vx", "rmse_vy", "mae_vy", "pcc_vy"]:
            v = best_metrics.get(f"val_{key}", None)
            if v is None:
                continue
            label = key.upper()
            if "PCC" in label:
                fmt = f"{v:.4f}"
            else:
                fmt = f"{v:.6f}"
            print(f"  Val {label:8s} :", fmt)

        best_json = os.path.join(run_dir, f"best_metrics_{tag}.json")
        with open(best_json, "w", encoding="utf-8") as f:
            json.dump(best_metrics, f, indent=2, ensure_ascii=False)
        print(f"[Best Metrics] 已保存到：{best_json}")
    else:
        print("[Best Metrics] 未记录到更优的验证指标（检查 val_interval 或数据/指标计算）。")

    # ===== 训练结束后的后处理：best 模型对验证集推理并保存 =====
    if getattr(args, 'post_eval', False):
        post_evaluate_best_model(args, model, val_loader, device, H)

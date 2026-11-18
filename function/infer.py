# -*- coding: utf-8 -*-
"""Inference for centered PCA/SVD: reconstruct Y = C @ B + mu. Load the separately stored mean from mean.npy or the checkpoint. Save sanitized nonnegative depth to *_Y_pred.npy and set vx/vy to zero where depth is below the wet/dry threshold."""

from __future__ import annotations

import os
import re
import math
import time
import json
import glob
from typing import Optional, Dict, Tuple, Any

import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm

from .data import FloodDataset
from .devices import resolve_inference_device
from .model import build_model
from .sanity import check_nan_inf

def _resolve_mean_path(args, task: str) -> Optional[str]:
    """Mean-path priority: explicit mean_depth/mean_vx/mean_vy (or mu aliases), mean.npy in a task-specific basis directory, then mean.npy in a shared basis directory."""
    # 1) explicit mean path
    for name in (f"mean_{task}", f"mu_{task}", f"{task}_mean", f"{task}_mu"):
        v = getattr(args, name, None)
        if v and os.path.isfile(str(v)):
            return os.path.abspath(str(v))

    # 2) task-specific basis dir
    bdir = _resolve_basis_dir(args, task)
    if bdir:
        bdir = os.path.abspath(str(bdir))
        if os.path.isdir(bdir):
            mp = os.path.join(bdir, "mean.npy")
            if os.path.isfile(mp):
                return mp
        elif os.path.isfile(bdir) and os.path.basename(bdir).lower().endswith(".npy"):
            # If a basis.npy path is supplied, look for mean.npy in the same directory.
            mp = os.path.join(os.path.dirname(bdir), "mean.npy")
            if os.path.isfile(mp):
                return mp

    # 3) shared basis dir fallback
    for name in ("basis_dir", "basis"):
        v = getattr(args, name, None)
        if not v:
            continue
        v = os.path.abspath(str(v))
        if os.path.isdir(v):
            mp = os.path.join(v, "mean.npy")
            if os.path.isfile(mp):
                return mp
        elif os.path.isfile(v) and v.lower().endswith(".npy"):
            mp = os.path.join(os.path.dirname(v), "mean.npy")
            if os.path.isfile(mp):
                return mp

    return None

# -----------------------------
# Velocity helpers
# -----------------------------
def compute_velocity_mag_dir_np(
    vx_np: np.ndarray,
    vy_np: np.ndarray,
    depth_np: np.ndarray | None = None,
    depth_threshold: float | None = None,
    mask_value: float = np.nan,
    eps: float = 1e-12,
):
    vx_np = np.asarray(vx_np, dtype=np.float32)
    vy_np = np.asarray(vy_np, dtype=np.float32)

    v_mag = np.sqrt(vx_np * vx_np + vy_np * vy_np + eps)
    v_dir_rad = np.arctan2(vy_np, vx_np)        # [-pi, pi]
    v_dir_deg = np.degrees(v_dir_rad)           # [-180, 180]
    v_dir_deg = (v_dir_deg + 360.0) % 360.0     # [0, 360)

    if (depth_np is not None) and (depth_threshold is not None):
        depth_np = np.asarray(depth_np, dtype=np.float32)
        dry = depth_np < depth_threshold
        v_mag[dry] = 0.0
        v_dir_rad[dry] = np.nan
        if np.isnan(mask_value):
            v_dir_deg[dry] = np.nan
        else:
            v_dir_deg[dry] = mask_value

    return v_mag.astype(np.float32), v_dir_rad.astype(np.float32), v_dir_deg.astype(np.float32)


# -----------------------------
# Sanitizers
# -----------------------------
def _sanitized_nonnegative(arr: np.ndarray, decimals: int = 3) -> np.ndarray:
    """Replace NaN/Inf with zero, clamp nonnegative, and round depth or speed arrays."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    return np.round(arr, decimals).astype(np.float32)

def _sanitized_signed(arr: np.ndarray, decimals: int = 3) -> np.ndarray:
    """Replace NaN/Inf with zero and round signed vx/vy arrays without clamping."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.round(arr, decimals).astype(np.float32)


# -----------------------------
# Basis / mean loading (project-fit)
# -----------------------------
def _guess_norm_stats_path_from_ckpt(ckpt_path: str) -> str | None:
    base = os.path.basename(ckpt_path)
    m = re.match(r'^best_(.+)\.pth$', base)
    if not m:
        return None
    tag = m.group(1)
    cand = os.path.join(os.path.dirname(ckpt_path), f"norm_stats_{tag}.npz")
    return cand if os.path.isfile(cand) else None

def _load_norm_stats(norm_stats_arg: str | None, ckpt_path: str):
    path_arg = norm_stats_arg if norm_stats_arg and os.path.isfile(norm_stats_arg) else None
    path_auto = _guess_norm_stats_path_from_ckpt(ckpt_path)
    if path_auto and (path_arg is None or os.path.abspath(path_auto) != os.path.abspath(path_arg)):
        print(f"[NormStats] 自动匹配到：{path_auto}")
    path = path_auto or path_arg
    if not path:
        raise FileNotFoundError("未提供 --norm_stats 且无法从 checkpoint 推断 norm_stats_{tag}.npz")
    ns = np.load(path)
    x_min = ns["x_min"].astype(np.float32)
    x_max = ns["x_max"].astype(np.float32)
    print(f"[NormStats] 使用归一化统计：{path} | x_min.shape={x_min.shape}, x_max.shape={x_max.shape}")
    return x_min, x_max

def _detect_norm_kind(state_dict) -> str:
    for k in state_dict.keys():
        if k.endswith("num_batches_tracked"):
            return "bn"
    return "in"

def _resolve_basis_dir(args, task: str) -> Optional[str]:
    """Resolve supported basis arguments: basis_dir, task-specific basis_dir_depth/vx/vy, basis, and basis_depth/vx/vy. Accept either directories or files where applicable."""
    # task-specific
    for name in (f"basis_dir_{task}", f"basis_{task}", f"{task}_basis_dir", f"{task}_basis"):
        v = getattr(args, name, None)
        if v:
            return str(v)

    # shared
    for name in ("basis_dir", "basis"):
        v = getattr(args, name, None)
        if v:
            return str(v)

    return None

def _find_basis_files(basis_path: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Accept a basis NPY file or a directory containing the basis, mean.npy, and basis_meta.json. Return the basis path and optional mean and metadata paths."""
    basis_path = os.path.abspath(basis_path)
    if os.path.isfile(basis_path):
        # file given
        B_path = basis_path
        d = os.path.dirname(basis_path)
        mean_path = os.path.join(d, "mean.npy")
        mean_path = mean_path if os.path.isfile(mean_path) else None
        meta_path = os.path.join(d, "basis_meta.json")
        meta_path = meta_path if os.path.isfile(meta_path) else None
        return B_path, mean_path, meta_path

    if not os.path.isdir(basis_path):
        raise FileNotFoundError(f"basis path not found: {basis_path}")

    meta_path = os.path.join(basis_path, "basis_meta.json")
    meta_path = meta_path if os.path.isfile(meta_path) else None

    # Common basis filenames.
    cand = [
        "basis.npy", "B.npy",
        "basis_depth.npy", "B_depth.npy",
        "components.npy", "svd_basis.npy",
    ]
    B_path = None
    for fn in cand:
        p = os.path.join(basis_path, fn)
        if os.path.isfile(p):
            B_path = p
            break

    # Fallback: search for a [K,H]-shaped .npy array, excluding mean.npy.
    if B_path is None:
        npys = [p for p in glob.glob(os.path.join(basis_path, "*.npy"))
                if os.path.basename(p).lower() not in ("mean.npy",)]
        for p in npys:
            try:
                arr = np.load(p, mmap_mode="r")
                if arr.ndim == 2 and arr.shape[0] <= 2048 and arr.shape[1] >= 1024:
                    B_path = p
                    break
            except Exception:
                pass

    if B_path is None:
        raise FileNotFoundError(f"Cannot find basis B in dir: {basis_path}")

    mean_path = os.path.join(basis_path, "mean.npy")
    mean_path = mean_path if os.path.isfile(mean_path) else None

    return B_path, mean_path, meta_path

def _load_basis_np(basis_dir_or_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the separately stored basis B:[K,H] and mean field mu:[H]."""
    B_path, mean_path, meta_path = _find_basis_files(basis_dir_or_file)

    B = np.load(B_path).astype(np.float32)
    if B.ndim != 2:
        raise ValueError(f"Basis B must be 2D [K,H], got {B.shape} from {B_path}")

    mu = None
    if mean_path is not None:
        mu = np.load(mean_path).astype(np.float32)
        mu = mu.reshape(-1)
        if mu.shape[0] != B.shape[1]:
            raise ValueError(f"mean.npy length != H. mean={mu.shape}, H={B.shape[1]}, path={mean_path}")

    # Optionally report extra metadata without requiring it.
    if meta_path is not None:
        try:
            meta = json.load(open(meta_path, "r", encoding="utf-8"))
            center = meta.get("center", None)
            append_mean = meta.get("append_mean", None)
            if center is not None or append_mean is not None:
                print(f"[BasisMeta] {meta_path} | center={center}, append_mean={append_mean}")
        except Exception:
            pass

    if mu is None:
        mu = np.zeros((B.shape[1],), dtype=np.float32)

    return B, mu

def _extract_B_mu_from_state(state: Dict[str, torch.Tensor], keys_B: Tuple[str, ...], keys_mu: Tuple[str, ...]):
    B = None
    mu = None
    for k in keys_B:
        if k in state:
            B = state[k]
            break
    for k in keys_mu:
        if k in state:
            mu = state[k]
            break
    return B, mu

def _get_task_basis_and_mu(
    *,
    state: Dict[str, torch.Tensor],
    args,
    task: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load B/mu from the checkpoint first, then the basis directory. Use a zero mean if no mean field is available."""
    # 1) from checkpoint
    if task == "depth":
        keys_B = (
            "decoders.depth.B", "module.decoders.depth.B",
            "decoders.depth.lowrank.B", "module.decoders.depth.lowrank.B",
            "lowrank.B", "module.lowrank.B",  # Support legacy models.
        )
        keys_mu = (
            "decoders.depth.mu", "module.decoders.depth.mu",
            "decoders.depth.lowrank.mu", "module.decoders.depth.lowrank.mu",
            "decoders.depth.mean", "module.decoders.depth.mean",
            "lowrank.mu", "module.lowrank.mu", "lowrank.mean", "module.lowrank.mean",
        )
    else:
        # Accept common multi-task names such as decoders.vx.B and decoders.vx.mu.
        keys_B = (
            f"decoders.{task}.B", f"module.decoders.{task}.B",
            f"decoders.{task}.lowrank.B", f"module.decoders.{task}.lowrank.B",
        )
        keys_mu = (
            f"decoders.{task}.mu", f"module.decoders.{task}.mu",
            f"decoders.{task}.lowrank.mu", f"module.decoders.{task}.lowrank.mu",
            f"decoders.{task}.mean", f"module.decoders.{task}.mean",
        )

    B_ckpt, mu_ckpt = _extract_B_mu_from_state(state, keys_B, keys_mu)

    if B_ckpt is not None:
        B_t = B_ckpt.to(device=device, dtype=torch.float32)
        if B_t.ndim != 2:
            raise ValueError(f"[CKPT] {task} basis must be 2D, got {tuple(B_t.shape)}")
        H = int(B_t.shape[1])

        if mu_ckpt is not None:
            mu_t = mu_ckpt.to(device=device, dtype=torch.float32).view(-1)
            if mu_t.numel() != H:
                raise ValueError(f"[CKPT] {task} mu length != H. mu={mu_t.numel()}, H={H}")
            print(f"[Basis] {task}: loaded B+mu from CKPT | B={tuple(B_t.shape)} mu={tuple(mu_t.shape)}")
            return B_t, mu_t

        # If a checkpoint contains B but lacks mu, try mean.npy.
        mean_path = _resolve_mean_path(args, task)
        if mean_path is not None:
            mu_np = np.load(mean_path).astype(np.float32).reshape(-1)
            if mu_np.shape[0] != H:
                raise ValueError(f"[Mean] {task}: mean.npy length != H. mean={mu_np.shape[0]}, H={H}, path={mean_path}")
            mu_t = torch.from_numpy(mu_np).to(device=device, dtype=torch.float32)
            print(f"[Basis] {task}: loaded B from CKPT, mu from DISK | mean={mean_path}")
            return B_t, mu_t

        # Warn and fall back to zero if the mean is still unavailable.
        print(f"\033[31m[WARN]\033[0m [Basis] {task}: CKPT has B but NO mu, and mean.npy not found. "
              f"Reconstruction will use mu=0 -> likely bias/negative depth.")
        mu_t = torch.zeros((H,), device=device, dtype=torch.float32)
        return B_t, mu_t


    # 2) from basis dir
    basis_path = _resolve_basis_dir(args, task)
    if not basis_path:
        raise FileNotFoundError(f"[Basis] {task}: ckpt has no B, and no basis_dir/basis provided in args.")

    B_np, mu_np = _load_basis_np(basis_path)
    B_t = torch.from_numpy(B_np).to(device=device, dtype=torch.float32)
    mu_t = torch.from_numpy(mu_np).to(device=device, dtype=torch.float32)

    print(f"[Basis] {task}: loaded from DISK | {basis_path} | B={tuple(B_t.shape)} mu={tuple(mu_t.shape)}")
    return B_t, mu_t

def _reconstruct_block(C: torch.Tensor, B: torch.Tensor, mu: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    """
    C:  [B,T,K]
    B:  [K,H]
    mu: [H]
    cols: [S] indices into H
    return: [B,T,S] = C @ B[:,cols] + mu[cols]
    """
    # B_sub [K,S], mu_sub [S]
    B_sub = B.index_select(dim=1, index=cols)
    mu_sub = mu.index_select(dim=0, index=cols)
    # einsum -> [B,T,S]
    Y = torch.einsum("btk,ks->bts", C, B_sub)
    Y = Y + mu_sub.view(1, 1, -1)
    return Y


# -----------------------------
# Main infer
# -----------------------------
@torch.no_grad()
def batch_infer(args):
    device = resolve_inference_device(getattr(args, "device", "cpu"))
    print(f"[Infer] Device={device} | CPU threads={torch.get_num_threads()}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) load checkpoint
    try:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.checkpoint, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    # 2) norm stats
    x_min, x_max = _load_norm_stats(getattr(args, "norm_stats", None), args.checkpoint)

    # 3) build model (only for predicting C)
    norm_kind = _detect_norm_kind(state)
    args.batch_size = 1 if norm_kind == "in" else max(2, getattr(args, "batch_size", 2))
    print(f"[Infer] Detected norm kind: {norm_kind.upper()}")

    # Use the depth basis shape (K,H) to obtain H for model construction.
    B_depth, mu_depth = _get_task_basis_and_mu(state=state, args=args, task="depth", device=device)
    K_depth, H = int(B_depth.shape[0]), int(B_depth.shape[1])
    args.rank_k = K_depth

    model = build_model(args, input_channels=26, T=args.timesteps, H=H, device=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    is_multi = hasattr(model, "decoders") and hasattr(model, "forward_coeffs")

    # Load vx/vy bases when the model provides these tasks.
    B_vx = mu_vx = B_vy = mu_vy = None

    # 4) dataset
    ds = FloodDataset(args.inflow, tar_dir=None, timesteps=args.timesteps,
                      is_train=False, ref_min=x_min, ref_max=x_max)
    sids = [os.path.splitext(f)[0] for f in ds.ids]
    print(f"[Infer] Samples={len(sids)} | inflow_dir={args.inflow}")

    # 5) infer configs
    flood_thr = float(getattr(args, "flood_threshold", getattr(args, "threshold", 0.1)))
    amp_en = device.type == "cuda"
    block = getattr(args, "val_block", 0) or min(H, 20000)
    nblocks = math.ceil(H / block)

    per_t = []
    t0_all = time.perf_counter()

    for sid in tqdm(sids, desc="Samples", unit="sample"):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        # --- read x (already normalized like training) ---
        x = ds._read_excel(os.path.join(ds.in_dir, f"{sid}.xlsx"))  # [26,T]
        below0 = float((x < 0).sum()) / x.size
        above1 = float((x > 1).sum()) / x.size
        if below0 > 0.01 or above1 > 0.01:
            print(f"\033[33m[Warn-Scale] {sid}: norm<0={below0:.2%}, >1={above1:.2%}\033[0m")

        x_t = torch.from_numpy(x).unsqueeze(0).to(device)  # [1,C,T]

        # --- forward to get C ---
        with autocast(device.type, enabled=amp_en):
            if is_multi:
                C_dict = model.forward_coeffs(x_t)  # task -> [1,T,K]
                C_depth = C_dict.get("depth", None)
                if C_depth is None:
                    raise RuntimeError("[Infer] multi-task model has no 'depth' head.")

                C_vx = C_dict.get("vx", None)
                C_vy = C_dict.get("vy", None)
                has_vx = (C_vx is not None)
                has_vy = (C_vy is not None)
            else:
                C_depth = model(x_t)  # [1,T,K]
                C_vx = C_vy = None
                has_vx = has_vy = False

        C_depth = check_nan_inf(C_depth, "infer-C-depth", raise_on_nan=False, fix=True)
        Tlen = int(C_depth.shape[1])

        # load vx/vy basis only if needed
        if has_vx and (B_vx is None):
            B_vx, mu_vx = _get_task_basis_and_mu(state=state, args=args, task="vx", device=device)
        if has_vy and (B_vy is None):
            B_vy, mu_vy = _get_task_basis_and_mu(state=state, args=args, task="vy", device=device)

        # allocate outputs (numpy buffers)
        depth_pred = np.zeros((Tlen, H), dtype=np.float32)
        vx_pred = np.zeros((Tlen, H), dtype=np.float32) if has_vx else None
        vy_pred = np.zeros((Tlen, H), dtype=np.float32) if has_vy else None

        # --- block reconstruction: Y = C@B + mu ---
        for b in tqdm(range(nblocks), desc=f"{sid} blocks", leave=False, unit="blk"):
            s = b * block
            e = min(H, s + block)
            cols = torch.arange(s, e, device=device)

            with autocast(device.type, enabled=amp_en):
                # depth
                piece_d = _reconstruct_block(C_depth, B_depth, mu_depth, cols=cols)  # [1,T,S]
                piece_d = check_nan_inf(piece_d, "infer-depth-piece", raise_on_nan=False, fix=True)

                # vx/vy
                if has_vx:
                    C_vx = check_nan_inf(C_vx, "infer-C-vx", raise_on_nan=False, fix=True)
                    piece_vx = _reconstruct_block(C_vx, B_vx, mu_vx, cols=cols)
                    piece_vx = check_nan_inf(piece_vx, "infer-vx-piece", raise_on_nan=False, fix=True)
                if has_vy:
                    C_vy = check_nan_inf(C_vy, "infer-C-vy", raise_on_nan=False, fix=True)
                    piece_vy = _reconstruct_block(C_vy, B_vy, mu_vy, cols=cols)
                    piece_vy = check_nan_inf(piece_vy, "infer-vy-piece", raise_on_nan=False, fix=True)

            # write back to numpy
            depth_pred[:, s:e] = piece_d.squeeze(0).detach().cpu().numpy().astype(np.float32)
            if has_vx:
                vx_pred[:, s:e] = piece_vx.squeeze(0).detach().cpu().numpy().astype(np.float32)
            if has_vy:
                vy_pred[:, s:e] = piece_vy.squeeze(0).detach().cpu().numpy().astype(np.float32)

            # free temp
            del piece_d
            if has_vx:
                del piece_vx
            if has_vy:
                del piece_vy
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # -----------------------------
        # save depth: raw + sanitized + compat
        # -----------------------------
        stem = os.path.join(args.output_dir, f"{sid}")

        depth_raw = np.nan_to_num(depth_pred, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        depth_san = _sanitized_nonnegative(depth_raw, decimals=3)

        # np.save(stem + "_Y_pred_raw.npy", depth_raw)
        # np.save(stem + "_Y_pred_sanitized.npy", depth_san)
        # Use *_Y_pred.npy for compatibility with postviz.
        np.save(stem + "_Y_pred.npy", depth_san)

        # -----------------------------
        # enforce wet/dry consistency for velocity (use sanitized depth)
        # -----------------------------
        dry = depth_san < flood_thr

        if has_vx:
            vx_raw = np.nan_to_num(vx_pred, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            vx_raw[dry] = 0.0
            vx_out = _sanitized_signed(vx_raw, decimals=3)
            np.save(stem + "_Vx_pred.npy", vx_out)
            # np.save(stem + "_Vx_pred_raw.npy", vx_raw)

        if has_vy:
            vy_raw = np.nan_to_num(vy_pred, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            vy_raw[dry] = 0.0
            vy_out = _sanitized_signed(vy_raw, decimals=3)
            np.save(stem + "_Vy_pred.npy", vy_out)
            # np.save(stem + "_Vy_pred_raw.npy", vy_raw)

        if has_vx and has_vy:
            vmag, vdir_rad, vdir_deg = compute_velocity_mag_dir_np(
                vx_out, vy_out,
                depth_np=depth_san,
                depth_threshold=flood_thr,
                mask_value=np.nan
            )
            vmag_out = _sanitized_nonnegative(vmag, decimals=3)
            np.save(stem + "_Vmag_pred.npy", vmag_out)
            # np.save(stem + "_Vmag_pred_raw.npy", vmag.astype(np.float32))
            # Enable the following exports if flow direction is needed.
            # np.save(stem + "_Vdir_deg_pred.npy", vdir_deg)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        per_t.append(time.perf_counter() - t0)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t1_all = time.perf_counter()

    print("\n[Infer][Time] -------------------------------")
    print(f"[Infer][Time] Total wall time      : {t1_all - t0_all:.3f} s")
    print(f"[Infer][Time] Avg per sample       : {np.mean(per_t):.3f} s over {len(per_t)} samples")
    print("[Infer][Time] --------------------------------\n")


def batch_infer_cpu(args):
    """Run inference on CPU without modifying the caller's argument namespace."""
    import copy
    cpu_args = copy.copy(args)
    cpu_args.device = "cpu"
    return batch_infer(cpu_args)

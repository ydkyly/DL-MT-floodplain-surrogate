# -*- coding: utf-8 -*-
import os, json, numpy as np, h5py
from typing import List, Optional
from .paths import DEPTH_PATH

def _find_label_sids(label_dir: str) -> List[str]:
    """Find sample IDs from filenames or subdirectory names in the label directory."""
    exts = ('.hdf', '.h5', '.hdf5', '.npy')
    bases = set()
    for name in os.listdir(label_dir):
        p = os.path.join(label_dir, name)
        base, ext = os.path.splitext(name)
        if os.path.isfile(p) and ext.lower() in exts:
            bases.add(base)
        elif os.path.isdir(p):
            bases.add(name)
    try:
        return sorted(bases, key=lambda s: int(s))
    except Exception:
        return sorted(bases)

def _read_field_TxH(label_dir: str,
                    sid: str,
                    timesteps: int,
                    data_path: Optional[str] = None,
                    dtype=np.float32) -> np.ndarray:
    """Read a [T,H] array from HDF, a single NPY, or per-time-step NPY files. label_dir contains labels; sid identifies a sample; timesteps selects the first T rows; data_path selects the HDF dataset, defaulting to DEPTH_PATH; dtype controls output precision. Supported layouts are <sid>.hdf/.h5/.hdf5, <sid>.npy, or <sid>/0.npy,1.npy,... ."""
    data_path = data_path or DEPTH_PATH

    # Case 1: HDF input.
    for ext in ('.hdf', '.h5', '.hdf5'):
        h5p = os.path.join(label_dir, f"{sid}{ext}")
        if os.path.isfile(h5p):
            with h5py.File(h5p, "r") as h5:
                if data_path not in h5:
                    raise KeyError(f"HDF 文件 {h5p} 中找不到数据集路径：{data_path}")
                arr = h5[data_path][:timesteps].astype(dtype)  # [T,H]
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Case 2: a single .npy file.
    npy = os.path.join(label_dir, f"{sid}.npy")
    if os.path.isfile(npy):
        arr = np.load(npy).astype(dtype)
        return arr[:timesteps]

    # Case 3: one .npy file per time step in a subdirectory.
    d = os.path.join(label_dir, sid)
    if os.path.isdir(d):
        files = sorted(
            [f for f in os.listdir(d) if f.endswith('.npy')],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        if not files:
            raise FileNotFoundError(f"{d} 目录下未找到任何 .npy 文件")
        vecs = [np.load(os.path.join(d, f)).astype(dtype) for f in files[:timesteps]]
        return np.stack(vecs, axis=0)

    raise FileNotFoundError(f"{sid} 无对应标签(.hdf/.npy/子目录)")

def _read_depth_TxH(label_dir: str, sid: str, timesteps: int, dtype=np.float32) -> np.ndarray:
    """Backward-compatible wrapper around the generic label reader."""
    return _read_field_TxH(label_dir, sid, timesteps, data_path=DEPTH_PATH, dtype=dtype)

def build_basis(args):
    """Construct a spatial basis B of shape [K,H] using incremental PCA on the [N_floods*T,H] matrix. Reads paths.DEPTH_PATH by default; use --data_path to construct separate vx or vy bases. Variables are not combined automatically."""
    try:
        from sklearn.decomposition import IncrementalPCA
    except Exception as e:
        raise RuntimeError("需要安装 scikit-learn：pip install scikit-learn") from e

    label_dir = args.label_dir
    assert os.path.isdir(label_dir), f"标签目录不存在：{label_dir}"
    sids = _find_label_sids(label_dir)
    assert sids, f"{label_dir} 中未找到 .hdf/.npy/子目录 标签"

    data_path = getattr(args, "data_path", None) or DEPTH_PATH
    print(f"[Basis] 使用 data_path = {data_path}")

    # Detect array shape.
    sample0 = _read_field_TxH(label_dir, sids[0], args.timesteps,
                              data_path=data_path, dtype=np.float32)
    if sample0.ndim != 2:
        raise ValueError(f"期望 [T,H]，但样本 {sids[0]} 得到 {sample0.shape}")
    H = sample0.shape[1]
    print(f"[Basis] H={H}, timesteps={args.timesteps}, n_components={args.n_components}, chunk_rows={args.chunk_rows}")

    ipca = IncrementalPCA(n_components=args.n_components, batch_size=args.chunk_rows)

    buf = []
    rows = 0

    def flush_buf(need: int) -> np.ndarray:
        nonlocal buf
        chunk_list = []
        remain = need
        while remain > 0 and buf:
            a = buf[0]
            if a.shape[0] <= remain:
                chunk_list.append(a)
                buf.pop(0)
                remain -= a.shape[0]
            else:
                chunk_list.append(a[:remain])
                buf[0] = a[remain:]
                remain = 0
        return np.concatenate(chunk_list, axis=0)

    # Read each flood event and fit incrementally.
    for sid in sids:
        Y = _read_field_TxH(label_dir, sid, args.timesteps,
                            data_path=data_path, dtype=np.float32)
        if Y.shape[1] != H:
            raise ValueError(f"{sid} 的 H 不一致：{Y.shape[1]} vs {H}")
        buf.append(Y)
        rows += Y.shape[0]
        while sum(x.shape[0] for x in buf) >= args.chunk_rows:
            X = flush_buf(args.chunk_rows)  # [chunk_rows, H]
            ipca.partial_fit(X)
            print(f"[Basis-fit] rows_fit += {X.shape[0]}")
    if buf:
        X = np.concatenate(buf, axis=0)
        ipca.partial_fit(X)
        print(f"[Basis-fit] rows_fit += {X.shape[0]} (last)")

    # Extract principal components and summary statistics.
    B = ipca.components_.astype(np.float32)                    # [K,H]
    mu = ipca.mean_.astype(np.float32)                         # [H]
    evr = ipca.explained_variance_ratio_.astype(np.float32)    # [K]
    cum = evr.cumsum()
    k95 = int((cum >= 0.95).argmax()) + 1
    k99 = int((cum >= 0.99).argmax()) + 1
    print(f"[Basis] EVR cum: 95%→K={k95}, 99%→K={k99}, last_cum={cum[-1]:.4f}")

    # Save outputs.
    os.makedirs(args.out_dir, exist_ok=True)
    if args.append_mean:
        B = np.vstack([B, mu[None, :]]).astype(np.float32)
        print(f"[Basis] append_mean=True，输出 B 形状为 {B.shape}。训练时 --rank_k 应设为同样行数。")
    else:
        np.save(os.path.join(args.out_dir, "mean.npy"), mu)

    np.save(os.path.join(args.out_dir, "basis.npy"), B)
    np.save(os.path.join(args.out_dir, "evr.npy"), evr)
    with open(os.path.join(args.out_dir, "basis_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "H": int(H),
            "timesteps": int(args.timesteps),
            "n_components": int(args.n_components),
            "append_mean": bool(args.append_mean),
            "data_path": str(data_path),
            "evr_cumsum_last": float(cum[-1]),
            "K95": int(k95),
            "K99": int(k99)
        }, f, ensure_ascii=False, indent=2)
    print(f"[Basis] Saved to: {args.out_dir} (basis.npy / evr.npy / basis_meta.json"
          f"{' / mean.npy' if not args.append_mean else ''})")

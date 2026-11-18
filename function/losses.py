# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

def rmse(pred, tgt):
    return torch.sqrt(F.mse_loss(pred, tgt))

def mae(pred, tgt):
    return F.l1_loss(pred, tgt)

def pcc(pred, tgt, eps: float = 1e-6):
    # Pearson correlation averaged over the batch.
    B = pred.shape[0]
    pred_f = pred.view(B, -1)
    tgt_f  = tgt.view(B, -1)
    pred_f = pred_f - pred_f.mean(dim=1, keepdim=True)
    tgt_f  = tgt_f  - tgt_f.mean(dim=1, keepdim=True)
    num = (pred_f * tgt_f).sum(dim=1)
    den = torch.sqrt((pred_f.pow(2).sum(dim=1) + eps) * (tgt_f.pow(2).sum(dim=1) + eps))
    corr = num / (den + eps)
    return corr.mean()

# Depth-weighted Huber loss with clipping and normalisation.
def _huber(x, delta: float = 0.1):
    ax = x.abs()
    return torch.where(ax <= delta, 0.5 * ax * ax / delta, ax - 0.5 * delta)

def depth_weighted_loss(pred, gt, gamma: float = 1.0,
                        use_huber: bool = True, delta: float = 0.1,
                        clip_q: float = 0.999):
    """Depth-weighted reconstruction loss for pred/gt:[B,T,S], with S equal to H or a column block. gamma controls the depth exponent; use_huber selects Huber instead of L1; delta is in metres; clip_q sets the weight clipping quantile, for example 0.999."""
    dep = gt.clamp_min(0.0)
    w = dep.pow(gamma)  # Assign larger weights to greater depths.
    if clip_q is not None and 0.9 <= clip_q < 1.0:
        # Clip extreme weights within each sample to limit the influence of very deep cells.
        with torch.no_grad():
            # Flatten to [B, -1] to compute the quantile.
            q = torch.quantile(w.view(w.shape[0], -1), clip_q, dim=1, keepdim=True)
            q = q.view(-1, 1, 1)
        w = torch.minimum(w, q)
    # Normalise weights to stabilise the overall gradient scale.
    w = w / (w.mean(dim=(1,2), keepdim=True) + 1e-6)

    diff = pred - gt
    if use_huber:
        l = _huber(diff, delta=delta)
    else:
        l = diff.abs()
    return (w * l).mean()

# Torch and the Huber and depth-weighted helpers are defined above.
# Dry-region penalty helper.

def dry_region_penalty(pred, gt, threshold: float = 0.1,
                       use_huber: bool = True, delta: float = 0.05, power: float = 1.0):
    """Penalise positive predicted depths in reference-dry cells (gt<threshold) to suppress false alarms. Accept [B,T,S] or [B,T,H]. Use the evaluation wet/dry threshold for consistency. use_huber/delta control Huber loss; otherwise use abs(pred)**power."""
    with torch.no_grad():
        dry_mask = (gt < threshold).to(pred.dtype)  # One indicates a dry cell.
    # Penalise only positive predicted depth.
    pos_pred = torch.clamp(pred, min=0.0)
    if use_huber:
        l = _huber(pos_pred, delta=delta)
    else:
        l = pos_pred.abs().pow(power)
    return (dry_mask * l).mean()

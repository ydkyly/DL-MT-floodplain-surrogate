# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

def rmse(pred, tgt):
    return torch.sqrt(F.mse_loss(pred, tgt))

def mae(pred, tgt):
    return F.l1_loss(pred, tgt)

def pcc(pred, tgt, eps: float = 1e-6):
    # 皮尔逊相关：按 batch 求平均
    B = pred.shape[0]
    pred_f = pred.view(B, -1)
    tgt_f  = tgt.view(B, -1)
    pred_f = pred_f - pred_f.mean(dim=1, keepdim=True)
    tgt_f  = tgt_f  - tgt_f.mean(dim=1, keepdim=True)
    num = (pred_f * tgt_f).sum(dim=1)
    den = torch.sqrt((pred_f.pow(2).sum(dim=1) + eps) * (tgt_f.pow(2).sum(dim=1) + eps))
    corr = num / (den + eps)
    return corr.mean()

# --------- 新增：Huber + 深度加权（截断 + 归一） ----------
def _huber(x, delta: float = 0.1):
    ax = x.abs()
    return torch.where(ax <= delta, 0.5 * ax * ax / delta, ax - 0.5 * delta)

def depth_weighted_loss(pred, gt, gamma: float = 1.0,
                        use_huber: bool = True, delta: float = 0.1,
                        clip_q: float = 0.999):
    """
    pred, gt: [B, T, S]（S=H 或列块大小）
    gamma : 深度权重指数
    use_huber: True 用 Huber, False 用 L1
    delta : Huber 阈值（米）
    clip_q: 权重分位截断，例如 0.999
    """
    dep = gt.clamp_min(0.0)
    w = dep.pow(gamma)  # 深度越大权越大
    if clip_q is not None and 0.9 <= clip_q < 1.0:
        # 对每个 batch 截断极端权重，防止极深值主导
        with torch.no_grad():
            # 展平到 [B, -1] 求分位
            q = torch.quantile(w.view(w.shape[0], -1), clip_q, dim=1, keepdim=True)
            q = q.view(-1, 1, 1)
        w = torch.minimum(w, q)
    # 归一化：避免整体梯度被放大/缩小
    w = w / (w.mean(dim=(1,2), keepdim=True) + 1e-6)

    diff = pred - gt
    if use_huber:
        l = _huber(diff, delta=delta)
    else:
        l = diff.abs()
    return (w * l).mean()

# 在文件顶部已有 import Torch 与 _huber / depth_weighted_loss 定义，保持不动
# 追加以下函数到文件末尾（或 depth_weighted_loss 后面）：

def dry_region_penalty(pred, gt, threshold: float = 0.1,
                       use_huber: bool = True, delta: float = 0.05, power: float = 1.0):
    """
    在干区 (gt < threshold) 对预测的正深度进行惩罚，以抑制虚警（降低 FAR）。
    pred, gt: [B,T,S] 或 [B,T,H]
    threshold: 干/湿判别阈值（与评估阈值一致更合理）
    use_huber/delta: 是否用 Huber；否则用 |pred|^power
    """
    with torch.no_grad():
        dry_mask = (gt < threshold).to(pred.dtype)  # 1=干区
    # 只惩罚正深度（<0 不应出现）
    pos_pred = torch.clamp(pred, min=0.0)
    if use_huber:
        l = _huber(pos_pred, delta=delta)
    else:
        l = pos_pred.abs().pow(power)
    return (dry_mask * l).mean()

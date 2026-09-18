# -*- coding: utf-8 -*-
import torch

def make_epoch_chunks(H: int, S: int, shuffle: bool = True, device=None):
    """一个 epoch 内把 [0..H-1] 切成若干块（最后一块<=S），保证一次 epoch 全覆盖。"""
    if S >= H:
        idx = torch.randperm(H, device=device) if shuffle else torch.arange(H, device=device)
        return [idx]
    idx = torch.randperm(H, device=device) if shuffle else torch.arange(H, device=device)
    return list(idx.split(S))

class StepCycler:
    """跨训练步循环覆盖列；连续 ceil(H/S) 次 next() 覆盖全 H。"""
    def __init__(self, H: int, S: int, shuffle: bool = True, device=None):
        self.H, self.S, self.shuffle, self.device = H, S, shuffle, device
        self.perm = torch.randperm(H, device=device) if shuffle else torch.arange(H, device=device)
        self.ptr = 0

    def next(self) -> torch.Tensor:
        s, e = self.ptr, min(self.ptr + self.S, self.H)
        cols = self.perm[s:e]
        self.ptr = e
        if self.ptr >= self.H:
            self.perm = torch.randperm(self.H, device=self.device) if self.shuffle else torch.arange(self.H, device=self.device)
            self.ptr = 0
        return cols

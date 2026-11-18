# -*- coding: utf-8 -*-
import os, json, numpy as np, torch, torch.nn as nn
from typing import Optional

def conv3x3_1d(c_in, c_out, stride=1, groups=1, dilation=1):
    return nn.Conv1d(c_in, c_out, kernel_size=3, stride=stride, padding=dilation,
                     groups=groups, bias=False, dilation=dilation)

def conv1x1_1d(c_in, c_out, stride=1):
    return nn.Conv1d(c_in, c_out, kernel_size=1, stride=stride, bias=False)

class SafeInstanceNorm1d(nn.InstanceNorm1d):
    """Skip normalisation when the time dimension is one, preventing training errors for batch=1 and L=1."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) <= 1:
            return x
        return super().forward(x)

class BasicBlock1D(nn.Module):
    expansion=1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock1D仅支持groups=1且base_width=64")
        if dilation > 1:
            raise NotImplementedError("BasicBlock1D不支持 dilation > 1")
        self.conv1 = conv3x3_1d(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_1d(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        if not (isinstance(self.bn1, nn.InstanceNorm1d) and out.size(-1)==1):
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if not (isinstance(self.bn2, nn.InstanceNorm1d) and out.size(-1)==1):
            out = self.bn2(out)
        if self.downsample is not None: identity = self.downsample(x)
        out = self.relu(out + identity)
        return out

class Bottleneck1D(nn.Module):
    expansion=4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm1d
        width = int(planes*(base_width/64.0))*groups
        self.conv1 = conv1x1_1d(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3_1d(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1_1d(width, planes*self.expansion)
        self.bn3 = norm_layer(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        if not (isinstance(self.bn1, nn.InstanceNorm1d) and out.size(-1)==1):
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if not (isinstance(self.bn2, nn.InstanceNorm1d) and out.size(-1)==1):
            out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        if not (isinstance(self.bn3, nn.InstanceNorm1d) and out.size(-1)==1):
            out = self.bn3(out)
        if self.downsample is not None: identity = self.downsample(x)
        out = self.relu(out + identity)
        return out

class ResNet1D(nn.Module):
    def __init__(self, block, layers, num_classes=1000, input_channels=1,
                 output_channels=1, zero_init_residual=False,
                 groups=1, width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
        super().__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm1d
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        assert len(replace_stride_with_dilation)==3
        self._norm_layer = norm_layer
        self.output_channels = output_channels
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck1D):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock1D):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        prev_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                conv1x1_1d(self.inplanes, planes*block.expansion, stride),
                norm_layer(planes*block.expansion),
            )
        layers_list = [block(self.inplanes, planes, stride, downsample,
                             self.groups, self.base_width, prev_dilation, norm_layer)]
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers_list.append(block(self.inplanes, planes, groups=self.groups,
                                     base_width=self.base_width, dilation=self.dilation,
                                     norm_layer=norm_layer))
        return nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# TCN backbone.

class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size = int(chomp_size)
    def forward(self, x): return x[..., :-self.chomp_size] if self.chomp_size>0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation, dropout=0.2):
        super().__init__()
        pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=dilation)
        self.chomp1 = Chomp1d(pad)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=dilation)
        self.chomp2 = Chomp1d(pad)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch!=out_ch else None
        self.reset()
    def reset(self):
        nn.init.xavier_uniform_(self.conv1.weight); nn.init.xavier_uniform_(self.conv2.weight)
        if self.down is not None: nn.init.xavier_uniform_(self.down.weight)
    def forward(self, x):
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        res = x if self.down is None else self.down(x)
        return out + res

class TemporalConvNet(nn.Module):
    def __init__(self, in_ch, channels_list, k=3, dropout=0.2):
        super().__init__()
        layers=[]; C=in_ch
        for i, Co in enumerate(channels_list):
            layers.append(TemporalBlock(C, Co, k, dilation=2**i, dropout=dropout)); C=Co
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class TCNBackbone(nn.Module):
    """Return z:[B,feat_dim] for LowRankDecoder to transform into [B,T,K]."""
    def __init__(self, input_channels:int, feat_dim:int=512, k: int=3, dropout:float=0.2,
                 channels_list=(64,64,64)):
        super().__init__()
        self.tcn = TemporalConvNet(input_channels, channels_list, k, dropout)
        self.proj = nn.Linear(channels_list[-1], feat_dim)
    def forward(self, x):
        # x:[B,C,L] -> [B,C',L] -> last-time-step features -> [B,feat_dim].
        h = self.tcn(x)
        last = h[:, :, -1]
        z = self.proj(last)
        return z

# Low-rank decoding.

class LowRankDecoder(nn.Module):
    """Predict coefficients C:[B,T,K] and reconstruct Y:[B,T,H] using fixed or trainable B:[K,H], optionally adding a mean field mu."""
    def __init__(self, feat_dim:int, out_T:int, basis_B:torch.Tensor,
                 mu: Optional[torch.Tensor]=None, add_mu: bool=True):
        super().__init__()
        self.out_T = out_T
        self.rank_k, self.H = basis_B.shape
        self.fc = nn.Linear(feat_dim, out_T*self.rank_k)
        self.register_buffer("B", basis_B)
        if mu is not None:
            assert mu.ndim==1 and mu.shape[0]==self.H, f"mu 形状需为 [H]，got {tuple(mu.shape)}"
            self.register_buffer("mu", mu)
        else:
            self.register_buffer("mu", None)
        self.add_mu = bool(add_mu)

    def forward_coeff(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z).view(z.size(0), self.out_T, self.rank_k)

    def set_mu(self, mu: torch.Tensor):
        assert mu.ndim==1 and mu.shape[0]==self.H
        self.mu = mu

    def reconstruct(self, C: torch.Tensor, cols: Optional[torch.Tensor]=None) -> torch.Tensor:
        if cols is None:
            Y = torch.einsum('btk,kh->bth', C, self.B)
            if self.add_mu and (self.mu is not None):
                Y = Y + self.mu.view(1,1,-1)
            return Y
        else:
            B_part = self.B[:, cols]
            Y = torch.einsum('btk,kh->bth', C, B_part)
            if self.add_mu and (self.mu is not None):
                Y = Y + self.mu[cols].view(1,1,-1)
            return Y

class ResNet1DHead(nn.Module):
    """Apply global average pooling to produce z:[B,512*exp], then predict C:[B,T,K] with the head."""
    def __init__(self, backbone: nn.Module, lowrank_head: LowRankDecoder):
        super().__init__()
        self.backbone = backbone
        self.lowrank  = lowrank_head

    def forward(self, x):
        z = self.backbone(x)
        C = self.lowrank.forward_coeff(z)
        return C

class MultiTaskModel(nn.Module):
    """Multi-task model with a shared temporal backbone such as ResNet1D or TCN and separate LowRankDecoder instances for depth, vx, and vy, stored in the decoders dictionary."""

    def __init__(self, backbone: nn.Module, decoders: dict[str, "LowRankDecoder"]):
        super().__init__()
        from torch.nn import ModuleDict
        self.backbone = backbone
        self.decoders = ModuleDict(decoders)

    def forward_coeffs(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return task-specific coefficient tensors [B,T,K_task], typically before blockwise reconstruction during training."""
        z = self.backbone(x)  # [B, C_feat, T]
        out: dict[str, torch.Tensor] = {}
        for name, dec in self.decoders.items():
            out[name] = dec.forward_coeff(z)
        return out

    def reconstruct(self,
                    C_dict: dict[str, torch.Tensor],
                    cols: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Reconstruct all task fields for selected cols, or all H columns when cols=None. Return {task:[B,T,S]}."""
        out: dict[str, torch.Tensor] = {}
        for name, C in C_dict.items():
            dec = self.decoders[name]
            out[name] = dec.reconstruct(C, cols=cols)
        return out

    def forward(self, x: torch.Tensor,
                cols: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Convenience end-to-end interface returning reconstructed fields for all tasks."""
        z = self.backbone(x)
        out: dict[str, torch.Tensor] = {}
        for name, dec in self.decoders.items():
            C = dec.forward_coeff(z)
            out[name] = dec.reconstruct(C, cols=cols)
        return out

# Model construction for training and inference.

def _load_B_and_mu_from_path(basis_path: Optional[str], H:int, device: torch.device):
    B_np = None; mu_np = None; append_mean = False
    meta = None
    if basis_path and os.path.isfile(basis_path):
        B_np = np.load(basis_path).astype(np.float32)
        bd = os.path.dirname(basis_path)
        meta_path = os.path.join(bd, "basis_meta.json")
        if os.path.isfile(meta_path):
            try:
                meta = json.load(open(meta_path, "r", encoding="utf-8"))
                append_mean = bool(meta.get("append_mean", False))
            except Exception:
                pass
        # Try mean.npy when the mean has not been appended to the basis.
        if not append_mean:
            mean_path = os.path.join(bd, "mean.npy")
            if os.path.isfile(mean_path):
                mu_np = np.load(mean_path).astype(np.float32)
    if B_np is None:
        # Initialise B randomly.
        rng = np.random.default_rng(42)
        rand = rng.standard_normal((H if H<1 else 1,))  # dummy to avoid error
        # Without a basis path, the caller must provide (K,H); leave the basis empty here.
        pass
    return B_np, mu_np, append_mean, meta

def _parse_tasks(tasks_str: str | None) -> list[str]:
    """Parse --tasks into internal task names: depth/d selects depth; flow selects vx/vy; depth,flow or depth,vx,vy selects all three. Derive angle during inference rather than treating it as a training task."""
    if not tasks_str:
        return ["depth"]
    raw = [t.strip().lower() for t in tasks_str.split(",") if t.strip()]
    tasks: list[str] = []
    for t in raw:
        if t in ("depth", "d"):
            if "depth" not in tasks:
                tasks.append("depth")
        elif t == "flow":
            for u in ("vx", "vy"):
                if u not in tasks:
                    tasks.append(u)
        elif t in ("vx", "vy"):
            if t not in tasks:
                tasks.append(t)
        else:
            # Derive quantities such as angle from vx/vy during inference.
            continue
    if not tasks:
        tasks = ["depth"]
    # Place depth first for convenient loss and metric calculation.
    if "depth" in tasks:
        tasks = ["depth"] + [t for t in tasks if t != "depth"]
    return tasks

def build_model(args, input_channels: int, T: int, H: int, device: torch.device):
    """Build a depth-only model if tasks is absent or contains only depth; otherwise build MultiTaskModel with separate low-rank decoders and bases for depth/vx/vy."""
    # Parse the task list; depth-only is the fallback.
    tasks = _parse_tasks(getattr(args, "tasks", None))

    # Build the backbone.
    backbone_kind = getattr(args, 'backbone', None)
    if backbone_kind in (None, '', 'resnet_basic', 'resnet_bottleneck'):
        # Support the legacy args.block argument.
        use_basic = (backbone_kind == 'resnet_basic') or (getattr(args, 'block', 'basic') == 'basic')
        block, layers = (BasicBlock1D, [2, 2, 2, 2]) if use_basic else (Bottleneck1D, [3, 4, 6, 3])
        # Use SafeInstanceNorm1d for batch_size==1 to handle L=1.
        if getattr(args, "batch_size", 1) != 1:
            norm_layer = nn.BatchNorm1d
        else:
            norm_layer = lambda C: SafeInstanceNorm1d(C, affine=True)
        backbone = ResNet1D(
            block,
            layers,
            num_classes=512 * block.expansion,
            input_channels=input_channels,
            output_channels=1,
            norm_layer=norm_layer
        ).to(device)
        feat_dim = 512 * block.expansion
        # Replace fc with Identity and let the decoder produce [B,T,K].
        backbone.fc = nn.Identity()
    elif backbone_kind == 'tcn':
        # TCN output z has shape [B,512].
        backbone = TCNBackbone(
            input_channels=input_channels,
            feat_dim=512,
            k=int(getattr(args, 'tcn_kernel', 3)),
            dropout=float(getattr(args, 'tcn_dropout', 0.2)),
            channels_list=tuple(getattr(args, 'tcn_channels', (64, 64, 64)))
        ).to(device)
        feat_dim = 512
    else:
        raise ValueError(f"未知 backbone: {backbone_kind}")

    # Depth-only path, compatible with legacy training scripts.
    if tasks == ["depth"]:
        rank_k = int(getattr(args, "rank_k", 256))
        # Load B and mu, preferring the basis directory.
        B_np, mu_np, append_mean, _ = _load_B_and_mu_from_path(
            getattr(args, 'basis', None), H, device
        )
        if B_np is None:
            # Use a random placeholder when the checkpoint will supply the basis through state_dict.
            rng = np.random.default_rng(42)
            rand = rng.standard_normal((rank_k, H), dtype=np.float32) / max(1, np.sqrt(H))
            row_norm = np.linalg.norm(rand, axis=1, keepdims=True) + 1e-8
            B_np = rand / row_norm
        assert B_np.shape == (rank_k, H), f"basis 需 {rank_k}×{H}, got {B_np.shape}"

        basis_B = torch.from_numpy(B_np).to(device)
        mu_t = torch.from_numpy(mu_np).to(device) if (mu_np is not None) else None
        # With append_mean=True, mu is already a basis row and must not be added separately.
        add_mu = False if append_mean else True
        head = LowRankDecoder(
            feat_dim=feat_dim,
            out_T=T,
            basis_B=basis_B,
            mu=mu_t,
            add_mu=add_mu
        )
        return ResNet1DHead(backbone, head).to(device)

    # Multi-task path: create a separate LowRankDecoder for each task.
    decoders: dict[str, LowRankDecoder] = {}
    for task in tasks:
        # Select depth_basis, vx_basis, or vy_basis; depth_basis takes precedence over basis.
        if task == "depth":
            basis_path = getattr(args, "depth_basis", None) or getattr(args, "basis", None)
        elif task == "vx":
            basis_path = getattr(args, "vx_basis", None)
        elif task == "vy":
            basis_path = getattr(args, "vy_basis", None)
        else:
            # Skip unsupported task names.
            continue

        # Use task-specific ranks if supplied; otherwise use rank_k.
        rank_k = int(getattr(args, f"rank_k_{task}", getattr(args, "rank_k", 256)))

        B_np, mu_np, append_mean, _ = _load_B_and_mu_from_path(basis_path, H, device)
        if B_np is None:
            # Use a placeholder basis when loading from a checkpoint.
            rng = np.random.default_rng(42)
            rand = rng.standard_normal((rank_k, H), dtype=np.float32) / max(1, np.sqrt(H))
            row_norm = np.linalg.norm(rand, axis=1, keepdims=True) + 1e-8
            B_np = rand / row_norm
        else:
            # If file rank differs from rank_k, use the file rank and report a warning.
            if B_np.shape[1] != H:
                raise ValueError(f"{task} 的 basis 形状需 (?, {H})，got {B_np.shape}")
            if B_np.shape[0] != rank_k:
                # Synchronise rank_k to avoid assertion failures.
                rank_k = B_np.shape[0]
        assert B_np.shape == (rank_k, H), f"{task} 的 basis 需 {rank_k}×{H}，got {B_np.shape}"

        basis_B = torch.from_numpy(B_np).to(device)
        mu_t = torch.from_numpy(mu_np).to(device) if (mu_np is not None) else None
        add_mu = False if append_mean else True

        decoders[task] = LowRankDecoder(
            feat_dim=feat_dim,
            out_T=T,
            basis_B=basis_B,
            mu=mu_t,
            add_mu=add_mu
        )

    if "depth" not in decoders:
        raise ValueError("多任务模式下必须至少包含 'depth' 任务，以保证水深训练和指标计算。")

    model = MultiTaskModel(backbone, decoders)
    return model.to(device)


def build_model_for_infer(args, H:int, device: torch.device) -> ResNet1DHead:
    # Use the training backbone for inference; prefer instance normalisation if unknown.
    backbone_kind = getattr(args, 'backbone', 'resnet_basic')
    if backbone_kind in ('resnet_basic','resnet_bottleneck'):
        use_basic = (backbone_kind=='resnet_basic')
        block, layers = (BasicBlock1D, [2,2,2,2]) if use_basic else (Bottleneck1D,[3,4,6,3])
        norm_layer = lambda C: nn.InstanceNorm1d(C, affine=True)
        backbone = ResNet1D(block, layers, num_classes=512*block.expansion,
                            input_channels=26, output_channels=1,
                            norm_layer=norm_layer).to(device)
        feat_dim = 512*block.expansion
        backbone.fc = nn.Identity()
    elif backbone_kind == 'tcn':
        backbone = TCNBackbone(input_channels=26, feat_dim=512,
                               k=int(getattr(args,'tcn_kernel',3)),
                               dropout=float(getattr(args,'tcn_dropout',0.2)),
                               channels_list=tuple(getattr(args,'tcn_channels', (64,64,64)))).to(device)
        feat_dim = 512
    else:
        raise ValueError(f"未知 backbone: {backbone_kind}")

    rank_k = args.rank_k
    B_np, mu_np, append_mean, _ = _load_B_and_mu_from_path(getattr(args,'basis',None), H, device)
    if B_np is None:
        basis_B = torch.zeros((rank_k, H), dtype=torch.float32, device=device)  # Placeholder overwritten by state_dict.
    else:
        assert B_np.shape==(rank_k,H)
        basis_B = torch.from_numpy(B_np).to(device)
    mu_t = torch.from_numpy(mu_np).to(device) if (mu_np is not None) else None
    add_mu = False if append_mean else True
    head = LowRankDecoder(feat_dim=feat_dim, out_T=getattr(args,'timesteps',24),
                          basis_B=basis_B, mu=mu_t, add_mu=add_mu)
    return ResNet1DHead(backbone, head).to(device)

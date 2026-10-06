"""Self-contained RSML and ablation detectors for DeepfakeBench.

Generated only from the user-supplied efnb4mamba_short(2).py, model(4).py,
CNN(4).py, MBlock(4).py and DWGCA(4).py.  No project-local model imports are
required.  The only removed objects are iwpm1/iwpm2 and norm1/norm2: they are
constructed in the uploaded Encoder/Decoder but never read by either
forward method.  Removing them avoids false IWP.py and hard-coded .cuda()
dependencies without changing logits or any active loss.
"""

from __future__ import annotations

import collections
import logging
import math
import re
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Callable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from timm.models.layers import DropPath
from torch.utils import model_zoo

# from detectors import DETECTOR
# from metrics.base_metrics_class import calculate_metrics_for_train
# from .base_detector import AbstractDetector

logger = logging.getLogger(__name__)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SS2D_Block(nn.Module):

    def __init__(self, d_model, d_state=16, d_conv=3, expand=2.0, dt_rank='auto', dt_min=0.001, dt_max=0.1, dt_init='random', dt_scale=1.0, dt_init_floor=0.0001, dropout=0.0, conv_bias=True, bias=False, device=None, dtype=None, ss2d_type=None):
        super().__init__()
        if ss2d_type == None:
            raise 'ss2d_type undefine'
        self.ss2d_type = ss2d_type
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == 'auto' else dt_rank
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(in_channels=self.d_inner, out_channels=self.d_inner, groups=self.d_inner, bias=conv_bias, kernel_size=d_conv, padding=(d_conv - 1) // 2)
        self.act = nn.SiLU()
        self.x_proj = [nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False) for _ in range(ss2d_type)]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj
        self.dt_projs = [self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor) for _ in range(ss2d_type)]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.ss2d_type, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=self.ss2d_type, merge=True)
        self.selective_scan = selective_scan_fn
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        if self.ss2d_type != 1:
            b1_stride = 2
            stride = 1
            d_conv = 3
            self.conv2d1_b1 = nn.Conv2d(in_channels=d_model, out_channels=d_model, groups=d_model, kernel_size=7, stride=b1_stride, padding=3)
            self.upsample = nn.ConvTranspose2d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=2, padding=1, groups=d_model, bias=conv_bias, output_padding=1)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init='random', dt_min=0.001, dt_max=0.1, dt_init_floor=0.0001):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** (-0.5) * dt_scale
        if dt_init == 'constant':
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == 'random':
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), 'n -> d n', d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, 'd n -> r d n', r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, 'n1 -> r n1', r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = self.ss2d_type
        if K == 3:
            x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
            xs = xs[:, 1:, :, :]
        else:
            xs = x
        x_dbl = torch.einsum('b k d l, k c d -> b k c l', xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum('b k r l, k d r -> b k d l', dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(xs, dts, As, Bs, Cs, Ds, z=None, delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        if K == 3:
            inv_y = torch.flip(out_y[:, 1:3], dims=[-1]).view(B, 2, -1, L)
            wh_y = torch.transpose(out_y[:, 0].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            invwh_y = torch.transpose(inv_y[:, 0].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            seq = (inv_y[:, 0], wh_y, invwh_y)
        else:
            seq = out_y[:, 0]
        return seq

    def forward(self, x: torch.Tensor, **kwargs):
        if self.ss2d_type == 3:
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.conv2d1_b1(x).permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        if self.ss2d_type == 3:
            y1, y2, y3 = self.forward_core(x)
            y = y1 + y2 + y3
        elif self.ss2d_type == 1:
            y = self.forward_core(x)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.ss2d_type == 3:
            out = out.permute(0, 3, 1, 2).contiguous()
            out = self.upsample(out).permute(0, 2, 3, 1).contiguous()
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class MDMamba(nn.Module):

    def __init__(self, hidden_dim: int=0, drop_path: float=0, norm_layer: Callable[..., torch.nn.Module]=partial(nn.LayerNorm, eps=1e-06), attn_drop_rate: float=0, d_state: int=16, expand: float=2.0, is_light_sr: bool=False, **kwargs):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention_1 = SS2D_Block(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate, ss2d_type=1, **kwargs)
        self.self_attention_3 = SS2D_Block(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate, ss2d_type=3, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = Mlp(hidden_dim, hidden_dim, hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input):
        B, C, H, W = input.shape
        input = input.permute(0, 2, 3, 1).contiguous()
        x = self.ln_1(input)
        x_1 = self.self_attention_1(x)
        x_3 = self.self_attention_3(x)
        x = x_1 + x_3
        x = input * self.skip_scale + self.drop_path(x)
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x))
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x

class BasicConv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, act=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False
        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if act:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = self.main(x)
        return x

class SCM(nn.Module):

    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(BasicConv(3, out_plane // 4, kernel_size=3, stride=1, act=True, bias=False, norm=True), BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, act=True, bias=False, norm=True), BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, act=True, bias=False, norm=True), BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, act=False), nn.InstanceNorm2d(out_plane, affine=True))

    def forward(self, x):
        x = self.main(x)
        return x

class FAM(nn.Module):

    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel * 2, channel, kernel_size=3, stride=1, act=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))

class ResidualDepthBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.depthwise = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False)
        self.pointwise = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.groups == m.out_channels:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.depthwise(out)
        out = self.pointwise(out)
        out = self.bn2(out)
        x = F.gelu(out + shortcut)
        return x

def build_wavelet_kernels(device=None, dtype=torch.float32):
    """
    返回 2x2 的四个 2D 分析核：LL, LH, HL, HH
    对于 Db1 (Haar)：h0=[1/sqrt2, 1/sqrt2], h1=[-1/sqrt2, 1/sqrt2]
    2D 核是外积：h_row^T * h_col
    """
    s = 1.0 / math.sqrt(2.0)
    h0 = torch.tensor([s, s], dtype=dtype, device=device)
    h1 = torch.tensor([-s, s], dtype=dtype, device=device)
    LL = torch.ger(h0, h0)
    LH = torch.ger(h0, h1)
    HL = torch.ger(h1, h0)
    HH = torch.ger(h1, h1)
    filt = torch.stack([LL, LH, HL, HH], dim=0).unsqueeze(1)
    return filt

class SE(nn.Module):
    """
    轻量通道注意力（Squeeze-Excitation风格）
    - 使用 GAP 降维 -> 两层1x1线性 -> Sigmoid
    - 用于与 Softmax 通道归一化互补：SE 强调“绝对重要性”，Softmax 强调“相对分配”
    """

    def __init__(self, channels, r=8):
        super().__init__()
        hidden = max(channels // r, 4)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x):
        z = F.adaptive_avg_pool2d(x, 1)
        z = F.relu(self.fc1(z), inplace=True)
        z = torch.sigmoid(self.fc2(z))
        return z

class WGAM(nn.Module):

    def __init__(self, channels, use_fc=True, se_ratio=8, gam=0.5):
        super().__init__()
        self.channels = channels
        self.use_fc = use_fc
        self.theta = nn.Parameter(torch.zeros(3, channels, 1, 1))
        self.dir_gate = nn.Parameter(torch.zeros(3, channels, 1, 1))
        if use_fc:
            self.fc = nn.Linear(channels, channels, bias=True)
        self.sub_redistribute = nn.Parameter(torch.zeros(channels, 3))
        self.se = SE(channels, r=se_ratio)
        self.gamma = nn.Parameter(torch.tensor(gam, dtype=torch.float32))
        filt = build_wavelet_kernels()
        self.register_buffer('w_analysis', filt)
        self.register_buffer('w_synthesis', filt)

    def dwt(self, x):
        """
        x: (B,C,H,W)
        返回：LH, HL, HH, LL
        """
        B, C, H, W = x.shape
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
        weight = self.w_analysis.repeat(C, 1, 1, 1)
        y = F.conv2d(x, weight=weight, bias=None, stride=2, padding=0, groups=C)
        y = y.view(B, C, 4, y.size(-2), y.size(-1)).contiguous()
        LL = y[:, :, 0]
        LH = y[:, :, 1]
        HL = y[:, :, 2]
        HH = y[:, :, 3]
        return (LH, HL, HH, LL)

    def idwt(self, LH, HL, HH, LL):
        """
        逆变换：将四个子带重建为 (B,C,H,W)
        """
        B, C, h, w = LL.shape
        y = torch.stack([LL, LH, HL, HH], dim=2).view(B, 4 * C, h, w)
        weight = self.w_synthesis.repeat(C, 1, 1, 1)
        x_rec = F.conv_transpose2d(y, weight=weight, bias=None, stride=2, padding=0, groups=C)
        return x_rec

    @staticmethod
    def soft_threshold(x, thr):
        return torch.sign(x) * F.relu(torch.abs(x) - thr)

    def forward(self, x):
        """
        x: [B,C,H,W]
        输出：与输入同形状 [B,C,H,W]
        """
        B, C, H, W = x.shape
        LH, HL, HH, LL = self.dwt(x)
        LH_hat = LH
        HL_hat = HL
        HH_hat = HH
        g = torch.sigmoid(self.dir_gate)
        LH_hat = LH_hat * g[0].unsqueeze(0)
        HL_hat = HL_hat * g[1].unsqueeze(0)
        HH_hat = HH_hat * g[2].unsqueeze(0)
        h = LH_hat.size(-2)
        w = LH_hat.size(-1)
        stack3 = torch.stack([LH_hat, HL_hat, HH_hat], dim=2)
        attn_band = F.softmax(stack3.abs().mean(dim=1, keepdim=True), dim=2)
        H_mix = (stack3 * attn_band).sum(dim=2)
        w_redis = F.softmax(self.sub_redistribute, dim=1)
        w_redis = w_redis.view(1, C, 3, 1, 1)
        H_mix_exp = H_mix.unsqueeze(2)
        H_redist = H_mix_exp * w_redis
        LH2 = H_redist[:, :, 0]
        HL2 = H_redist[:, :, 1]
        HH2 = H_redist[:, :, 2]
        X_re = self.idwt(LH2, HL2, HH2, LL)
        gap_vec = F.adaptive_avg_pool2d(X_re, 1).view(B, C)
        if self.use_fc:
            gap_vec = self.fc(gap_vec)
        w_softmax = F.softmax(gap_vec, dim=1).view(B, C, 1, 1)
        w_se = self.se(X_re)
        out = x * w_softmax + self.gamma * (X_re * w_se)
        return out

class LayerNorm2d(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class Encoder(nn.Module):

    def __init__(self, base_channel=3, num_blocks=[4, 4, 4]):
        super(Encoder, self).__init__()
        self.num_blocks_stage2 = num_blocks[1]
        self.num_blocks_stage3 = num_blocks[2]
        self.encoder_sampling = nn.ModuleList([BasicConv(base_channel, base_channel * 2, kernel_size=3, act=True, stride=2, bias=False, norm=True), BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, act=True, stride=2, bias=False, norm=True)])
        self.cnn_stem = nn.Sequential(BasicConv(3, base_channel, kernel_size=3, act=True, stride=1, bias=False, norm=True), BasicConv(base_channel, base_channel, kernel_size=3, act=True, stride=1, bias=False, norm=True))
        self.ecnn_stage = nn.Sequential(*[ResidualDepthBlock(base_channel, base_channel) for i in range(num_blocks[0])])
        self.MBlock1 = nn.ModuleList([nn.Sequential(MDMamba(base_channel * 2), WGAM(channels=base_channel * 2, use_fc=True, se_ratio=3)) for i in range(num_blocks[1] // 2)])
        self.MBlock2 = nn.ModuleList([nn.Sequential(MDMamba(base_channel * 4), WGAM(channels=base_channel * 4, use_fc=True, se_ratio=3)) for i in range(num_blocks[2] // 2)])
        self.SCM1 = SCM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 4)
        self.FAM1 = FAM(base_channel * 2)
        self.FAM2 = FAM(base_channel * 4)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        x_2 = self.SCM1(x_2)
        x_4 = self.SCM2(x_4)
        x = self.cnn_stem(x)
        feature_list = []
        for i in range(len(self.ecnn_stage)):
            x = self.ecnn_stage[i](x)
        feature_list.append(x)
        x = self.encoder_sampling[0](x)
        x = self.FAM1(x, x_2)
        for i in range(len(self.MBlock1)):
            x = self.MBlock1[i](x)
        feature_list.append(x)
        x = self.encoder_sampling[1](x)
        x = self.FAM2(x, x_4)
        for i in range(len(self.MBlock2)):
            x = self.MBlock2[i](x)
        feature_list.append(x)
        return feature_list

class Decoder(nn.Module):

    def __init__(self, base_channel, num_blocks=[4, 4, 4], num_heads=[3, 6], mlp_ratio=2.66, window_size=8, deep_supv=True):
        super(Decoder, self).__init__()
        self.window_size = window_size
        self.deep_supv = deep_supv
        if deep_supv:
            self.decoder_out_conv = nn.ModuleList([BasicConv(base_channel * 4, 3, kernel_size=3, act=False, stride=1, bias=False), BasicConv(base_channel * 2, 3, kernel_size=3, act=False, stride=1, bias=False)])
        self.output = BasicConv(base_channel, 3, kernel_size=3, act=False, stride=1, bias=False, norm=True)
        self.num_blocks_stage2 = num_blocks[1]
        self.num_blocks_stage3 = num_blocks[2]
        self.decoder_sampling = nn.ModuleList([BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, act=True, stride=2, transpose=True, bias=False, norm=True), BasicConv(base_channel * 2, base_channel, kernel_size=4, act=True, stride=2, transpose=True, bias=False, norm=True)])
        self.decoder_compress_channel = nn.ModuleList([BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, act=True, stride=1), BasicConv(base_channel * 2, base_channel, kernel_size=1, act=True, stride=1)])
        self.dmix_stage2 = nn.ModuleList([nn.Sequential(MDMamba(base_channel * 4), WGAM(channels=base_channel * 4, use_fc=True, se_ratio=3)) for i in range(num_blocks[2] // 2)])
        self.dmix_stage1 = nn.ModuleList([nn.Sequential(MDMamba(base_channel * 2), WGAM(channels=base_channel * 2, use_fc=True, se_ratio=3)) for i in range(num_blocks[1] // 2)])
        self.dcnn_stage = nn.Sequential(*[ResidualDepthBlock(base_channel, base_channel) for i in range(num_blocks[0])])

    def forward(self, x0, enc_feature_list):
        if self.deep_supv:
            outputlist = []
            x_2 = F.interpolate(x0, scale_factor=0.5)
            x_4 = F.interpolate(x_2, scale_factor=0.5)
        f1, f2, f3 = enc_feature_list
        x = f3
        for i in range(len(self.dmix_stage2)):
            x = self.dmix_stage2[i](x)
        if self.deep_supv:
            outputlist.append(self.decoder_out_conv[0](x) + x_4)
        x = self.decoder_sampling[0](x)
        x = torch.cat([x, f2], dim=1)
        x = self.decoder_compress_channel[0](x)
        for i in range(len(self.dmix_stage1)):
            x = self.dmix_stage1[i](x)
        if self.deep_supv:
            outputlist.append(self.decoder_out_conv[1](x) + x_2)
        x = self.decoder_sampling[1](x)
        x = torch.cat([x, f1], dim=1)
        x = self.decoder_compress_channel[1](x)
        for i in range(len(self.dcnn_stage)):
            x = self.dcnn_stage[i](x)
        x = self.output(x)
        if self.deep_supv:
            outputlist.append(x + x0)
            return outputlist
        return x + x0

GlobalParams = collections.namedtuple('GlobalParams', ['batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'num_classes', 'width_coefficient', 'depth_coefficient', 'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size'])

BlockArgs = collections.namedtuple('BlockArgs', ['kernel_size', 'num_repeat', 'input_filters', 'output_filters', 'expand_ratio', 'id_skip', 'stride', 'se_ratio'])

GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

efficientnet_params = {'efficientnet-b0': (1.0, 1.0, 224, 0.2), 'efficientnet-b1': (1.0, 1.1, 240, 0.2), 'efficientnet-b2': (1.1, 1.2, 260, 0.3), 'efficientnet-b3': (1.2, 1.4, 300, 0.3), 'efficientnet-b4': (1.4, 1.8, 380, 0.4), 'efficientnet-b5': (1.6, 2.2, 456, 0.4), 'efficientnet-b6': (1.8, 2.6, 528, 0.5), 'efficientnet-b7': (2.0, 3.1, 600, 0.5), 'efficientnet-b8': (2.2, 3.6, 672, 0.5), 'efficientnet-l2': (4.3, 5.3, 800, 0.5)}

blocks_args_str = ['r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25', 'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25', 'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25', 'r1_k3_s11_e6_i192_o320_se0.25']

url_map = {'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth', 'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth', 'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth', 'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth', 'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth', 'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth', 'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth', 'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth'}

url_map_advprop = {'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b0-b64d5a18.pth', 'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b1-0f3ce85a.pth', 'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b2-6e9d97e5.pth', 'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b3-cdd7c0f4.pth', 'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth', 'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b5-86493f6b.pth', 'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b6-ac80338e.pth', 'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b7-4652b6dd.pth', 'efficientnet-b8': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b8-22a8fe65.pth'}

class BlockDecoder:

    @staticmethod
    def _decode_block_string(block_string):
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split('(\\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value
        assert 's' in options and len(options['s']) == 1 or (len(options['s']) == 2 and options['s'][0] == options['s'][1])
        return BlockArgs(kernel_size=int(options['k']), num_repeat=int(options['r']), input_filters=int(options['i']), output_filters=int(options['o']), expand_ratio=int(options['e']), id_skip='noskip' not in block_string, se_ratio=float(options['se']) if 'se' in options else None, stride=[int(options['s'][0])])

    @staticmethod
    def decode(string_list):
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

class SwishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):

    def forward(self, x):
        return SwishImplementation.apply(x)

class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)

def round_filters(filters, global_params):
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, global_params):
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))

def drop_connect(inputs, p, training):
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output

class Conv2dStaticSamePadding(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = (math.ceil(ih / sh), math.ceil(iw / sw))
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = None

    def forward(self, x):
        if self.static_padding:
            x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

class MBConvBlock(nn.Module):

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = self._block_args.se_ratio is not None and 0 < self._block_args.se_ratio <= 1
        self.id_skip = block_args.id_skip
        Conv2d = partial(Conv2dStaticSamePadding, image_size=global_params.image_size)
        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(in_channels=oup, out_channels=oup, groups=oup, kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x
        x = self._bn2(self._project_conv(x))
        if self.id_skip and self._block_args.stride == 1 and (self._block_args.input_filters == self._block_args.output_filters):
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x

    def set_swish(self, memory_efficient=True):
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()

class REM(nn.Module):

    def __init__(self, in_channels=3):
        super().__init__()
        self.res_norm = nn.InstanceNorm2d(in_channels, affine=True)
        self.spatial_small = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False)
        self.spatial_fuse = nn.Sequential(nn.Sigmoid())
        squeezed = in_channels
        self.channel_gate = nn.Sequential(nn.Linear(in_channels * 2, squeezed), nn.ReLU(), nn.Linear(squeezed, in_channels), nn.Sigmoid())
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.guidance_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, x_res):
        B, C, H, W = x_res.shape
        x_res_norm = self.res_norm(x_res)
        sp_small = self.spatial_small(x_res_norm)
        spatial_w = self.spatial_fuse(sp_small)
        res_mean = x_res_norm.mean(dim=[2, 3])
        res_std = x_res_norm.std(dim=[2, 3])
        ch_feat = torch.cat([res_mean, res_std], dim=1)
        channel_w = self.channel_gate(ch_feat)
        channel_w = channel_w.unsqueeze(-1).unsqueeze(-1)
        alpha = torch.sigmoid(self.alpha)
        guidance = alpha * spatial_w + (1 - alpha) * channel_w
        scale = torch.abs(self.guidance_scale)
        x_guided = x * (1.0 + scale * guidance)
        return x_guided

def wavelet_loss(real_rec, real_x, fake_rec, fake_x, margin=0.3):

    def hf_energy(rec, orig):
        err = rec - orig
        x00 = err[:, :, 0::2, 0::2]
        x01 = err[:, :, 0::2, 1::2]
        x10 = err[:, :, 1::2, 0::2]
        x11 = err[:, :, 1::2, 1::2]
        LH = ((x00 - x01 + x10 - x11) / 4).abs()
        HL = ((x00 + x01 - x10 - x11) / 4).abs()
        HH = ((x00 - x01 - x10 + x11) / 4).abs()
        hf = torch.stack([LH, HL, HH], dim=0).max(dim=0)[0]
        return hf.mean(dim=[1, 2, 3])
    real_score = hf_energy(real_rec, real_x)
    fake_score = hf_energy(fake_rec, fake_x)
    n = min(real_score.shape[0], fake_score.shape[0])
    real_score = real_score[:n]
    fake_score = fake_score[:n]
    return F.relu(real_score.mean() - fake_score.mean() + margin)

class Fourier_loss(nn.Module):

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def _high_freq_stat(self, x):
        x = x.float()
        if x.numel() == 0:
            return torch.tensor(0.0, device=x.device)
        fft_result = torch.fft.fft2(x)
        mag = torch.abs(fft_result)
        mag_shift = torch.fft.fftshift(mag, dim=[-2, -1])
        B, C, H, W = mag_shift.shape
        h_cut, w_cut = (H // 4, W // 4)
        mag_high = mag_shift.clone()
        mag_high[:, :, H // 2 - h_cut:H // 2 + h_cut, W // 2 - w_cut:W // 2 + w_cut] = 0.0
        mean = mag_high.mean(dim=[-2, -1])
        std = mag_high.std(dim=[-2, -1], unbiased=False)
        result = torch.cat([mean, std], dim=1)
        return result

    def forward(self, anchor, real, fake):
        if anchor.shape[0] == 0 or real.shape[0] == 0 or fake.shape[0] == 0:
            return torch.tensor(0.0, device=anchor.device, requires_grad=True)
        a_stat = self._high_freq_stat(anchor)
        r_stat = self._high_freq_stat(real)
        f_stat_all = self._high_freq_stat(fake)
        f_stat = f_stat_all.mean(dim=0, keepdim=True)
        f_stat = f_stat.expand(a_stat.shape[0], -1)
        d_ar = F.l1_loss(a_stat, r_stat, reduction='none').mean(dim=1)
        d_af = F.l1_loss(a_stat, f_stat, reduction='none').mean(dim=1)
        loss = F.relu(d_ar - d_af + self.margin).mean()
        return loss

class RSML(nn.Module):

    def __init__(self, model_cfg):
        super().__init__()
        model_name = model_cfg['VARIANT']
        self.check_model_name_is_valid(model_name)
        blocks_args = BlockDecoder.decode(blocks_args_str)
        w, d, s, p = efficientnet_params[model_name]
        global_params = GlobalParams(batch_norm_momentum=0.99, batch_norm_epsilon=0.001, dropout_rate=p, drop_connect_rate=0.2, num_classes=2, width_coefficient=w, depth_coefficient=d, depth_divisor=8, min_depth=None, image_size=s)
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self.escape = ''
        self._global_params = global_params
        self._blocks_args = blocks_args
        Conv2d = partial(Conv2dStaticSamePadding, image_size=global_params.image_size)
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon
        in_channels = 3
        out_channels = round_filters(32, self._global_params)
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._blocks = nn.ModuleList([])
        self.stage_map = []
        stage_count = 0
        for block_args in self._blocks_args:
            block_args = block_args._replace(input_filters=round_filters(block_args.input_filters, self._global_params), output_filters=round_filters(block_args.output_filters, self._global_params), num_repeat=round_repeats(block_args.num_repeat, self._global_params))
            stage_count += 1
            self.stage_map += [''] * (block_args.num_repeat - 1)
            self.stage_map.append('b%s' % stage_count)
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))
        in_channels = block_args.output_filters
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()
        if model_cfg['PRETRAINED']:
            self.load_pretrained_weights(model_name, advprop=True)
        self.encoder = Encoder(base_channel=3)
        self.decoder = Decoder(base_channel=3)
        self.residual_guidance = REM(in_channels=3)
        self.align_half = nn.Conv2d(6, 3, 1)
        self.align_quar = nn.Conv2d(12, 3, 1)

    def set_swish(self, memory_efficient=True):
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, x, layers, start_idx=0, end_idx=None):
        assert start_idx >= 0
        if start_idx == 0:
            x = self._swish(self._bn0(self._conv_stem(x)))
            layers['b0'] = x
        if end_idx is None or end_idx > len(self._blocks):
            end_idx = len(self._blocks)
        assert start_idx < end_idx
        for idx in range(start_idx, end_idx):
            block = self._blocks[idx]
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            stage = self.stage_map[idx]
            if stage:
                layers[stage] = x
                if stage == self.escape:
                    return None
        if end_idx == len(self._blocks):
            x = self._bn1(self._conv_head(x))
            x = self._swish(x)
        return x

    def load_pretrained_weights(self, model_name, advprop=False):
        url_map_ = url_map_advprop if advprop else url_map
        state_dict = model_zoo.load_url(url_map_[model_name])
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        res = self.load_state_dict(state_dict, strict=False)
        assert set(res.missing_keys) == set(['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'
        print('Loaded pretrained weights for {}'.format(model_name))

    def check_model_name_is_valid(self, model_name):
        valid_models = ['efficientnet-b' + str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))

    def forward(self, samples, y):
        layers = {}
        x = samples
        bs = x.size(0)
        x = self.extract_features(x, layers)
        layers['final'] = x
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        logits_orig = self._fc(x)
        if torch.is_grad_enabled():
            layers = {}
            x = samples
            features = self.encoder(x)
            defeatures = self.decoder(x, features)
            x_res = x - defeatures[2]
            x_guided = self.residual_guidance(x, x_res)
            x = self.extract_features(x_guided, layers)
            layers['final'] = x
            x = self._avg_pooling(x)
            x = x.view(bs, -1)
            x = self._dropout(x)
            logits_aug = self._fc(x)
            loss_aug = F.cross_entropy(logits_aug, y)
            recons_loss = self.exp_recons_loss4(defeatures, (features, y), samples)
            # p_orig = F.softmax(logits_orig.detach(), dim=1)
            # p_aug = F.log_softmax(logits_aug, dim=1)
            # consistency_loss = F.kl_div(p_aug, p_orig, reduction='batchmean')
            return (logits_orig, 6 * loss_aug + recons_loss)
        return (logits_orig, torch.tensor(0.0, device=x.device))

    def exp_recons_loss4(self, defeatures, x_input, samples):
        x, y = x_input
        if not torch.is_grad_enabled():
            return torch.tensor(0.0, device=y.device)
        y = y.long()
        real_index = torch.where(1 - y)[0]
        fake_index = torch.where(y)[0]
        loss = torch.tensor(0.0, device=y.device, dtype=torch.float32)
        if real_index.numel() == 0 or fake_index.numel() < 1:
            return loss
        real_rec_full = defeatures[2][real_index]
        # real_rec_half = defeatures[1][real_index]
        # real_rec_quar = defeatures[0][real_index]
        real_enc_full = x[0][real_index]
        real_enc_full = x[0][real_index]
        # real_enc_half = x[1][real_index]
        # real_enc_quar = x[2][real_index]
        l1_full = F.l1_loss(real_rec_full, real_enc_full)
        # l1_half = F.l1_loss(real_rec_half, self.align_half(real_enc_half))
        # l1_quar = F.l1_loss(real_rec_quar, self.align_quar(real_enc_quar))
        loss += l1_full
        real_x = real_enc_full
        fake_x = samples[fake_index]
        real_rec = real_rec_full
        loss += 0.15 * wavelet_loss(real_rec, real_x, defeatures[2][fake_index], fake_x, margin=0.15)
        fcr = Fourier_loss(margin=0.25)
        loss += 0.1 * fcr(real_rec, real_x, fake_x)
        return loss

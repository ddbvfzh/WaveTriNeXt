"""ConvNeXt-T + parallel improvedPMAM + six early-stage Haar WTConv layers.

The wavelet algorithm is adapted from BGU-CS-VIL/WTConv at the pinned commit
below, under its MIT license (see third_party/wtconv/LICENSE). This specialization
uses fixed db1 filters, stride one, persistent buffers, and pretrained base DW
convolutions. It does not require PyWavelets or download pretrained weights.
"""
from __future__ import annotations

import math
import re

import torch
from torch import nn
from torch.nn import functional as F

from .convnext_pmam import build_model as _build_pmam


MODEL_ID = "convnext_tiny_improved_pmam_parallel_s34_wtconv_s12"
WT_ADDED_PARAMETERS = 346_464
ADDED_PARAMETERS = 2_560_644
WT_SOURCE_COMMIT = "4a84da2218924b3973de2235d641f2ecf969c346"
WT_CONFIG = {
    "stages": [1, 2], "blocks_per_stage": [3, 3], "wt_type": "db1",
    "wt_levels": 2, "kernel_size": 7, "stride": 1, "base_bias": True,
    "wavelet_bias": False, "base_scale_init": 1.0,
    "wavelet_scale_init": 0.1, "filters": "persistent_buffers",
    "reference_commit": WT_SOURCE_COMMIT,
}


def create_haar_filters(channels: int, dtype=torch.float32):
    """Match the upstream db1 coefficient casting, outer products, and ordering.

Keep coefficient multiplication in the requested dtype: float32 outer products
of sqrt(1/2) differ slightly from filling the final kernels with exactly 0.5.
"""
    value = math.sqrt(0.5)
    low = torch.tensor([value, value], dtype=dtype)
    high = torch.tensor([value, -value], dtype=dtype)
    filters = torch.stack([
        low.unsqueeze(0) * low.unsqueeze(1),
        low.unsqueeze(0) * high.unsqueeze(1),
        high.unsqueeze(0) * low.unsqueeze(1),
        high.unsqueeze(0) * high.unsqueeze(1),
    ])[:, None].repeat(channels, 1, 1, 1)
    # For Haar, the reversed decomposition coefficients and reconstruction
    # coefficients used in the upstream cross-correlation code are identical.
    return filters, filters.clone()


def wavelet_2d_transform(x, filters):
    """Analyze an even-sized NCHW tensor into N,C,4,H/2,W/2 subbands."""
    batch, channels, height, width = x.shape
    if height % 2 or width % 2:
        raise ValueError("DWT input must have even spatial dimensions; pad before analysis")
    padding = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    transformed = F.conv2d(x, filters, stride=2, groups=channels, padding=padding)
    return transformed.reshape(batch, channels, 4, height // 2, width // 2)


def inverse_2d_wavelet_transform(x, filters):
    """Reconstruct N,C,4,H,W subbands into an even-sized NCHW tensor."""
    batch, channels, bands, height, width = x.shape
    if bands != 4:
        raise ValueError("IWT requires exactly four Haar subbands")
    padding = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    return F.conv_transpose2d(x.reshape(batch, channels * 4, height, width), filters,
                             stride=2, groups=channels, padding=padding)


class ScaleModule(nn.Module):
    """Learned per-channel multiplication, excluded from weight decay."""

    def __init__(self, channels: int, init_scale: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1) * init_scale)

    def forward(self, x):
        return self.weight * x


class WTConv2d(nn.Module):
    """Upstream multilevel WTConv, specialized to Haar and spatial stride one."""

    def __init__(self, in_channels, out_channels=None, kernel_size=7, stride=1,
                 bias=True, wt_levels=2, wt_type="db1"):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        if type(in_channels) is not int or in_channels < 1 or in_channels != out_channels:
            raise ValueError("WTConv requires equal positive input and output channel counts")
        if type(kernel_size) is not int or kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("WTConv kernel_size must be a positive odd integer")
        if type(wt_levels) is not int or wt_levels < 1:
            raise ValueError("WTConv wt_levels must be a positive integer")
        if stride != 1 or wt_type not in ("db1", "haar"):
            raise ValueError("This experiment supports only Haar/db1 with stride=1")
        self.in_channels = self.out_channels = in_channels
        self.wt_levels, self.wt_type = wt_levels, "db1"
        self.kernel_size, self.stride, self.dilation = kernel_size, stride, 1
        analysis, reconstruction = create_haar_filters(in_channels)
        self.register_buffer("wt_filter", analysis, persistent=True)
        self.register_buffer("iwt_filter", reconstruction, persistent=True)
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding="same",
                                   groups=in_channels, bias=bias)
        self.base_scale = ScaleModule(in_channels, 1.0)
        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(4 * in_channels, 4 * in_channels, kernel_size, padding="same",
                      groups=4 * in_channels, bias=False) for _ in range(wt_levels)
        ])
        self.wavelet_scale = nn.ModuleList([ScaleModule(4 * in_channels, 0.1)
                                            for _ in range(wt_levels)])

    def forward(self, x):
        processed_low, processed_high, original_shapes = [], [], []
        raw_low = x
        for level in range(self.wt_levels):
            original_shapes.append(raw_low.shape)
            height, width = raw_low.shape[-2:]
            if height % 2 or width % 2:
                raw_low = F.pad(raw_low, (0, width % 2, 0, height % 2))
            bands = wavelet_2d_transform(raw_low, self.wt_filter)
            # Recursion depends on the unprocessed LL, not the learned LL.
            raw_low = bands[:, :, 0]
            batch, channels, _, half_h, half_w = bands.shape
            processed = self.wavelet_scale[level](self.wavelet_convs[level](
                bands.reshape(batch, channels * 4, half_h, half_w))).reshape(bands.shape)
            processed_low.append(processed[:, :, 0])
            processed_high.append(processed[:, :, 1:4])
        reconstructed = 0
        for level in range(self.wt_levels - 1, -1, -1):
            low = processed_low[level] + reconstructed
            bands = torch.cat([low.unsqueeze(2), processed_high[level]], dim=2)
            reconstructed = inverse_2d_wavelet_transform(bands, self.iwt_filter)
            height, width = original_shapes[level][-2:]
            reconstructed = reconstructed[:, :, :height, :width]
        return self.base_scale(self.base_conv(x)) + reconstructed


def baseline_parameter_name(name: str) -> str:
    """Map a copied base DW tensor name to its corresponding A+a tensor name."""
    return re.sub(r"^(features\.(?:1|3)\.[0-2]\.block\.0)\.base_conv\.(weight|bias)$",
                  r"\1.\2", name)


def build_model(num_classes, weight_path=None):
    """Initialize exactly A+a first; add six independent WTConv layers afterward."""
    model = _build_pmam(num_classes, weight_path)
    with torch.random.fork_rng(devices=[]):
        for stage_index, channels in [(1, 96), (3, 192)]:
            if len(model.features[stage_index]) != 3:
                raise ValueError("Expected exactly three ConvNeXt blocks in each early stage")
            for block in model.features[stage_index]:
                original = block.block[0]
                if not (isinstance(original, nn.Conv2d) and original.in_channels == channels
                        and original.out_channels == channels and original.groups == channels
                        and original.kernel_size == (7, 7) and original.stride == (1, 1)
                        and original.bias is not None):
                    raise ValueError("Expected the original pretrained 7x7 depthwise convolution")
                replacement = WTConv2d(channels, channels, kernel_size=7, wt_levels=2)
                replacement.base_conv.load_state_dict(original.state_dict(), strict=True)
                block.block[0] = replacement
    return model.float()


def optimizer_for(model):
    """Keep imported base DW parameters in the backbone role; new WT parts are new."""
    a3_ids = {id(p) for p in model.a3.parameters()}
    a4_ids = {id(p) for p in model.a4.parameters()}
    if a3_ids & a4_ids:
        raise ValueError("a3 and a4 parameters must not be shared")
    attention_ids = a3_ids | a4_ids
    head_ids = {id(p) for p in model.classifier[2].parameters()}
    wavelet_ids, imported_base_ids = set(), set()
    for _name, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, WTConv2d):
            base_ids = {id(p) for p in module.base_conv.parameters()}
            new_ids = {id(p) for p in module.parameters()} - base_ids
            if (wavelet_ids | imported_base_ids) & (new_ids | base_ids):
                raise ValueError("WTConv modules must not share parameters")
            wavelet_ids.update(new_ids)
            imported_base_ids.update(base_ids)
    roles = [attention_ids, head_ids, wavelet_ids, imported_base_ids]
    if any(left & right for index, left in enumerate(roles) for right in roles[index + 1:]):
        raise ValueError("Attention, head, wavelet and imported base parameters must be independent")
    no_decay_ids = set()
    for module in model.modules():
        norm = isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.GroupNorm))
        for name, parameter in module.named_parameters(recurse=False):
            if norm or isinstance(module, ScaleModule) or name in ("bias", "layer_scale") or parameter.ndim <= 1:
                no_decay_ids.add(id(parameter))
    rates = {"backbone": 1e-4, "attention": 1e-3, "wavelet": 1e-3, "head": 1e-3}
    groups, assigned = {}, set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        identity = id(parameter)
        if identity in assigned:
            raise ValueError(f"Parameter assigned more than once: {name}")
        assigned.add(identity)
        role = ("attention" if identity in attention_ids else "head" if identity in head_ids
                else "wavelet" if identity in wavelet_ids else "backbone")
        decay = identity not in no_decay_ids
        key = (role, decay)
        if key not in groups:
            groups[key] = {"params": [], "param_names": [], "lr": rates[role],
                           "initial_lr": rates[role], "weight_decay": 0.05 if decay else 0.0,
                           "role": role}
        groups[key]["params"].append(parameter)
        groups[key]["param_names"].append(name)
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if assigned != expected:
        raise ValueError("Optimizer groups do not cover exactly all trainable parameters")
    return torch.optim.AdamW(list(groups.values()), betas=(0.9, 0.999), eps=1e-8, foreach=False)

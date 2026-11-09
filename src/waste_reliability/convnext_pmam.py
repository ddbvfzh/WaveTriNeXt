"""ConvNeXt-T with two complete, parallel ImprovedPMAM stage-output modules.

Channel and spatial weights are both computed directly from the same PreConv
feature. The original ConvNeXt parameter names and all eighteen blocks remain
intact; only a3 and a4 are new. No pretrained weights are downloaded here.
"""
from __future__ import annotations

import torch
from torch import nn

from .closed_set_baseline import build_model as _build_baseline


MODEL_ID = "convnext_tiny_improved_pmam_parallel_s34"
ADDED_PARAMETERS = 2_214_180


class ParallelPMAMDualPath(nn.Module):
    """Parallel channel/spatial weight generation with multiplicative fusion."""

    def __init__(self, channels: int):
        super().__init__()
        self.avg_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.max_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.spatial_conv = nn.Conv2d(2, 1, 3, padding=1, bias=False)

    def channel_weights(self, u):
        avg = self.avg_conv(u.mean((2, 3), keepdim=True))
        maximum = self.max_conv(torch.amax(u, (2, 3), keepdim=True))
        return (avg + maximum).sigmoid()

    def spatial_weights(self, u):
        pooled = torch.cat([u.mean(1, keepdim=True), u.max(1, keepdim=True).values], dim=1)
        return self.spatial_conv(pooled).sigmoid()

    def forward(self, u):
        channel = self.channel_weights(u)
        spatial = self.spatial_weights(u)
        return u * channel * spatial


class ParallelImprovedPMAM(nn.Module):
    """Original PreConv/BN/ReLU and outer residual, with parallel attention."""

    def __init__(self, channels: int):
        super().__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.pmam = ParallelPMAMDualPath(channels)

    def forward(self, x):
        return x + self.pmam(self.pre_conv(x))


class ConvNeXtParallelPMAM(nn.Module):
    """Insert a3 after features[5] and a4 after features[7], once each."""

    def __init__(self, backbone, a3, a4):
        super().__init__()
        # Register the original modules directly, without a backbone.* prefix.
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier
        self.a3 = a3
        self.a4 = a4

    def forward(self, x):
        for index, stage in enumerate(self.features):
            x = stage(x)
            if index == 5:
                x = self.a3(x)
            elif index == 7:
                x = self.a4(x)
        return self.classifier(self.avgpool(x))


def build_model(num_classes, weight_path=None):
    """Construct A first, then initialize a3/a4 without advancing its RNG state."""
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    backbone = _build_baseline(num_classes, weight_path)
    # Conv2d/BatchNorm defaults are intentionally retained. The saved CPU RNG
    # state is restored on exit; no CUDA RNG stream is reseeded or consumed.
    with torch.random.fork_rng(devices=[]):
        a3 = ParallelImprovedPMAM(384)
        a4 = ParallelImprovedPMAM(768)
    return ConvNeXtParallelPMAM(backbone, a3, a4).float()


def optimizer_for(model):
    """Identity-based backbone/attention/head groups under the fixed protocol."""
    attention_ids = {id(parameter) for module in (model.a3, model.a4) for parameter in module.parameters()}
    head_ids = {id(parameter) for parameter in model.classifier[2].parameters()}
    if attention_ids & head_ids:
        raise ValueError("Attention and classifier parameters must be independent")
    if {id(parameter) for parameter in model.a3.parameters()} & {id(parameter) for parameter in model.a4.parameters()}:
        raise ValueError("a3 and a4 parameters must not be shared")
    no_decay_ids = set()
    for module in model.modules():
        normalization = isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.GroupNorm))
        for name, parameter in module.named_parameters(recurse=False):
            if normalization or name in ("bias", "layer_scale") or parameter.ndim <= 1:
                no_decay_ids.add(id(parameter))
    rates = {"backbone": 1e-4, "attention": 1e-3, "head": 1e-3}
    groups, assigned = {}, set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        identity = id(parameter)
        if identity in assigned:
            raise ValueError(f"Parameter assigned more than once: {name}")
        assigned.add(identity)
        role = "attention" if identity in attention_ids else "head" if identity in head_ids else "backbone"
        decay = identity not in no_decay_ids
        key = (role, decay)
        if key not in groups:
            groups[key] = {"params": [], "param_names": [], "lr": rates[role], "initial_lr": rates[role],
                           "weight_decay": 0.05 if decay else 0.0, "role": role}
        groups[key]["params"].append(parameter)
        groups[key]["param_names"].append(name)
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if assigned != expected:
        raise ValueError("Optimizer groups do not cover exactly the trainable parameters")
    return torch.optim.AdamW(list(groups.values()), betas=(0.9, 0.999), eps=1e-8, foreach=False)

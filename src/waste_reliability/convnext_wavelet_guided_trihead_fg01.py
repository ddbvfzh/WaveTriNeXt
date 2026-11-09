"""A+a+b with differentiable wavelet-guided auxiliary regions; main-only predictions."""
from __future__ import annotations

import torch
from torch import nn
from .convnext_pmam_wtconv import build_model as build_original, WTConv2d, ScaleModule, WT_SOURCE_COMMIT

MODEL_ID = "convnext_tiny_pmam_wtconv_wavelet_guided_trihead_fg01"
PARAMETERS = {"trashnet": 30_595_446, "realwaste": 30_598_908}
TRIREGION_ADDED_PARAMETERS = {6: 210_060, 9: 211_215}
WT_ADDED_PARAMETERS = 346_464
ATTENTION_ADDED_PARAMETERS = 2_214_180
TRIREGION_CONFIG = {
    "mode": "wavelet_guided_parallel_auxiliary", "taps": [[1, 2, 1], [3, 2, 0]],
    "tap_semantics": "processed_scale_output_before_reconstruction; C_band_flatten",
    "inputs": [384, 768, 384], "projection_channels": 128, "projection_groups": 16,
    "projection_activation": None, "mask_kernel": 3, "mask_bias": False,
    "mask_order": ["foreground", "background", "ambiguous"], "mask_activation": "softmax_dim1",
    "normalization": {"eps": 1e-5, "weight_init": 1.0, "bias_init": 0.0, "bn_momentum": 0.1},
    "pool_source": "post_a3_F3", "pool_epsilon": 1e-6,
    "region_classifier": "shared_linear_384_K_bias_true_no_norm", "detach": False,
    "final_prediction": "logits_cls_only", "ambiguous_classifier": False,
}
LOSS_CONFIG = {"global_ce_weight": 1.0, "foreground_ce_weight": 0.1, "background_weight": 0.1,
               "label_smoothing": 0.1, "background": "logsumexp_all_minus_logsumexp_non_target",
               "extra_regularization": None}


class WaveletGuidedTriHead(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, 128, 1, bias=False), nn.GroupNorm(16, 128))
            for channels in (384, 768, 384)])
        self.mask_conv = nn.Conv2d(384, 3, 3, padding=1, bias=False)
        self.mask_bn = nn.BatchNorm2d(3, eps=1e-5, momentum=0.1)
        self.region_classifier = nn.Linear(384, classes)

    def forward(self, w1, w2, f3):
        for value, channels in zip((w1, w2, f3), (384, 768, 384)):
            if value.ndim != 4 or value.shape[1:] != (channels, 14, 14):
                raise ValueError("Unexpected wavelet-guided feature shape")
        guide = torch.cat([projection(value) for projection, value in
                           zip(self.projections, (w1, w2, f3))], dim=1)
        masks = self.mask_bn(self.mask_conv(guide)).softmax(dim=1)
        vectors = [((f3 * masks[:, i:i+1]).sum((2, 3)) /
                    (masks[:, i:i+1].sum((2, 3)) + 1e-6)) for i in (0, 1)]
        return {"logits_fg": self.region_classifier(vectors[0]),
                "logits_bg": self.region_classifier(vectors[1]), "region_maps": masks}


class ConvNeXtWaveletGuidedTriHead(nn.Module):
    def __init__(self, original, trihead):
        super().__init__()
        self.features, self.avgpool, self.classifier = original.features, original.avgpool, original.classifier
        self.a3, self.a4, self.trihead = original.a3, original.a4, trihead

    def forward(self, x, return_aux=False):
        captured, handles = {}, []
        try:
            if return_aux:
                def capture(key):
                    def hook(_module, _inputs, output):
                        if key in captured:
                            raise RuntimeError("Wavelet tap executed more than once")
                        captured[key] = output
                    return hook
                for key, stage, level in (("w1", 1, 1), ("w2", 3, 0)):
                    handles.append(self.features[stage][2].block[0].wavelet_scale[level]
                                   .register_forward_hook(capture(key)))
            f3 = None
            for index, stage in enumerate(self.features):
                x = stage(x)
                if index == 5:
                    x = self.a3(x)
                    f3 = x
                elif index == 7:
                    x = self.a4(x)
            logits = self.classifier(self.avgpool(x))
            if not return_aux:
                return logits
            return {"logits_cls": logits, **self.trihead(captured["w1"], captured["w2"], f3)}
        finally:
            for handle in handles:
                handle.remove()


def build_model(num_classes, weight_path=None):
    original = build_original(num_classes, weight_path)
    with torch.random.fork_rng(devices=[]):
        trihead = WaveletGuidedTriHead(num_classes)
    return ConvNeXtWaveletGuidedTriHead(original, trihead).float()


def optimizer_for(model):
    roles = {"attention": {id(p) for m in (model.a3, model.a4) for p in m.parameters()},
             "head": {id(p) for p in model.classifier[2].parameters()},
             "triregion": {id(p) for p in model.trihead.parameters()}, "wavelet": set()}
    for module in model.modules():
        if isinstance(module, WTConv2d):
            roles["wavelet"].update({id(p) for p in module.parameters()} -
                                    {id(p) for p in module.base_conv.parameters()})
    ids = list(roles.values())
    if any(a & b for i, a in enumerate(ids) for b in ids[i+1:]):
        raise ValueError("Optimizer role overlap")
    no_decay = set()
    for module in model.modules():
        norm = isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.GroupNorm))
        for name, parameter in module.named_parameters(recurse=False):
            if norm or isinstance(module, ScaleModule) or name in ("bias", "layer_scale") or parameter.ndim <= 1:
                no_decay.add(id(parameter))
    groups, seen = {}, set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError("Duplicate optimizer parameter")
        seen.add(id(parameter))
        role = next((role for role, members in roles.items() if id(parameter) in members), "backbone")
        decay = id(parameter) not in no_decay
        key = role, decay
        rate = 1e-4 if role == "backbone" else 1e-3
        group = groups.setdefault(key, {"params": [], "param_names": [], "role": role,
                                       "lr": rate, "initial_lr": rate, "weight_decay": .05 if decay else 0.0})
        group["params"].append(parameter)
        group["param_names"].append(name)
    if seen != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError("Optimizer parameter coverage mismatch")
    return torch.optim.AdamW(list(groups.values()), betas=(.9, .999), eps=1e-8, foreach=False)

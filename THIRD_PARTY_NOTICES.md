# Third-party notices

## WTConv

`src/waste_reliability/convnext_pmam_wtconv.py` adapts the multilevel wavelet
algorithm and Haar filter convention from
[BGU-CS-VIL/WTConv](https://github.com/BGU-CS-VIL/WTConv), revision
`4a84da2218924b3973de2235d641f2ecf969c346`.

The upstream MIT notice is retained verbatim in
[`third_party/wtconv/LICENSE`](third_party/wtconv/LICENSE). The specialization
and initialization details are recorded in
[`third_party/wtconv/README.md`](third_party/wtconv/README.md).

## PyTorch and Torchvision

The implementation imports ConvNeXt-T and image transforms from Torchvision;
their source and model checkpoint are not vendored here. Preserve their own
licenses and attribution when redistributing those dependencies. The exact
official pretrained checkpoint is documented in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Datasets and project code

Dataset-specific provenance and terms are recorded in `datasets/README.md`
and its accompanying source/license files. Dataset terms are separate from
the software terms. No blanket license over the author's original project
code is assigned by this packaging step; the author can select a project
license before publishing. The retained WTConv notice continues to apply to
that adapted component.

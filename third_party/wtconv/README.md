# WTConv provenance

`src/waste_reliability/convnext_pmam_wtconv.py` adapts the multilevel algorithm
and Haar filter convention from [BGU-CS-VIL/WTConv](https://github.com/BGU-CS-VIL/WTConv),
commit `4a84da2218924b3973de2235d641f2ecf969c346`:

- `wtconv/wtconv2d.py`
- `wtconv/util/wavelet.py`

The upstream MIT license is retained in `LICENSE` beside this file.

The local experiment specializes the implementation to `db1` (Haar), spatial
stride one, and equal input/output channels. The approved configuration uses
7×7 convolutions and two decomposition levels. Haar coefficients are constructed
directly, in the same dtype and ordering as the upstream PyWavelets-based
implementation, so no new runtime dependency is needed. Fixed transform filters
are persistent buffers instead of non-trainable Parameters. Pretrained ConvNeXt
depthwise weights/biases initialize the base branches; the upstream scale
initializations (1.0 and 0.1) and raw-LL recursion are unchanged.

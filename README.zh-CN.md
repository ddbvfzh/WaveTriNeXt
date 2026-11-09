# WaveTriNeXt 论文代码与实验归档

[English](README.md) · [结果表](results/README.md) · [数据说明](datasets/README.md) · [独立评估](docs/EVALUATION.md) · [复现说明](docs/REPRODUCIBILITY.md) · [上传 GitHub](docs/GITHUB_UPLOAD.md)

本文件夹整理了论文表格中的五组实验，统一使用 `baseline`、`pmam`、`wtconv`、`wtconv_pmam`、`wavetrinext` 五个名称。包含代码、配置、两个完整数据集、固定数据划分，以及 30 次正式运行的原始结果。

## 从哪里看

| 内容 | 位置 |
|---|---|
| 论文主结果表 | [results/README.md](results/README.md) |
| 可直接读取的表格 | [results/paper_table.csv](results/paper_table.csv) |
| 三种子均值和标准差 | [results/summary.csv](results/summary.csv) |
| 逐种子结果 | [results/per_seed.csv](results/per_seed.csv) |
| 每次训练的原始文件 | `results/模型名/数据集/seed_14或42或70/` |
| 模型与训练代码 | `src/waste_reliability/` |
| 模型权重导入清单与可用文件 | `pretrained/manifest.json`、`pretrained/模型名/数据集/seed_N/best.pt` |
| 五组配置 | `configs/` |
| 图片与固定划分 | `datasets/trashnet/`、`datasets/realwaste/` |
| 原始目录映射与数据来源 | `provenance/`、[datasets/README.md](datasets/README.md) |

## 数据与结果

两套数据共 **7,279 张图片，约 698 MiB**，已经复制到本文件夹。TrashNet 为六类、RealWaste 为九类，独立训练，未混入旧项目的五类跨域实验。

TrashNet 原始图片 2,527 张，正式固定清单 2,521 张，训练/验证/测试为 1,765/379/377；6 张重复组标签冲突图片仍保留，但不会进入训练清单。RealWaste 共 4,752 张，划分为 3,327/713/712。

五组模型各有两个数据集 × 三个种子（14、42、70），共 30 次运行，每次 50 轮。整理时核验了原始文件 SHA256，并从 16,335 条预测重新计算 Accuracy 和 Macro-F1，结果与提供的论文表格一致。

`wavetrinext` 对应 `wavelet_guided_trihead_fg01`，前景辅助损失和背景损失权重都是 0.1，预测只用主分类头。F3-only 和背景权重为 0 的额外实验未纳入本次五组归档。

## 使用

进入本文件夹，先检查数据与结果（仅需 Python 标准库）：

```bash
python scripts/verify_datasets.py
python scripts/summarize_results.py
```

如需后续训练，先阅读 [复现说明](docs/REPRODUCIBILITY.md)，安装依赖，再使用统一入口：

```bash
python -m pip install -r requirements.txt
python scripts/train.py --model wavetrinext --check-data-only
python scripts/train.py --model wavetrinext --datasets trashnet realwaste --seeds 14 42 70 --device cuda
```

将 `--model` 换成其他四个标识即可选择对应模型。新训练输出放在 `outputs/`。便携入口处理新目录路径，保留历史训练算法；本次整理没有重新跑完整训练，归档结果来自原实验。

## 直接评估已训练模型

```bash
python scripts/evaluate.py --model wavetrinext --datasets trashnet --seeds 14 --device cpu --output outputs/eval_wavetrinext_trashnet14
```

该命令加载已归档的 `best.pt`，使用固定测试集重新推理，输出指标、逐样本预测和
与原始记录的对比。两个数据集、全部种子及五组模型的批量命令见[独立评估说明](docs/EVALUATION.md)。
每次评估使用新的输出目录；`--limit` 可做少量图片检查，截断后的结果会标为部分测试集。

## 上传

已配置 `.gitignore` 和 `.gitattributes`：数据图片以及 `pretrained/` 中导入的 `best.pt` 由 Git LFS 管理；`outputs/` 中新训练生成的权重和缓存仍不提交。具体命令见 [上传说明](docs/GITHUB_UPLOAD.md)。原始实验记录关闭自动换行转换，以免 SHA256 改变。

五组模型的 **30 份最佳权重已全部归档**，每组包含两个数据集 × 三个种子，总计约 **3.53 GB**，无缺失项。文件大小、对应实验和 SHA256 记录于 `pretrained/manifest.json`，全部与原始 `complete.json` 中 `artifacts_sha256.best.pt` 一致。运行 `python scripts/verify_pretrained.py --require-all` 可重新核验；用于接续训练的 `last.pt` 不在本次归档范围内。

导入的 `best.pt` 是训练完成的论文模型；训练命令的 `--weights` 参数用于官方 ImageNet 初始化文件，两者用途不同。私有的云端下载位置单独保存在父目录的 `云端权重位置_仅本地.md`，无需上传。数据来源及许可原文保留在数据说明和第三方声明中；论文题目、作者和项目代码许可证留待你正式发布时填写。GitHub 仓库为 [ddbvfzh/WaveTriNeXt](https://github.com/ddbvfzh/WaveTriNeXt)，当前设为**私有**，仅授权账号可访问。

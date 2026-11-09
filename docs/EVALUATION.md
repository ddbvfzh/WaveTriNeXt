# Evaluate the trained paper models / 独立评估

`scripts/evaluate.py` loads the archived `pretrained/.../best.pt` files, reads
the fixed test split, and performs a new forward pass. It does not retrain models
or download the ImageNet initialization. Run the commands below from the
repository root after installing `requirements.txt` and fetching Git LFS files.

## One checkpoint

```bash
python scripts/evaluate.py --model wavetrinext --datasets trashnet --seeds 14 --device cpu --output outputs/eval_wavetrinext_trashnet14
```

Use `--device cuda` on a compatible NVIDIA GPU, or `--device xpu` with a compatible
PyTorch XPU installation. `--batch-size` defaults to 8, matching the archived
evaluation batch size. `--threads` controls CPU threads and defaults to 2.
The five model aliases are `baseline`, `pmam`, `wtconv`,
`wtconv_pmam`, and `wavetrinext`.

## Both datasets and all seeds

The default datasets are TrashNet and RealWaste, and default seeds are 14, 42, 70:

```bash
python scripts/evaluate.py --model wavetrinext --device cuda --output outputs/eval_wavetrinext_all
```

To evaluate all 30 archived checkpoints:

```bash
python scripts/evaluate.py --models baseline pmam wtconv wtconv_pmam wavetrinext --device cuda --output outputs/eval_all
```

Choose a fresh output directory for each evaluation. Archived data, weights,
source code and paper results are protected from being used as output locations.

## Small input check

```bash
python scripts/evaluate.py --model wavetrinext --datasets trashnet --seeds 14 --device cpu --limit 8 --output outputs/eval_smoke
```

When fewer than all test images are evaluated, outputs are explicitly marked as
partial. These scores are diagnostic only and cannot be used as paper test scores.
Remove `--limit` for a complete test-set evaluation.

## What is checked and computed

- The checkpoint bytes are checked against the SHA256 in the original completed
  run and the published checkpoint manifest. Model, dataset, seed, classes and
  classifier dimensions must match; state dictionaries load strictly.
- The fixed dataset manifest and selected test image bytes are verified. The
  existing split and class order are preserved.
- Input preprocessing uses the archived ConvNeXt-T evaluation transform:
  resize short edge to 236, center crop to 224, RGB tensor and ImageNet
  normalization. The original EXIF orientation handling is preserved.
- Evaluation uses `eval()` and inference mode, FP32 with TF32 disabled. Only the
  main classification logits determine predictions, including for WaveTriNeXt.
- Predictions are computed from images and weights. Only afterward are they
  aligned by `sample_id` with the archived predictions for comparison.

Different devices, numerical kernels and batch sizes can change floating-point
logits slightly. The comparison reports prediction agreement, logit differences
and metric differences; a difference is reported rather than silently changing
predictions to match the archived table.

## Outputs

Each selected model/dataset/seed has its own result directory containing:

| File | Content |
|---|---|
| `test_predictions.csv` | Sample identity, true/predicted class and newly computed logits |
| `test_metrics.json` | Accuracy, Macro-F1, macro precision/recall, ordinary main-head CE and class metrics |
| `class_report.csv` | Per-class precision, recall, F1 and support |
| `confusion_matrix.csv` | Counts, with true classes as rows and predictions as columns |
| `confusion_matrix_normalized.csv` | Row-normalized confusion matrix |
| `evaluation.json` | Input provenance, checkpoint identity, runtime, test coverage and archive comparison |

Top-level `per_seed.csv` and `summary.csv` summarize the requested evaluations.
Seed averages and sample standard deviations are calculated across runs; the same
test images are not treated as three times as many independent samples.
Classification scores in CSV/JSON use [0, 1]; `loss` is ordinary cross-entropy.
For a single seed the standard deviation is left blank. The generated `README.md`
displays scores as percentages, while `status.json` records completion or failure.

To run the integrity and reporting tests without model inference:

```bash
python -m unittest discover -s tests -p test_evaluation.py -v
```

## Release validation (2026-11-09)

Fresh CPU inference completed on the full test splits for all five models and
both datasets at seed 14: **10 runs and 5,445 predictions**, all matching the
archived predicted classes. Accuracy and Macro-F1 matched exactly; the maximum
absolute logit difference was about 2.95e-5. Six additional four-image
WaveTriNeXt checks covered both datasets and all three seeds, including partial
reporting and rejection of an existing output directory.

The integrity/reporting suite passed 19 tests; one directory-symlink test was
skipped because Windows denied symlink creation. See
[`evaluation_validation.json`](../provenance/evaluation_validation.json) for the
measured results and scope. This check did not run the full test sets for all
30 checkpoints or repeat training.

## 中文说明

此入口直接使用已下载的论文模型权重重新推理。`train.py --weights` 的用途仍是
指定官方 ImageNet 初始化，不能用来代替这里的评估命令。

`verify_pretrained.py` 检查权重文件完整性，`summarize_results.py` 重算存档预测表格；
`evaluate.py` 才会实际读取图片并运行模型。它会生成新的指标、预测和与原实验的
对比记录，原始论文结果保持不变。少量图片的 `--limit` 检查不会被标记为完整测试。

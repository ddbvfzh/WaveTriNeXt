# Paper results

Means over training seeds 14, 42 and 70. Accuracy and Macro-F1 are percentages.

| Method | Params 6 classes (M) | Params 9 classes (M) | TrashNet Accuracy | TrashNet Macro-F1 | RealWaste Accuracy | RealWaste Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt-T | 27.82 | 27.83 | 95.84 | 95.36 | 94.71 | 94.75 |
| ConvNeXt-T + PMAM | 30.04 | 30.04 | 96.20 | 95.81 | 94.52 | 94.61 |
| ConvNeXt-T + WTConv | 28.17 | 28.17 | 95.93 | 95.51 | 94.99 | 94.97 |
| ConvNeXt-T + WTConv + PMAM | 30.39 | 30.39 | 96.55 | 96.26 | 95.18 | 95.26 |
| WaveTriNeXt | 30.60 | 30.60 | 97.08 | 96.57 | 95.22 | 95.17 |

See `summary.csv` for sample standard deviations (ddof=1), and `per_seed.csv` for each run.

Each `MODEL/DATASET/seed_N/` directory holds byte-preserved original configuration, completion record, history, predictions and metrics. Recovered checkpoints are stored separately under `pretrained/`; see `../pretrained/manifest.json` for availability. Historical checkpoint and source hashes in these records describe the original runs, not a newly trained model.

# WaveTriNeXt

**Waste image classification with ConvNeXt-T, WTConv, PMAM, and wavelet-guided TriHead.**

[中文说明](README.zh-CN.md) · [Results](results/README.md) · [Datasets](datasets/README.md) · [Evaluate](docs/EVALUATION.md) · [Reproducibility](docs/REPRODUCIBILITY.md) · [GitHub upload](docs/GITHUB_UPLOAD.md)

This repository organizes the five configurations used in the paper's ablation table. It includes model code, fixed dataset splits, the original images, and the original records from **30 completed runs**: five configurations × two datasets × three training seeds.

## Configurations

| Alias | Model | Description |
|---|---|---|
| `baseline` | ConvNeXt-T | Backbone baseline |
| `pmam` | ConvNeXt-T + PMAM | Parallel PMAM at stages 3 and 4 |
| `wtconv` | ConvNeXt-T + WTConv | WTConv at stages 1 and 2 |
| `wtconv_pmam` | ConvNeXt-T + WTConv + PMAM | Combined backbone without auxiliary branches |
| `wavetrinext` | WaveTriNeXt | Combined backbone and TriHead guided by W1, W2, and F3 |

WaveTriNeXt corresponds to the historical `wavelet_guided_trihead_fg01` experiment: foreground and background auxiliary loss weights are both 0.1. Final predictions use the main classification logits. The F3-only control and background-weight-zero variant are outside this five-model package.

## Results

Classification scores are **mean ± sample standard deviation** (ddof = 1) over seeds **14, 42, 70**, using the validation-selected checkpoint for each run. Scores are percentages; parameter counts are in millions (M). The best mean for each metric is bold.

| Method | Params, 6 classes (M) | Params, 9 classes (M) | TrashNet Accuracy ↑ | TrashNet Macro-F1 ↑ | RealWaste Accuracy ↑ | RealWaste Macro-F1 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt-T | 27.82 | 27.83 | 95.84 ± 0.55 | 95.36 ± 0.59 | 94.71 ± 0.08 | 94.75 ± 0.15 |
| ConvNeXt-T + PMAM | 30.04 | 30.04 | 96.20 ± 0.41 | 95.81 ± 0.53 | 94.52 ± 0.37 | 94.61 ± 0.49 |
| ConvNeXt-T + WTConv | 28.17 | 28.17 | 95.93 ± 0.41 | 95.51 ± 0.58 | 94.99 ± 0.45 | 94.97 ± 0.39 |
| ConvNeXt-T + WTConv + PMAM | 30.39 | 30.39 | 96.55 ± 0.96 | 96.26 ± 1.28 | 95.18 ± 0.21 | **95.26** ± 0.20 |
| WaveTriNeXt | 30.60 | 30.60 | **97.08** ± 0.46 | **96.57** ± 0.80 | **95.22** ± 0.74 | 95.17 ± 0.78 |

[Per-seed results](results/per_seed.csv), [means and sample standard deviations](results/summary.csv), and the [machine-readable paper table](results/paper_table.csv) are included. The result audit recomputes accuracy and Macro-F1 from all **16,335 prediction rows** and checks the original records' hashes.

## Layout

```text
WaveTriNeXt/
├── configs/                 # Five named experiment configurations
├── src/waste_reliability/   # Models, original training routines, portable adapter
├── scripts/                # Train, evaluate, verify data, rebuild tables
├── datasets/
│   ├── trashnet/            # images/, fixed manifest and split lists
│   └── realwaste/           # images/, fixed manifest and split lists
├── results/
│   └── MODEL/DATASET/seed_N/# Original config, history, metrics, predictions
├── pretrained/             # Checkpoint import manifest and available best.pt files
├── provenance/             # Source mappings, hashes and validation
├── third_party/            # Upstream component notices
└── docs/                   # Reproduction and publication instructions
```

## Data

**We use two existing datasets and credit their original creators.** This repository packages copies of their images together with the fixed splits used in our experiments.

- **TrashNet — Gary Thung and Mindy Yang.** Our copy is the `dataset-resized` release (512 × 384 pixels) from the [original TrashNet repository](https://github.com/garythung/trashnet). The authors collected waste-object photographs against a white background for their Stanford CS229 project. Please cite their repository and the report by Mindy Yang and Gary Thung, [*Classification of Trash for Recyclability Status* (2016)](https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf).
- **RealWaste — Sam Single, Saeid Iranmanesh, and Raad Raad (2023).** Our copy was downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/908/realwaste), with dataset DOI [10.24432/C5SS4G](https://doi.org/10.24432/C5SS4G). The images were collected at the Whyte's Gully Waste and Resource Recovery facility in Wollongong, Australia, as described in the [authors' repository](https://github.com/sam-single/realwaste). Please cite both the UCI dataset record and their [original paper in *Information*, 14(12), 633](https://doi.org/10.3390/info14120633).

The train/validation/test assignments below are **our fixed experimental splits**. Download URLs, dates, archive sizes, and SHA256 values are recorded in the [TrashNet download provenance](datasets/trashnet/provenance/source_download.json) and [RealWaste download provenance](datasets/realwaste/provenance/source_download.json).

| Dataset | Classes | Included images | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|
| TrashNet | 6 | 2,527 | 1,765 | 379 | 377 |
| RealWaste | 9 | 4,752 | 3,327 | 713 | 712 |

Six TrashNet images were excluded from the historical experimental manifest because of conflicting labels in duplicate groups; their original image files are retained. The fixed TrashNet manifest therefore contains 2,521 images. The two datasets are trained and evaluated independently, using their full six/nine classes.

Images total approximately **698 MiB**. The local package contains actual image files; `.gitattributes` routes them through Git LFS when committed. For a GitHub clone, run `git lfs pull` before verification. See the [dataset documentation](datasets/README.md) for source attribution, license records, exclusions and checksums.

<details>
<summary>Dataset references (BibTeX)</summary>

```bibtex
@misc{thung2016trashnet,
  author       = {Thung, Gary and Yang, Mindy},
  title        = {{TrashNet}: Dataset of Images of Trash},
  year         = {2016},
  howpublished = {GitHub repository},
  url          = {https://github.com/garythung/trashnet}
}

@techreport{yang2016recyclability,
  author      = {Yang, Mindy and Thung, Gary},
  title       = {Classification of Trash for Recyclability Status},
  institution = {Stanford University},
  type        = {CS229 Project Report},
  year        = {2016},
  url         = {https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf}
}

@misc{single2023realwasteDataset,
  author       = {Single, Sam and Iranmanesh, Saeid and Raad, Raad},
  title        = {{RealWaste}},
  year         = {2023},
  howpublished = {UCI Machine Learning Repository},
  note         = {Dataset},
  doi          = {10.24432/C5SS4G},
  url          = {https://doi.org/10.24432/C5SS4G}
}

@article{single2023realwaste,
  author  = {Single, Sam and Iranmanesh, Saeid and Raad, Raad},
  title   = {{RealWaste}: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning},
  journal = {Information},
  year    = {2023},
  volume  = {14},
  number  = {12},
  pages   = {633},
  doi     = {10.3390/info14120633},
  url     = {https://doi.org/10.3390/info14120633}
}
```

</details>

## Quick start

From this repository's root, audit data and archived results using Python's standard library:

```bash
python scripts/verify_datasets.py
python scripts/summarize_results.py
```

For model execution, install the dependencies described in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). The historical CUDA environment used Python 3.12, PyTorch 2.8.0 and torchvision 0.23.0; saved run configurations record the observed versions.

```bash
python -m pip install -r requirements.txt
python scripts/train.py --help
python scripts/train.py --model wavetrinext --check-data-only
python scripts/train.py --model wavetrinext --datasets trashnet realwaste --seeds 14 42 70 --device cuda
```

Use any of the five aliases with `--model`. The default budget is 50 epochs, effective batch size 32, micro-batch size 8, FP32, with AMP and TF32 disabled. Training uses the official ConvNeXt-T ImageNet-1K V1 initialization and validates its SHA256; supply `--weights PATH` to use an existing copy of that official initialization. Outputs go into `outputs/`, separate from the archived paper results.

The portable entry point adapts paths and output organization. Historical records remain unchanged, with their original hashes and frozen source provenance. Packaging and CPU checks are not a new 30-run reproduction; exact metrics after retraining are not guaranteed across hardware and software versions.

## Evaluate trained checkpoints

Evaluate a trained checkpoint directly, without training:

```bash
python scripts/evaluate.py --model wavetrinext --datasets trashnet --seeds 14 --device cpu --output outputs/eval_wavetrinext_trashnet14
```

The evaluator runs fresh inference on the fixed test images, saves predictions
and metrics, and compares them with the archived results. See
[EVALUATION.md](docs/EVALUATION.md) for all-model evaluation and partial-input checks.

## Weights, attribution, and publishing

All **30 validation-selected paper checkpoints** are included under `pretrained/MODEL/DATASET/seed_N/best.pt`, totaling 3,532,874,238 bytes (approximately 3.53 GB). `pretrained/manifest.json` records each file's size, identity and SHA256. Every file has been verified against `artifacts_sha256.best.pt` in its archived `results/MODEL/DATASET/seed_N/complete.json`; no runs are missing. Original `last.pt` training-resume files are outside this import scope. Run `python scripts/verify_pretrained.py --require-all` to recheck the complete collection.

The imported `best.pt` files are trained paper models. The training CLI's `--weights` argument instead expects the official ImageNet initialization listed above. The checkpoints in `pretrained/` use Git LFS when committed; private cloud retrieval locations are kept separately by the repository owner.

See [LICENSE_POLICY.md](LICENSE_POLICY.md) for the project license status and `third_party/` for upstream notices. This package does not invent a manuscript title, author list or DOI. The GitHub repository is [ddbvfzh/WaveTriNeXt](https://github.com/ddbvfzh/WaveTriNeXt), currently **private**. See [GITHUB_UPLOAD.md](docs/GITHUB_UPLOAD.md) for cloning with Git LFS and future updates.

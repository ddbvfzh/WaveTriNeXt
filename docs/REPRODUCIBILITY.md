# Reproducing the five paper configurations

This release contains the fixed TrashNet six-class and RealWaste nine-class
classification experiments. It does not use the older common-five-class
cross-domain protocol from the working repository.

## Environment

Use Python 3.10 or newer. The paper CUDA runs recorded PyTorch 2.8.0+cu128,
Torchvision 0.23.0+cu128, NumPy 2.2.6, pandas 2.3.3, Pillow 12.1.0 and
SciPy 1.15.3. Install PyTorch/Torchvision for the machine's accelerator, then:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python scripts/train.py --help
python scripts/smoke_models.py
```

The smoke command constructs all five models for six and nine classes, checks
224×224 CPU inference, finite logits, trainable parameter counts and optimizer
coverage. It uses random initialization, downloads nothing, and does not train.
Installing a CPU or XPU wheel enables local checks, but the paper training was
performed on CUDA. Identical results across hardware/library versions are not
guaranteed.

## Data and pretrained initialization

Keep `datasets/trashnet/manifest.csv` and `datasets/realwaste/manifest.csv`.
They retain the original sample IDs, labels, image digests and train/val/test
assignments. Only the image paths were relocated. See `datasets/README.md`.

```bash
python scripts/verify_datasets.py
python scripts/train.py --model wavetrinext --check-data-only
```

The training adapter checks the image SHA256 values and existing split counts,
then writes an absolute-path runtime copy under `outputs/<model>/data/`.
It does not resplit data and never writes into the archived `results/`.

The official ImageNet initialization is not bundled. On the first actual
training invocation the adapter downloads that initialization to
`weights/convnext_tiny-983f1562.pth`:

- Source: <https://download.pytorch.org/models/convnext_tiny-983f1562.pth>
- Enum: `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`
- SHA256: `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`

For offline training, provide an existing copy with
`--weights /path/to/convnext_tiny-983f1562.pth`. The full digest is verified.

## Imported trained checkpoints

`pretrained/manifest.json` records the retrieval and verification status for
the paper's validation-selected checkpoints. Available files are organized as
`pretrained/<model>/<dataset>/seed_<seed>/best.pt`. The manifest is the source
of truth for which runs are available; a result directory alone does not imply
that its model checkpoint was recovered.

Each imported file must match the SHA256 stored under
`artifacts_sha256.best.pt` in the corresponding archived `complete.json`.
`last.pt` files, which include training-resume state, are not imported.
Imported `best.pt` files are tracked by Git LFS.

These trained checkpoints are distinct from the official ImageNet checkpoint
required by `train.py --weights`. Use the archived run's model alias and class
count when loading a trained checkpoint; do not pass it as the ImageNet
initialization for a new paper-protocol training run.

The independent evaluator loads these trained checkpoints directly:

```bash
python scripts/evaluate.py --model wavetrinext --datasets trashnet --seeds 14 --device cpu --output outputs/eval_wavetrinext_trashnet14
```

It performs new image inference and exports predictions, metrics and a comparison
with the archive. See [EVALUATION.md](EVALUATION.md) for batch evaluation of all
30 checkpoints, output files and the explicit partial-test mode.

## Training

Run each command from the repository root:

```bash
python scripts/train.py --model baseline --device cuda
python scripts/train.py --model pmam --device cuda
python scripts/train.py --model wtconv --device cuda
python scripts/train.py --model wtconv_pmam --device cuda
python scripts/train.py --model wavetrinext --device cuda
```

Each command runs both datasets, seeds 14/42/70, 50 epochs, micro batch 8 and
effective batch 32. The preserved implementations use AdamW, backbone LR 1e-4,
new-module/head LR 1e-3, five warmup epochs, update-wise cosine decay, label
smoothing 0.1 and gradient clipping 1.0. AMP and TF32 are disabled. The best
checkpoint is selected by validation Macro-F1, then ordinary main-head cross
entropy, then the earlier epoch. Evaluation uses main logits only.

`wavetrinext` is the FG=0.1, BG=0.1 wavelet-guided TriHead model. Its mask head
uses W1/W2/F3, its foreground and background losses are auxiliary, and its
inference classifier remains the main classifier. It is not the later BG=0 or
F3-only ablation.

The model JSON files under `configs/` are **locked reference specifications**,
checked against `provenance/config_sha256.json`; they are not an arbitrary
hyperparameter interface. The supported command-line overrides are datasets,
seeds, epochs, micro batch size, device, pretrained weight location and output
directory. A changed training budget/device is a new run, not a reproduction
of the paper result. Model/optimizer/loss changes require an explicit source
change and a separate experiment, rather than editing descriptive JSON fields.

For one dataset/seed, or an interrupted run:

```bash
python scripts/train.py --model wavetrinext --datasets trashnet --seeds 14 --device cuda
python scripts/train.py --model wavetrinext --device cuda --resume
```

New runs are saved as `outputs/<model>/runs/<model>__<dataset>__seed<seed>/`.
They contain config, history, best/last checkpoints, predictions, metrics and
completion digests. Existing output configurations must match exactly for
resume. Use `--output` to create a separate experiment when changing settings.

## Historical code and release adapter

Ten modules under `src/waste_reliability/` were copied byte-for-byte from the
historical frozen snapshots, including the five models, optimizer builders,
augmentation/data loader, metric functions and WaveTriNeXt auxiliary losses.
`provenance/model_source_inventory.json` records their SHA256 and source.
The `data.py` dependency contains older data utilities because frozen
`closed_set_data.py` imports its hash helpers; the release entry point does
not invoke its five-class preparation logic.

`portable.py`, `_portable_loop.py` and `scripts/train.py` are new adapters. The
loop is adapted from the frozen baseline epoch/checkpoint transaction and
dispatches to the correct model-specific optimizer and training/evaluation
functions. The adapter uses new runtime manifest digests and records its own
source hashes. It does not claim byte identity with the old cloud orchestration
scripts or bypass their old deployment checks. Old scripts depended on cloud
paths, original manifest byte digests and deployment inventories, so running
those scripts directly from a relocated folder is unsupported.

`evaluation.py` and `scripts/evaluate.py` provide standalone checkpoint
evaluation, reusing the frozen model, data loader and classification metrics.
They verify the frozen source hashes and record the current adapter hashes in
each evaluation receipt.

The archived results are the completed historical experiments. Packaging and
CPU checks do not constitute a new 30-run training reproduction. Use
`scripts/summarize_results.py` to independently recalculate the published table
from archived predictions.

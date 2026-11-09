"""Train the five paper configurations from portable, fixed dataset splits."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['baseline','pmam','wtconv','wtconv_pmam','wavetrinext'], required=True)
    parser.add_argument('--datasets', choices=['trashnet','realwaste'], nargs='+', default=['trashnet','realwaste'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[14,42,70])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--micro-batch-size', type=int, default=8)
    parser.add_argument('--device', choices=['cuda','cpu','xpu'], default='cuda')
    parser.add_argument('--weights', type=Path, help='Official ConvNeXt-T ImageNet-1K checkpoint; otherwise downloaded to weights/')
    parser.add_argument('--output', type=Path, help='New training output; default outputs/<model>')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check-data-only', action='store_true', help='Verify/relocate image manifests, then exit without downloading weights or training')
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.micro_batch_size < 2 or 32 % args.micro_batch_size:
        parser.error('epochs >= 1 and micro batch >= 2 dividing effective batch 32 required')
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds) or len(set(args.datasets)) != len(args.datasets):
        parser.error('Datasets and nonnegative seeds must be unique')
    output = (args.output or ROOT / 'outputs' / args.model).resolve()
    for name in ['results','datasets','src','configs','provenance']:
        protected = ROOT / name
        if output == ROOT or output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            parser.error('Output must not overlap repository source, datasets or archived paper results')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sys.path.insert(0, str(ROOT / 'src'))
    import torch
    from waste_reliability import closed_set_baseline as base
    from waste_reliability.portable import prepare_manifest, WEIGHT_SHA256
    from waste_reliability._portable_loop import train_run
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifests = {dataset: prepare_manifest(dataset, output) for dataset in args.datasets}
    if args.check_data_only:
        print('Verified image bytes and fixed splits:', ', '.join(manifests))
        return
    device = base.resolve_device(args.device)
    weights = args.weights.resolve() if args.weights else base.fetch_weights(ROOT / 'weights')
    if base.digest(weights) != WEIGHT_SHA256:
        raise ValueError('Pretrained checkpoint SHA256 does not match the paper initialization')
    with base.suite_lock(output):
        for dataset in args.datasets:
            for seed in args.seeds:
                train_run(output, args.model, dataset, seed, manifests[dataset], weights,
                          device, args.epochs, args.micro_batch_size, args.resume)


if __name__ == '__main__':
    main()

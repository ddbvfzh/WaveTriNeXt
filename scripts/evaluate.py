"""Evaluate archived best.pt checkpoints on the fixed paper test sets."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', '--models', dest='models', nargs='+', required=True,
                        choices=['baseline','pmam','wtconv','wtconv_pmam','wavetrinext'])
    parser.add_argument('--datasets', nargs='+', choices=['trashnet','realwaste'], default=['trashnet','realwaste'])
    parser.add_argument('--seeds', nargs='+', type=int, choices=[14,42,70], default=[14,42,70])
    parser.add_argument('--device', choices=['cpu','cuda','xpu'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=2, help='CPU thread count (default 2)')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/evaluation', help='New or empty output directory')
    parser.add_argument('--limit', type=int, help='Debug only: evaluate the first N test samples; marks subset scores as partial')
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error('Batch size, threads and sample limit must be positive')
    if any(len(set(values)) != len(values) for values in (args.models,args.datasets,args.seeds)):
        parser.error('Models, datasets and seeds must not repeat')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sys.path.insert(0, str(ROOT/'src'))
    from waste_reliability.evaluation import run_evaluation
    try:
        result = run_evaluation(ROOT, args.output, args.models, args.datasets, args.seeds,
                                args.device, args.batch_size, args.limit, args.threads)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.exit(1, f'Evaluation failed: {error}\n')
    print(f"Completed {result['completed_runs']} evaluation runs; full_test={result['full_test']}", flush=True)


if __name__ == '__main__':
    main()

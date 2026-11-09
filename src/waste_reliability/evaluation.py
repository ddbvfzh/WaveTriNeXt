"""Independent inference of archived paper checkpoints on the fixed test splits.

Imports of PyTorch are deferred so validation and comparison logic can be tested
without loading accelerator libraries. This module never trains or downloads.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
import time

MODEL_NAMES = {
    'baseline': 'convnext_tiny',
    'pmam': 'convnext_tiny_improved_pmam_parallel_s34',
    'wtconv': 'convnext_tiny_wtconv_s12',
    'wtconv_pmam': 'convnext_tiny_improved_pmam_parallel_s34_wtconv_s12',
    'wavetrinext': 'convnext_tiny_pmam_wtconv_wavelet_guided_trihead_fg01',
}
CLASSES = {
    'trashnet': ('cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'),
    'realwaste': ('cardboard', 'food organics', 'glass', 'metal',
                  'miscellaneous trash', 'paper', 'plastic', 'textile trash', 'vegetation'),
}
METRICS = ('accuracy', 'macro_f1', 'macro_precision', 'macro_recall')
TEST_SIZES = {'trashnet': 377, 'realwaste': 712}
PREPROCESSING = {'exif_transpose': True, 'color': 'RGB', 'resize_shorter_side': 236,
                 'center_crop': 224, 'interpolation': 'bilinear', 'antialias': True,
                 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                 'implementation': 'ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()',
                 'test_time_augmentation': False}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_output(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root.is_relative_to(output):
        raise ValueError('Evaluation output must not be the repository or its ancestor')
    protected = ('results', 'pretrained', 'datasets', 'src', 'configs', 'provenance',
                 'scripts', 'docs', 'third_party', 'tests', 'weights', '.git')
    for name in protected:
        path = (root/name).resolve()
        if output == path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError('Evaluation output overlaps protected repository content: '+name)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Evaluation output must be a new or empty directory; existing files are never overwritten')
    return output


def validate_checkpoint(root, alias, dataset, seed):
    root = Path(root).resolve()
    if alias not in MODEL_NAMES or dataset not in CLASSES or seed not in (14, 42, 70):
        raise ValueError('Unknown paper model, dataset or seed')
    relative = f'pretrained/{alias}/{dataset}/seed_{seed}/best.pt'
    result_relative = f'results/{alias}/{dataset}/seed_{seed}'
    path, reference = (root/relative).resolve(), (root/result_relative).resolve()
    if not path.is_relative_to(root/'pretrained') or not reference.is_relative_to(root/'results'):
        raise ValueError('Checkpoint or result path escapes its repository directory')
    entries = read_json(root/'pretrained/manifest.json')['checkpoints']
    matches = [x for x in entries if (x.get('model'), x.get('dataset'), x.get('seed')) == (alias, dataset, seed)]
    if len(matches) != 1:
        raise ValueError(f'{alias}/{dataset}/{seed}: checkpoint manifest entry is missing or duplicated')
    item = matches[0]
    if item['path'] != relative or item['result_directory'] != result_relative or item['status'] != 'verified':
        raise ValueError('Checkpoint manifest identity or availability is invalid: '+relative)
    config, complete = read_json(reference/'config.json'), read_json(reference/'complete.json')
    if (config.get('model'), config.get('dataset'), config.get('seed'), config.get('classes')) != (
            MODEL_NAMES[alias], dataset, seed, list(CLASSES[dataset])):
        raise ValueError('Archived checkpoint configuration has the wrong model, dataset, seed or class order')
    if complete.get('status') != 'completed' or complete.get('config') != config:
        raise ValueError('Original completion record does not match the archived configuration')
    expected = complete['artifacts_sha256']['best.pt']
    if item.get('expected_sha256') != expected or item.get('sha256') != expected:
        raise ValueError('Checkpoint manifest digest differs from the original experiment')
    if not path.is_file():
        raise FileNotFoundError('Checkpoint not available; retrieve Git LFS contents: '+relative)
    if path.stat().st_size != item.get('bytes') or digest(path) != expected:
        raise ValueError('Checkpoint size or SHA256 mismatch: '+relative)
    return {'path': path, 'relative_path': relative, 'sha256': expected, 'config': config,
            'complete': complete, 'classes': list(CLASSES[dataset]), 'reference_directory': reference}


def validate_loaded_checkpoint(checkpoint, metadata):
    if not isinstance(checkpoint, dict) or 'model' not in checkpoint:
        raise ValueError('Expected an archived best.pt containing model and config')
    if checkpoint.get('config') != metadata['config']:
        raise ValueError('Loaded checkpoint configuration differs from its original run')
    if checkpoint.get('epoch') != metadata['complete'].get('best_epoch'):
        raise ValueError('Loaded checkpoint epoch differs from the validation-selected epoch')


def load_test_rows(root, dataset, limit=None):
    from .closed_set_data import load_manifest, CLASSES as CORE_CLASSES
    root = Path(root).resolve()
    if dataset not in CLASSES or tuple(CORE_CLASSES[dataset]) != CLASSES[dataset]:
        raise ValueError('Unsupported dataset or changed class order')
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit must be a positive integer')
    source = root/'datasets'/dataset/'manifest.csv'
    inventory = read_json(root/'datasets/dataset_inventory.json')['datasets'][dataset]
    if digest(source) != inventory['manifest_sha256']:
        raise ValueError('Fixed dataset manifest SHA256 mismatch: '+dataset)
    rows = load_manifest(source, dataset)
    test = rows[rows.split == 'test'].reset_index(drop=True)
    if len(test) != TEST_SIZES[dataset]:
        raise ValueError('Fixed test-set size differs from the paper protocol')
    full_size = len(test)
    if limit is not None:
        test = test.head(limit).copy()
    paths = []
    for row in test.to_dict('records'):
        relative = Path(row['path'])
        path = (root/relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(root/'datasets'/dataset/'images'):
            raise ValueError('Test image path escapes the dataset: '+row['path'])
        if not path.is_file() or digest(path) != row['sha256']:
            raise ValueError('Missing or changed test image: '+row['path'])
        paths.append(str(path))
    test['path'] = paths
    return test, {'manifest_sha256': digest(source), 'full_test_samples': full_size,
                  'evaluated_samples': len(test), 'full_test': len(test) == full_size,
                  'test_image_sha256_checked': True}


def compare_predictions(rows, logits, reference, classes, full_test):
    import numpy as np
    from .closed_set_metrics import classification_metrics
    rows = rows.to_dict('records') if hasattr(rows, 'to_dict') else list(rows)
    reference = reference.to_dict('records') if hasattr(reference, 'to_dict') else list(reference)
    logits = np.asarray(logits)
    if logits.shape != (len(rows), len(classes)) or not np.isfinite(logits).all():
        raise ValueError('Invalid evaluated logits')
    reference_ids = [x['sample_id'] for x in reference]
    actual_ids = [x['sample_id'] for x in rows]
    if len(set(reference_ids)) != len(reference_ids) or len(set(actual_ids)) != len(actual_ids):
        raise ValueError('Duplicate sample IDs in comparison')
    lookup = dict(zip(reference_ids, reference))
    if not set(actual_ids).issubset(lookup) or (full_test and set(actual_ids) != set(lookup)):
        raise ValueError('Evaluation and archived prediction sample IDs differ')
    aligned = [lookup[x] for x in actual_ids]
    for row, original in zip(rows, aligned):
        for key in ('label', 'class_name', 'group_id'):
            if str(row[key]) != str(original[key]):
                raise ValueError('Archived prediction metadata mismatch: '+row['sample_id'])
    labels = np.asarray([int(x['label']) for x in rows])
    predicted = logits.argmax(axis=1)
    archived = np.asarray([int(x['prediction']) for x in aligned])
    original_logits = np.asarray([[float(x[f'logit_{i}']) for i in range(len(classes))] for x in aligned])
    if not np.isfinite(original_logits).all() or not np.array_equal(original_logits.argmax(1), archived):
        raise ValueError('Archived predictions have invalid or inconsistent logits')
    current = classification_metrics(labels, predicted, classes)
    old = classification_metrics(labels, archived, classes)
    changed = int(np.count_nonzero(predicted != archived))
    return {'full_test': bool(full_test), 'compared_samples': len(rows),
            'comparison_scope': 'full_test' if full_test else 'matching_sample_subset_only',
            'prediction_agreement': float(np.mean(predicted == archived)),
            'changed_predictions': changed,
            'changed_sample_ids': [rows[i]['sample_id'] for i in np.flatnonzero(predicted != archived)],
            'max_abs_logit_difference': float(np.max(np.abs(logits-original_logits))),
            'metric_deltas': {key: current[key]-old[key] for key in METRICS},
            'reference_metrics': {'n_samples': len(rows), **{key: old[key] for key in METRICS}}}


def summarize(records):
    groups = {}
    for row in records:
        groups.setdefault((row['model'], row['dataset'], row['full_test'], row['n_samples']), []).append(row)
    result = []
    for (model, dataset, full_test, n_samples), group in groups.items():
        if len({x['seed'] for x in group}) != len(group):
            raise ValueError('Duplicate seeds in evaluation summary')
        row = {'model': model, 'dataset': dataset, 'full_test': full_test, 'n_samples': n_samples,
               'n_seeds': len(group), 'seeds': ','.join(str(x['seed']) for x in group)}
        for name in (*METRICS, 'loss'):
            row[name+'_mean'] = mean(x[name] for x in group)
            row[name+'_std'] = stdev(x[name] for x in group) if len(group) > 1 else None
        result.append(row)
    return result


def verify_sources(root):
    inventory = read_json(root/'provenance/model_source_inventory.json')['files']
    for item in inventory:
        if digest(root/item['path']) != item['sha256']:
            raise ValueError('Frozen model source changed: '+item['path'])
    paths = sorted((root/'src/waste_reliability').glob('*.py')) + [root/'scripts/evaluate.py']
    return {p.relative_to(root).as_posix(): digest(p) for p in paths}


def evaluate_run(root, output, alias, dataset, seed, rows, data_info, device, batch_size, source_hashes):
    import numpy as np
    import torch
    import torchvision
    from . import closed_set_baseline as base
    from .portable import model_module
    metadata = validate_checkpoint(root, alias, dataset, seed)
    reference_path = metadata['reference_directory']/'test_predictions.csv'
    expected_reference = metadata['complete']['artifacts_sha256']['test_predictions.csv']
    if digest(reference_path) != expected_reference:
        raise ValueError('Archived predictions changed: '+str(reference_path.relative_to(root)))
    destination = output/alias/dataset/f'seed_{seed}'
    destination.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'running', 'model': alias, 'dataset': dataset, 'seed': seed,
               'checkpoint': metadata['relative_path'], 'checkpoint_sha256': metadata['sha256'],
               'checkpoint_epoch': metadata['complete']['best_epoch'], 'classes': metadata['classes'],
               'device': str(device), 'batch_size': batch_size, 'dtype': 'float32', 'amp': False,
               'tf32': False, 'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
               'preprocessing': PREPROCESSING, 'source_sha256': source_hashes, **data_info}
    write_json(destination/'evaluation.json', receipt)
    started = time.monotonic()
    try:
        checkpoint = torch.load(metadata['path'], map_location='cpu', weights_only=True)
        validate_loaded_checkpoint(checkpoint, metadata)
        model = model_module(alias).build_model(len(metadata['classes']))
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        model.to(device)
        # Use the ordinary main-head evaluator for every model. WaveTriNeXt's
        # default forward returns logits_cls; auxiliary branches are not fused.
        metrics, labels, logits = base.evaluate(model, base.make_loader(rows, batch_size, False, seed), device, metadata['classes'])
        base.synchronize(device)
        if not np.array_equal(labels, rows.label.to_numpy()):
            raise ValueError('Test loader changed sample/label order')
        predictions = rows[['sample_id', 'class_name', 'label', 'group_id']].copy()
        predictions['prediction'] = logits.argmax(1)
        for index in range(len(metadata['classes'])):
            predictions[f'logit_{index}'] = logits[:, index]
        predictions.to_csv(destination/'test_predictions.csv', index=False)
        write_json(destination/'test_metrics.json', {**metrics, 'full_test': data_info['full_test']})
        write_csv(destination/'class_report.csv', metrics['per_class'])
        for name in ('confusion_matrix', 'confusion_matrix_normalized'):
            matrix_rows = [{'true_class': label, **dict(zip(metadata['classes'], values))}
                           for label, values in zip(metadata['classes'], metrics[name])]
            write_csv(destination/(name+'.csv'), matrix_rows)
        comparison = compare_predictions(rows, logits, read_csv(reference_path), metadata['classes'], data_info['full_test'])
        receipt.update(status='completed', elapsed_seconds=time.monotonic()-started, comparison=comparison,
                       output_sha256={p.name: digest(p) for p in destination.iterdir() if p.is_file() and p.name!='evaluation.json'})
        write_json(destination/'evaluation.json', receipt)
        del model
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
        if device.type == 'xpu': torch.xpu.empty_cache()
        return {'model': alias, 'dataset': dataset, 'seed': seed, 'full_test': data_info['full_test'],
                'n_samples': len(rows), **{name: metrics[name] for name in (*METRICS, 'loss')},
                'prediction_agreement': comparison['prediction_agreement'],
                'changed_predictions': comparison['changed_predictions'],
                'max_abs_logit_difference': comparison['max_abs_logit_difference'],
                'accuracy_delta': comparison['metric_deltas']['accuracy'],
                'macro_f1_delta': comparison['metric_deltas']['macro_f1'],
                'run_directory': destination.relative_to(output).as_posix()}
    except Exception as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(destination/'evaluation.json', receipt)
        raise


def run_evaluation(root, output, models, datasets, seeds, device='cpu', batch_size=8, limit=None, threads=2):
    import torch
    from . import closed_set_baseline as base
    root, output = Path(root).resolve(), validate_output(root, output)
    for values, allowed, name in ((models, MODEL_NAMES, 'models'), (datasets, CLASSES, 'datasets'), (seeds, (14,42,70), 'seeds')):
        if not values or len(set(values)) != len(values) or any(x not in allowed for x in values):
            raise ValueError('Invalid or repeated '+name)
    if type(batch_size) is not int or batch_size < 1 or type(threads) is not int or threads < 1:
        raise ValueError('Batch size and CPU thread count must be positive integers')
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit must be a positive integer')
    device = base.resolve_device(device)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sources = verify_sources(root)
    tests = {dataset: load_test_rows(root, dataset, limit) for dataset in datasets}
    output.mkdir(parents=True, exist_ok=True)
    records = []
    planned = len(models)*len(datasets)*len(seeds)
    status = {'status':'running', 'requested_runs':planned, 'completed_runs':0,
              'models':list(models), 'datasets':list(datasets), 'seeds':list(seeds), 'limit':limit}
    write_json(output/'status.json', status)
    try:
        for alias in models:
            for dataset in datasets:
                rows, info = tests[dataset]
                for seed in seeds:
                    record = evaluate_run(root, output, alias, dataset, seed, rows, info, device, batch_size, sources)
                    records.append(record)
                    write_csv(output/'per_seed.csv', records)
                    write_csv(output/'summary.csv', summarize(records))
                    status['completed_runs'] = len(records)
                    write_json(output/'status.json', status)
                    print(json.dumps(record), flush=True)
        status.update(status='completed', full_test=all(x['full_test'] for x in records),
                      total_prediction_rows=sum(x['n_samples'] for x in records))
        write_json(output/'status.json', status)
        heading = '# Independent test-set evaluation' if status['full_test'] else '# PARTIAL evaluation — smoke check only'
        lines = [heading, '', 'Metrics are recomputed by model inference. Accuracy and Macro-F1 below are percentages.',
                 'CSV/JSON metrics use [0, 1]; summary standard deviations are sample SD (ddof=1), blank for one seed.', '',
                 '| Model | Dataset | Test samples | Seeds | Accuracy (mean ± SD) | Macro-F1 (mean ± SD) |', '|---|---|---:|---|---:|---:|']
        for row in summarize(records):
            scores = []
            for metric in ('accuracy', 'macro_f1'):
                spread = row[metric+'_std']
                scores.append(f"{row[metric+'_mean']*100:.2f} ± " + (f'{spread*100:.2f}' if spread is not None else '—'))
            lines.append(f"| {row['model']} | {row['dataset']} | {row['n_samples']} | {row['seeds']} | {' | '.join(scores)} |")
        lines += ['', 'Per-run `evaluation.json` records hashes, preprocessing, device and comparisons with archived predictions.',
                  'Floating-point logit differences across hardware are reported; no archived result is changed.']
        if not status['full_test']:
            lines += ['', '**A sample limit was applied. These subset scores are not the paper test-set results.**',
                      'Comparison deltas use only matching archived sample IDs, not the full archived test scores.']
        (output/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    except Exception as error:
        status.update(status='failed', completed_runs=len(records), error=f'{type(error).__name__}: {error}')
        write_json(output/'status.json', status)
        raise
    return status

"""Portable paths and model selection for the five paper configurations."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

from . import closed_set_baseline as base
from .closed_set_data import load_manifest

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    'baseline': 'closed_set_baseline',
    'pmam': 'convnext_pmam',
    'wtconv': 'convnext_wtconv',
    'wtconv_pmam': 'convnext_pmam_wtconv',
    'wavetrinext': 'convnext_wavelet_guided_trihead_fg01',
}
WEIGHT_SHA256 = '983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d'


def model_module(alias):
    return importlib.import_module('.' + MODELS[alias], __package__)


def epoch_functions(alias):
    if alias == 'wavetrinext':
        from . import wavelet_guided_trihead_fg01_training as training
        return training.train_epoch, training.evaluate
    return base.train_epoch, base.evaluate


def source_hashes():
    paths = sorted((ROOT / 'src').rglob('*.py')) + [ROOT / 'scripts/train.py']
    return {p.relative_to(ROOT).as_posix(): base.digest(p) for p in paths}


def portable_config(alias, dataset, seed, manifest, weights, device, epochs, micro_batch):
    config_path = ROOT / 'configs' / f'{alias}.json'
    locked = json.loads((ROOT / 'provenance/config_sha256.json').read_text(encoding='utf-8'))
    if base.digest(config_path) != locked[config_path.name]:
        raise ValueError('Paper configuration JSON changed. These are locked reference specifications, not general hyperparameter inputs; supported overrides are CLI arguments.')
    template = json.loads(config_path.read_text(encoding='utf-8'))
    config = base.run_config(dataset, seed, manifest, weights, device, epochs, micro_batch)
    runtime = {key: config[key] for key in (
        'dataset', 'seed', 'classes', 'epochs', 'micro_batch_size', 'device',
        'manifest_sha256', 'weights_sha256', 'torch', 'torchvision', 'packages', 'warmup_epochs')}
    config.update(template)
    config.update(runtime)
    config.update({'alias': alias, 'parameter_count': template['parameter_counts'][dataset],
                   'code_sha256': source_hashes(), 'release_adapter': 'portable-v1',
                   'config_file_sha256': base.digest(ROOT / 'configs' / f'{alias}.json'),
                   'source_execution': 'portable_release_adapter; historical algorithms unchanged'})
    if config['weights_sha256'] != WEIGHT_SHA256:
        raise ValueError('Expected the official ConvNeXt-T ImageNet-1K V1 checkpoint')
    return config


def prepare_manifest(dataset, output):
    """Resolve paths into a separate runtime copy; never regenerate the split."""
    source = ROOT / 'datasets' / dataset / 'manifest.csv'
    inventory = json.loads((ROOT / 'datasets/dataset_inventory.json').read_text(encoding='utf-8'))
    expected_manifest = inventory['datasets'][dataset]['manifest_sha256']
    if base.digest(source) != expected_manifest:
        raise ValueError('Portable manifest digest changed; the paper sample IDs, labels and split assignments are locked')
    rows = load_manifest(source, dataset)
    counts = rows.groupby('split').size().to_dict()
    expected = {'trashnet': {'train':1765, 'val':379, 'test':377},
                'realwaste': {'train':3327, 'val':713, 'test':712}}
    if counts != expected[dataset]:
        raise ValueError('Dataset split counts differ from the paper protocol')
    resolved = []
    for row in rows.to_dict('records'):
        relative = Path(row['path'])
        path = (ROOT / relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(ROOT / 'datasets' / dataset / 'images'):
            raise ValueError('Manifest image escapes its dataset directory')
        if not path.is_file() or base.digest(path) != row['sha256']:
            raise ValueError('Missing image or image SHA256 mismatch: ' + row['path'])
        resolved.append(str(path))
    rows['path'] = resolved
    destination = Path(output) / 'data' / dataset / 'manifest.csv'
    content = rows.to_csv(index=False, lineterminator='\n').encode('utf-8')
    if destination.exists() and destination.read_bytes() != content:
        raise ValueError('Runtime manifest changed; use a new output directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    base.save_json(destination.parent / 'relocation.json', {
        'portable_manifest_sha256': base.digest(source),
        'runtime_manifest_sha256': base.digest(destination),
        'split_policy': 'fixed existing sample IDs; only image paths relocated',
        'counts': counts,
    })
    return destination

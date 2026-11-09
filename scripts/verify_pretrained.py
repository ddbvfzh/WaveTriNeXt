"""Verify recovered paper checkpoints against original completion-record hashes."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(root):
    manifest = json.loads((root/'pretrained/manifest.json').read_text(encoding='utf-8'))
    mapping = json.loads((root/'provenance/experiment_mapping.json').read_text(encoding='utf-8'))
    expected = {}
    for model, group in mapping.items():
        for run in group['runs']:
            path = f"pretrained/{model}/{run['dataset']}/seed_{run['seed']}/best.pt"
            expected[path] = (model, run['dataset'], run['seed'], run['packaged_run'], run['original_run'])
    entries = manifest['checkpoints']
    if len(expected) != 30 or len(entries) != 30 or {x['path'] for x in entries} != set(expected):
        raise ValueError('Checkpoint manifest must contain exactly the 30 original paper runs, without missing or duplicate entries')
    for item in entries:
        identity = (item['model'], item['dataset'], item['seed'], item['result_directory'], item['original_run'])
        if identity != expected[item['path']]:
            raise ValueError('Checkpoint identity differs from experiment mapping: '+item['path'])
    verified, unavailable = [], []
    for item in manifest['checkpoints']:
        path = (root/item['path']).resolve()
        if not path.is_relative_to(root/'pretrained'):
            raise ValueError('Checkpoint path escapes pretrained/')
        complete = json.loads((root/item['result_directory']/'complete.json').read_text(encoding='utf-8'))
        expected = complete['artifacts_sha256']['best.pt']
        if item['expected_sha256'] != expected:
            raise ValueError('Manifest differs from original result: '+item['path'])
        if not path.exists():
            if item['status'] == 'verified':
                raise ValueError('Previously verified checkpoint is missing: '+item['path'])
            unavailable.append({'path':item['path'], 'status':item['status']})
            continue
        actual = sha256(path)
        if actual != expected:
            raise ValueError('Checkpoint hash mismatch: '+item['path'])
        if item['status'] != 'verified' or item.get('sha256') != actual or item.get('bytes') != path.stat().st_size:
            raise ValueError('Checkpoint manifest is not finalized: '+item['path'])
        verified.append({'path':item['path'], 'bytes':path.stat().st_size, 'sha256':actual})
    expected_paths = {x['path'] for x in manifest['checkpoints']}
    unexpected = [p.relative_to(root).as_posix() for p in (root/'pretrained').rglob('*.pt') if p.relative_to(root).as_posix() not in expected_paths]
    if unexpected:
        raise ValueError('Unexpected checkpoints: '+str(unexpected))
    return {'status':'complete' if not unavailable else 'incomplete', 'expected':len(manifest['checkpoints']), 'verified':len(verified), 'unavailable':len(unavailable), 'verified_bytes':sum(x['bytes'] for x in verified), 'verified_files':verified, 'unavailable_files':unavailable}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--require-all',action='store_true')
    args = parser.parse_args()
    result = verify(args.root.resolve())
    print(json.dumps(result,indent=2))
    if args.require_all and result['unavailable']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

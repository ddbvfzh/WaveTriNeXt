"""Audit archived predictions and regenerate the paper's tables (standard library)."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

MODELS = {
    'baseline': 'ConvNeXt-T', 'pmam': 'ConvNeXt-T + PMAM',
    'wtconv': 'ConvNeXt-T + WTConv', 'wtconv_pmam': 'ConvNeXt-T + WTConv + PMAM',
    'wavetrinext': 'WaveTriNeXt',
}
EXPECTED = {
    'baseline': [95.84, 95.36, 94.71, 94.75],
    'pmam': [96.20, 95.81, 94.52, 94.61],
    'wtconv': [95.93, 95.51, 94.99, 94.97],
    'wtconv_pmam': [96.55, 96.26, 95.18, 95.26],
    'wavetrinext': [97.08, 96.57, 95.22, 95.17],
}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    index = json.loads((root/'provenance/result_files.json').read_text(encoding='utf-8'))
    for item in index:
        require(hashlib.sha256((root/item['path']).read_bytes()).hexdigest() == item['sha256'], 'Changed archived file: '+item['path'])
    per_seed, summary, table = [], [], []
    fixed_test_ids = {
        dataset: {r['sample_id']: int(r['label']) for r in read_csv(root/'datasets'/dataset/'manifest.csv') if r['split'] == 'test'}
        for dataset in ('trashnet', 'realwaste')
    }
    dataset_ids = {}
    for variant, name in MODELS.items():
        model_row = {'model': variant, 'method': name}
        actual = []
        for dataset in ('trashnet', 'realwaste'):
            group = []
            for seed in (14, 42, 70):
                run = root/'results'/variant/dataset/f'seed_{seed}'
                config = json.loads((run/'config.json').read_text(encoding='utf-8'))
                complete = json.loads((run/'complete.json').read_text(encoding='utf-8'))
                metrics = json.loads((run/'test_metrics.json').read_text(encoding='utf-8'))
                pred = read_csv(run/'test_predictions.csv')
                history = read_csv(run/'history.csv')
                require(complete['status'] == 'completed', f'{run}: incomplete')
                require(len(history) == 50 and {int(x['epoch']) for x in history} == set(range(1,51)), f'{run}: history')
                require(config['dataset'] == dataset and config['seed'] == seed, f'{run}: identity')
                ids = {r['sample_id']: int(r['label']) for r in pred}
                require(len(ids) == len(pred), f'{run}: duplicate prediction IDs')
                require(len(pred) == {'trashnet':377, 'realwaste':712}[dataset], f'{run}: prediction count')
                require(dataset_ids.setdefault(dataset, ids) == ids, f'{run}: different test labels/IDs')
                require(ids == fixed_test_ids[dataset], f'{run}: predictions do not match the packaged fixed test split')
                classes = metrics['classes']
                matrix = [[0]*len(classes) for _ in classes]
                for row in pred:
                    label, prediction = int(row['label']), int(row['prediction'])
                    require(0 <= label < len(classes) and 0 <= prediction < len(classes), f'{run}: out-of-range label')
                    matrix[label][prediction] += 1
                accuracy = sum(matrix[k][k] for k in range(len(classes)))/len(pred)
                f1s = []
                for k in range(len(classes)):
                    tp = matrix[k][k]
                    denominator = sum(matrix[k]) + sum(row[k] for row in matrix)
                    f1s.append(2*tp/denominator if denominator else 0)
                macro_f1 = mean(f1s)
                require(math.isclose(accuracy, metrics['accuracy'], abs_tol=1e-12), f'{run}: accuracy mismatch')
                require(math.isclose(macro_f1, metrics['macro_f1'], abs_tol=1e-12), f'{run}: macro-F1 mismatch')
                for filename, expected_hash in complete.get('artifacts_sha256', {}).items():
                    if (run/filename).is_file():
                        require(hashlib.sha256((run/filename).read_bytes()).hexdigest() == expected_hash, f'{run}: original hash mismatch {filename}')
                # Baseline count is the full torchvision ConvNeXt-T (classifier included).
                parameters = config.get('parameter_count', 27824742 if dataset == 'trashnet' else 27827049)
                row = {'model':variant, 'dataset':dataset, 'seed':seed, 'best_epoch':complete['best_epoch'], 'epochs':len(history), 'test_samples':len(pred), 'parameters':parameters, 'accuracy_pct':accuracy*100, 'macro_f1_pct':macro_f1*100}
                per_seed.append(row)
                group.append(row)
            require(len({r['parameters'] for r in group}) == 1, f'{variant}: parameter counts vary')
            s = {'model':variant, 'dataset':dataset, 'n_seeds':3, 'parameters':group[0]['parameters'], 'accuracy_mean_pct':mean(r['accuracy_pct'] for r in group), 'accuracy_std_pct':stdev(r['accuracy_pct'] for r in group), 'macro_f1_mean_pct':mean(r['macro_f1_pct'] for r in group), 'macro_f1_std_pct':stdev(r['macro_f1_pct'] for r in group)}
            summary.append(s)
            actual.extend([round(s['accuracy_mean_pct'],2), round(s['macro_f1_mean_pct'],2)])
            model_row[f'{dataset}_parameters_M'] = f"{s['parameters']/1e6:.2f}"
            model_row[f'{dataset}_accuracy_pct'] = f"{s['accuracy_mean_pct']:.2f}"
            model_row[f'{dataset}_macro_f1_pct'] = f"{s['macro_f1_mean_pct']:.2f}"
        require(actual == EXPECTED[variant], f'{variant}: screenshot mismatch {actual}')
        table.append(model_row)
    out = root/'results'
    write_csv(out/'per_seed.csv', per_seed)
    write_csv(out/'summary.csv', summary)
    write_csv(out/'paper_table.csv', table)
    lines = ['# Paper results', '', 'Means over training seeds 14, 42 and 70. Accuracy and Macro-F1 are percentages.', '', '| Method | Params 6 classes (M) | Params 9 classes (M) | TrashNet Accuracy | TrashNet Macro-F1 | RealWaste Accuracy | RealWaste Macro-F1 |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in table:
        lines.append('| '+' | '.join([row['method'], row['trashnet_parameters_M'], row['realwaste_parameters_M'], row['trashnet_accuracy_pct'], row['trashnet_macro_f1_pct'], row['realwaste_accuracy_pct'], row['realwaste_macro_f1_pct']])+' |')
    lines += ['', 'See `summary.csv` for sample standard deviations (ddof=1), and `per_seed.csv` for each run.', '', 'Each `MODEL/DATASET/seed_N/` directory holds byte-preserved original configuration, completion record, history, predictions and metrics. Recovered checkpoints are stored separately under `pretrained/`; see `../pretrained/manifest.json` for availability. Historical checkpoint and source hashes in these records describe the original runs, not a newly trained model.', '']
    (out/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    report = {'status':'passed', 'runs':30, 'epochs_total':1500, 'prediction_rows':sum(r['test_samples'] for r in per_seed), 'archived_files_checked':len(index), 'checks':['archived SHA256', 'original artifact hashes where included', '30 run identities and completion', '50 contiguous epochs per run', 'prediction IDs and labels identical across models/seeds', 'prediction IDs and labels match packaged fixed test splits', 'accuracy and macro-F1 recomputed from predictions', 'all screenshot metrics match'], 'weights_in_result_directories':False, 'checkpoint_availability':'pretrained/manifest.json; verify separately using scripts/verify_pretrained.py'}
    (root/'provenance/results_validation.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()

"""Integrity and reporting checks for standalone evaluation (no model inference)."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from waste_reliability import evaluation


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class EvaluationIntegrityTests(unittest.TestCase):
    def setUp(self):
        # Keep all temporary writes in a known repository output directory.
        self.temp_parent = PROJECT / 'outputs' / '.evaluation-tests'
        self.temp_parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='case-', dir=self.temp_parent)
        self.root = Path(self.temp.name).resolve()
        self.assertTrue(self.root.is_relative_to(self.temp_parent.resolve()))
        self.addCleanup(self.temp.cleanup)

    def make_checkpoint(self):
        classes = list(evaluation.CLASSES['trashnet'])
        relative = 'pretrained/baseline/trashnet/seed_14/best.pt'
        result_directory = 'results/baseline/trashnet/seed_14'
        path = self.root / relative
        path.parent.mkdir(parents=True)
        path.write_bytes(b'fixture checkpoint bytes; deliberately not a torch pickle')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        config = {
            'model': evaluation.MODEL_NAMES['baseline'], 'dataset': 'trashnet',
            'seed': 14, 'classes': classes, 'epochs': 50,
            'input_size': 224, 'eval_resize': 236,
            'weights_enum': 'ConvNeXt_Tiny_Weights.IMAGENET1K_V1',
        }
        complete = {
            'status': 'completed', 'dataset': 'trashnet', 'seed': 14,
            'requested_epochs': 50, 'epochs_completed': 50, 'best_epoch': 7,
            'classes': classes, 'config': config,
            'artifacts_sha256': {'best.pt': digest},
        }
        entry = {
            'model': 'baseline', 'dataset': 'trashnet', 'seed': 14,
            'original_run': 'convnext_tiny__trashnet__seed14',
            'path': relative, 'result_directory': result_directory,
            'status': 'verified', 'bytes': path.stat().st_size,
            'sha256': digest, 'expected_sha256': digest,
        }
        write_json(self.root / result_directory / 'config.json', config)
        write_json(self.root / result_directory / 'complete.json', complete)
        write_json(self.root / 'pretrained/manifest.json', {'checkpoints': [entry]})
        return path, config, complete, entry

    def test_valid_checkpoint_metadata_without_deserializing_pickle(self):
        path, config, _, _ = self.make_checkpoint()
        meta = evaluation.validate_checkpoint(self.root, 'baseline', 'trashnet', 14)
        self.assertEqual(meta['path'], path.resolve())
        self.assertEqual(meta['config'], config)

    def test_changed_checkpoint_bytes_are_rejected(self):
        path, _, _, _ = self.make_checkpoint()
        # Same file size: catches hash checks omitted in favor of size alone.
        content = bytearray(path.read_bytes())
        content[0] ^= 1
        path.write_bytes(content)
        with self.assertRaises((ValueError, RuntimeError)):
            evaluation.validate_checkpoint(self.root, 'baseline', 'trashnet', 14)

    def test_changed_manifest_expected_hash_is_rejected(self):
        _, _, _, entry = self.make_checkpoint()
        entry['expected_sha256'] = '0' * 64
        write_json(self.root / 'pretrained/manifest.json', {'checkpoints': [entry]})
        with self.assertRaises((ValueError, RuntimeError)):
            evaluation.validate_checkpoint(self.root, 'baseline', 'trashnet', 14)

    def test_checkpoint_for_different_seed_is_rejected(self):
        self.make_checkpoint()
        with self.assertRaises((ValueError, RuntimeError, FileNotFoundError)):
            evaluation.validate_checkpoint(self.root, 'baseline', 'trashnet', 42)

    def test_checkpoint_class_order_is_rejected(self):
        _, config, complete, entry = self.make_checkpoint()
        config['classes'] = list(reversed(config['classes']))
        complete['classes'] = config['classes']
        complete['config'] = config
        write_json(self.root / entry['result_directory'] / 'config.json', config)
        write_json(self.root / entry['result_directory'] / 'complete.json', complete)
        with self.assertRaises((ValueError, RuntimeError)):
            evaluation.validate_checkpoint(self.root, 'baseline', 'trashnet', 14)

    def test_loaded_checkpoint_identity_and_selected_epoch(self):
        _, config, complete, _ = self.make_checkpoint()
        meta = evaluation.validate_checkpoint(self.root, 'baseline', 'trashnet', 14)
        checkpoint = {'config': config, 'epoch': complete['best_epoch'], 'model': {}}
        evaluation.validate_loaded_checkpoint(checkpoint, meta)
        for field, wrong_value in [('dataset', 'realwaste'), ('seed', 42),
                                   ('model', evaluation.MODEL_NAMES['pmam'])]:
            with self.subTest(field=field):
                wrong = copy.deepcopy(checkpoint)
                wrong['config'][field] = wrong_value
                with self.assertRaises((ValueError, RuntimeError)):
                    evaluation.validate_loaded_checkpoint(wrong, meta)
        wrong = copy.deepcopy(checkpoint)
        wrong['epoch'] += 1
        with self.assertRaises((ValueError, RuntimeError)):
            evaluation.validate_loaded_checkpoint(wrong, meta)

    def test_dataset_labels_must_follow_paper_class_order(self):
        # The inventory digest agrees, so rejection must come from semantic
        # class/label validation rather than the earlier file-integrity check.
        row = {'sample_id': 'a', 'path': 'datasets/trashnet/images/a.png',
               'dataset': 'trashnet', 'class_name': 'cardboard', 'label': 1,
               'group_id': 'ga', 'sha256': 'a' * 64,
               'pixel_sha256': 'b' * 64, 'phash': 'c' * 16, 'split': 'test'}
        manifest = self.root / 'datasets/trashnet/manifest.csv'
        manifest.parent.mkdir(parents=True)
        with manifest.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        write_json(self.root / 'datasets/dataset_inventory.json', {
            'datasets': {'trashnet': {'manifest_sha256':
                                     hashlib.sha256(manifest.read_bytes()).hexdigest()}}})
        with self.assertRaisesRegex(ValueError, 'fixed class order'):
            evaluation.load_test_rows(self.root, 'trashnet')

    def test_archive_directories_and_ancestors_cannot_be_outputs(self):
        for name in ('results', 'pretrained', 'datasets', 'src', 'configs',
                     'provenance', 'scripts', 'docs', 'third_party'):
            with self.subTest(name=name):
                protected = self.root / name
                protected.mkdir()
                with self.assertRaises((ValueError, FileExistsError)):
                    evaluation.validate_output(self.root, protected / 'new-evaluation')
        for path in (self.root, self.root.parent):
            with self.assertRaises((ValueError, FileExistsError)):
                evaluation.validate_output(self.root, path)

    def test_nonempty_output_is_rejected_without_changes(self):
        output = self.root / 'outputs' / 'evaluation'
        output.mkdir(parents=True)
        sentinel = output / 'keep.txt'
        sentinel.write_text('must survive', encoding='utf-8')
        with self.assertRaises((ValueError, FileExistsError)):
            evaluation.validate_output(self.root, output)
        self.assertEqual(sentinel.read_text(encoding='utf-8'), 'must survive')

    def test_symlink_to_archive_cannot_bypass_output_guard(self):
        target = self.root / 'pretrained'
        target.mkdir()
        output_parent = self.root / 'outputs'
        output_parent.mkdir()
        link = output_parent / 'redirect'
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as error:
            self.skipTest('Directory symlinks unavailable on this platform: ' + str(error))
        with self.assertRaises((ValueError, FileExistsError)):
            evaluation.validate_output(self.root, link / 'new-evaluation')


class EvaluationComparisonTests(unittest.TestCase):
    def setUp(self):
        self.classes = list(evaluation.CLASSES['trashnet'])
        self.rows = [
            {'sample_id': 'a', 'label': 0, 'class_name': self.classes[0], 'group_id': 'ga'},
            {'sample_id': 'b', 'label': 1, 'class_name': self.classes[1], 'group_id': 'gb'},
        ]
        self.logits = [[2., 0., 0., 0., 0., 0.], [0., 0., 3., 0., 0., 0.]]
        self.reference = []
        for row, logits in zip(self.rows, self.logits):
            reference = dict(row)
            reference['prediction'] = max(range(6), key=lambda index: logits[index])
            reference.update({f'logit_{index}': value for index, value in enumerate(logits)})
            self.reference.append(reference)

    def test_reference_rows_are_matched_by_sample_id(self):
        compared = evaluation.compare_predictions(self.rows, self.logits,
                                                  self.reference[::-1], self.classes, True)
        self.assertEqual(compared['changed_predictions'], 0)
        self.assertEqual(compared['prediction_agreement'], 1.0)
        self.assertEqual(compared['max_abs_logit_difference'], 0.0)
        self.assertEqual(compared['metric_deltas']['accuracy'], 0.0)

    def test_changed_prediction_produces_a_delta_not_a_fake_match(self):
        logits = copy.deepcopy(self.logits)
        logits[1] = [0., 3., 0., 0., 0., 0.]
        compared = evaluation.compare_predictions(self.rows, logits, self.reference,
                                                  self.classes, True)
        self.assertEqual(compared['changed_predictions'], 1)
        self.assertEqual(compared['prediction_agreement'], .5)
        self.assertEqual(compared['metric_deltas']['accuracy'], .5)
        self.assertAlmostEqual(compared['metric_deltas']['macro_f1'], 1/6)

    def test_partial_comparison_uses_same_reference_subset(self):
        compared = evaluation.compare_predictions(self.rows[:1], self.logits[:1],
                                                  self.reference, self.classes, False)
        self.assertFalse(compared['full_test'])
        self.assertEqual(compared['reference_metrics']['n_samples'], 1)
        self.assertEqual(compared['metric_deltas']['accuracy'], 0.0)
        with self.assertRaises((ValueError, RuntimeError)):
            evaluation.compare_predictions(self.rows[:1], self.logits[:1],
                                           self.reference, self.classes, True)

    def test_missing_or_duplicate_reference_ids_are_rejected(self):
        for reference in (self.reference[:1], [self.reference[0], self.reference[0]]):
            with self.subTest(reference=reference):
                with self.assertRaises((ValueError, RuntimeError)):
                    evaluation.compare_predictions(self.rows, self.logits,
                                                   reference, self.classes, True)

    def test_reference_labels_must_match_manifest(self):
        reference = copy.deepcopy(self.reference)
        reference[0]['label'] = 1
        with self.assertRaises((ValueError, RuntimeError)):
            evaluation.compare_predictions(self.rows, self.logits, reference,
                                           self.classes, True)

    def test_nonfinite_logits_and_inconsistent_reference_are_rejected(self):
        logits = copy.deepcopy(self.logits)
        logits[0][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'logits'):
            evaluation.compare_predictions(self.rows, logits, self.reference,
                                           self.classes, True)
        reference = copy.deepcopy(self.reference)
        reference[0]['prediction'] = 1  # contradicts the archived argmax
        with self.assertRaisesRegex(ValueError, 'logits'):
            evaluation.compare_predictions(self.rows, self.logits, reference,
                                           self.classes, True)


class EvaluationSummaryTests(unittest.TestCase):
    def records(self):
        return [{'model': 'baseline', 'dataset': 'trashnet', 'seed': seed,
                 'full_test': True, 'n_samples': 377, 'accuracy': score,
                 'macro_f1': score, 'macro_precision': score, 'macro_recall': score,
                 'loss': 1. - score}
                for seed, score in zip((14, 42, 70), (.8, .9, 1.))]

    def test_mean_and_sample_sd_use_seeds_as_replicates(self):
        summary, = evaluation.summarize(self.records())
        self.assertEqual(summary['n_seeds'], 3)
        # Each seed predicts the same test set; this is not 1,131 test images.
        self.assertEqual(summary['n_samples'], 377)
        self.assertAlmostEqual(summary['accuracy_mean'], .9)
        self.assertAlmostEqual(summary['accuracy_std'], .1)
        self.assertAlmostEqual(summary['macro_f1_std'], .1)
        self.assertAlmostEqual(summary['loss_std'], .1)

    def test_single_seed_has_no_estimated_seed_variability(self):
        summary, = evaluation.summarize(self.records()[:1])
        self.assertEqual(summary['n_seeds'], 1)
        self.assertIsNone(summary['accuracy_std'])
        self.assertIsNone(summary['macro_f1_std'])

    def test_partial_and_full_results_cannot_be_pooled(self):
        records = self.records()
        records[-1].update(full_test=False, n_samples=10)
        summaries = evaluation.summarize(records)
        self.assertEqual(len(summaries), 2)
        full, = [row for row in summaries if row['full_test']]
        partial, = [row for row in summaries if not row['full_test']]
        self.assertEqual((full['n_seeds'], full['n_samples']), (2, 377))
        self.assertEqual((partial['n_seeds'], partial['n_samples']), (1, 10))
        self.assertAlmostEqual(full['accuracy_mean'], .85)

    def test_repeated_seed_cannot_fake_replicates(self):
        row = self.records()[0]
        with self.assertRaisesRegex(ValueError, 'Duplicate seeds'):
            evaluation.summarize([row, dict(row)])


if __name__ == '__main__':
    unittest.main()

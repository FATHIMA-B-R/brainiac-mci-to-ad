"""CPU integration checks; no checkpoint or real patient data required."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/mci_conversion.py'


def write(path, fields, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


class PipelineTests(unittest.TestCase):
    def test_evaluation_and_validation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            features, split = root/'features.csv', root/'split.csv'
            fr, sr = [], []
            for i in range(40):
                label = i % 2
                fr.append(dict(subject_id=f's{i}', label=label, Feature_0=label*2+i/100, Feature_1=i%3))
                sr.append(dict(subject_id=f's{i}', label=label, class_name='MCIc' if label else 'MCInc',
                               dev_or_test='dev' if i<32 else 'test', val_fold=(i//2)%8 if i<32 else -1))
            write(features, list(fr[0]), fr)
            write(split, list(sr[0]), sr)
            base = [sys.executable, str(SCRIPT), 'evaluate', '--features', str(features), '--split', str(split), '--c-values', '.1']
            subprocess.run(base+['--output', str(root/'dev'), '--dev-only'], check=True, capture_output=True)
            self.assertFalse((root/'dev/test_predictions.csv').exists())
            subprocess.run(base+['--output', str(root/'final')], check=True, capture_output=True)
            with (root/'final/test_predictions.csv').open() as f:
                pred = list(csv.DictReader(f))
            self.assertEqual([r['subject_id'] for r in pred], [f's{i}' for i in range(32,40)])
            for row in pred:
                mean = sum(float(row[f'fold_{k}_probability']) for k in range(8))/8
                self.assertAlmostEqual(float(row['probability']), mean)
            self.assertEqual(json.loads((root/'final/test_metrics.json').read_text())['n_test'], 8)
            # A label disagreement must stop before creating an output directory.
            sr[0]['label'] = 1
            write(split, list(sr[0]), sr)
            result = subprocess.run(base+['--output', str(root/'bad')], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('label mismatch', result.stderr)
            self.assertFalse((root/'bad').exists())

    def test_manifest_rejects_missing_and_ambiguous_images(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'MCIc').mkdir()
            cohort = root/'cohort.csv'
            write(cohort, ['subject_id', 'label'], [dict(subject_id='s1', label=1)])
            cmd = [sys.executable, str(SCRIPT), 'manifest', '--cohort', str(cohort), '--root', str(root), '--output', str(root/'manifest.csv')]
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            (root/'MCIc/s1_0000.nii.gz').touch()
            subprocess.run(cmd, check=True, capture_output=True)
            (root/'MCIc/s1.nii.gz').touch()
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()

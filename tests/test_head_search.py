import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).resolve().parents[1]/'scripts/search_attentive_head.py'
spec = importlib.util.spec_from_file_location('head_search', path)
search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search)


class SearchTests(unittest.TestCase):
    def test_search_and_final_commands_separate_test_access(self):
        args = argparse.Namespace(tokens=Path('tokens.pt'), split=Path('split.csv'),
                                  neurovfm_root=Path('nv'), split_seed=21, epochs=30,
                                  patience=10, device='cpu')
        params = dict(lr=.0001, hidden_dim=128, weight_decay=.01, drop_rate=.2)
        trial = search.command(args, params, Path('trial'))
        final = search.command(args, params, Path('final'), development=False)
        self.assertIn('--dev-only', trial)
        self.assertNotIn('--dev-only', final)
        for flag in ('--lr','--hidden-dim','--weight-decay','--drop-rate','--split'):
            self.assertEqual(trial[trial.index(flag)+1], final[final.index(flag)+1])

    def test_objective_requires_complete_development_folds(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'summary.json'
            data=dict(dev_only=True, fold_val_metrics=[dict(auc=.6)]*8, mean_cv_val_auc=.6,
                      ensemble_test_auc=.99)
            p.write_text(json.dumps(data))
            self.assertAlmostEqual(search.score_summary(p),.6)
            for change in [dict(dev_only=False),dict(fold_val_metrics=[dict(auc=.6)]*7),
                           dict(mean_cv_val_auc=.9),dict(fold_val_metrics=[dict(auc=float('nan'))]*8)]:
                p.write_text(json.dumps(data|change))
                with self.assertRaises(ValueError):search.score_summary(p)

    def test_initial_architecture_coverage(self):
        candidates = search.initial_candidates()
        self.assertEqual(len(candidates), 27)
        covered = {(r['hidden_dim'],len(r['mlp_dims'].split(','))) for r in candidates[:17]}
        self.assertEqual(covered, {(w,d) for w in (32,64,128,256) for d in (1,2,3)})
        self.assertEqual(candidates[0]['mlp_dims'],'256')
        self.assertEqual(candidates[0]['hidden_dim'],256)
        self.assertEqual(candidates[0]['head_norm'],0)
        self.assertEqual(candidates[0]['batch_size'],4)

    def test_adapter_passes_architecture_and_preserves_evaluation_loader(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as d:
            trainer=Path(d)/'trainer.py'
            trainer.write_text("""
import sys
def ClassifyThenAggregate(**kwargs): return kwargs
def DataLoader(**kwargs): return kwargs
def main():
    head=ClassifyThenAggregate(hidden_dim=32, mlp_hidden_dims=[32], use_norm=False)
    assert head['mlp_hidden_dims']==[32,32,32]
    assert head['use_norm'] is True
    assert DataLoader(shuffle=True,batch_size=4)['batch_size']==16
    assert DataLoader(shuffle=False,batch_size=4)['batch_size']==4
    assert '--dev-only' in sys.argv
""")
            adapter=path.with_name('attentive_search_adapter.py')
            subprocess.run([sys.executable,str(adapter),'--trainer-source',str(trainer),
                            '--head-depth','3','--head-norm','1','--batch-size','16',
                            '--dev-only'],check=True,capture_output=True)

    def test_tapered_candidates_are_new_and_complete(self):
        candidates = search.initial_candidates()
        new = candidates[-10:]
        expected = {'256,128', '256,64', '256,32', '128,64', '128,32', '64,32',
                    '256,128,64', '256,128,32', '256,64,32', '128,64,32'}
        self.assertEqual({r['mlp_dims'] for r in new}, expected)
        self.assertFalse(expected & {r['mlp_dims'] for r in candidates[:-10]})
        for r in new:
            self.assertEqual(r['hidden_dim'],int(r['mlp_dims'].split(',')[0]))

    def test_tapered_forwarding(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as d:
            trainer=Path(d)/'trainer.py'
            trainer.write_text("""
def ClassifyThenAggregate(**kwargs): return kwargs
def DataLoader(**kwargs): return kwargs
def main():
    head=ClassifyThenAggregate(hidden_dim=256)
    assert head['mlp_hidden_dims']==[128,64,32]
    assert head['hidden_dim']==256
""")
            cmd=[sys.executable,str(path.with_name('attentive_search_adapter.py')),
                 '--trainer-source',str(trainer),'--mlp-dims','128,64,32']
            subprocess.run(cmd,check=True,capture_output=True)
            cmd[-1]='32,64'
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)

    def test_content_hash_changes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input';p.write_text('before');first=search.digest(p)
            p.write_text('after');self.assertNotEqual(first, search.digest(p))


if __name__=='__main__':unittest.main()

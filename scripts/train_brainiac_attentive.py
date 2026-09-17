"""Run the existing NeuroVFM attentive probe on frozen BrainIAC token bags."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def prepare_records(records, rows):
    indexed = {r['study_id']: r for r in records}
    if len(indexed) != len(records):
        raise ValueError('Duplicate embedding IDs')
    if not rows or len({r['subject_id'] for r in rows}) != len(rows):
        raise ValueError('Empty split or duplicate split IDs')
    import torch
    selected = []
    for row in rows:
        sid = row['subject_id']
        label = int(row['label'])
        if sid not in indexed or label not in (0, 1) or label != int(indexed[sid]['label']):
            raise ValueError(f'Missing embedding or label mismatch: {sid}')
        part, fold = row['dev_or_test'], int(row['val_fold'])
        if part not in ('dev', 'test') or (part == 'test' and fold != -1) or (part == 'dev' and fold not in range(8)):
            raise ValueError(f'Invalid partition/fold: {sid}')
        tokens = indexed[sid]['embedding']
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.shape[0] < 2 or tokens.shape[1] != 768 or not torch.isfinite(tokens).all():
            raise ValueError(f'{sid}: expected finite [N_tokens, 768] with N_tokens > 1; re-extract with --tokens')
        selected.append(dict(study_id=sid, label=label, embedding=tokens.float(), dev_or_test=part,
                             split='test' if part == 'test' else 'train', fold=fold))
    if {r['fold'] for r in selected if r['dev_or_test']=='dev'} != set(range(8)):
        raise ValueError('All eight development folds are required')
    for fold in range(8):
        for population in ([r for r in selected if r['dev_or_test']=='dev' and r['fold']==fold],
                           [r for r in selected if r['dev_or_test']=='dev' and r['fold']!=fold]):
            if {r['label'] for r in population} != {0, 1}:
                raise ValueError(f'Fold {fold}: both classes required for training and validation')
    if {r['label'] for r in selected if r['dev_or_test']=='test'} != {0,1}:
        raise ValueError('Held-out test must contain both classes')
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokens', type=Path, required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--neurovfm-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split-seed', type=int, default=0, help='Original split seed for logging only; 0 means unspecified. No new split is generated')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=.05)
    p.add_argument('--hidden-dim', type=int, default=256)
    p.add_argument('--drop-rate', type=float, default=.1)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--grad-clip', type=float, default=1.)
    p.add_argument('--device', default=None)
    p.add_argument('--head-depth', type=int, choices=(1, 2, 3), default=1)
    p.add_argument('--head-norm', type=int, choices=(0, 1), default=0)
    p.add_argument('--batch-size', type=int, choices=(4, 8, 16), default=4)
    p.add_argument('--dev-only', action='store_true')
    args = p.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    trainer = args.neurovfm_root.resolve()/'examples/train_attentive_probe_cv.py'
    if not trainer.is_file():
        raise FileNotFoundError(trainer)
    if args.output.exists():
        raise ValueError('Choose a new output directory to preserve previous results')
    with args.split.open() as f:
        rows = list(csv.DictReader(f))
    selected = prepare_records(torch.load(args.tokens, map_location='cpu', weights_only=False), rows)
    args.output.mkdir(parents=True)
    packed = args.output.resolve()/'brainiac_tokens_with_splits.pt'
    torch.save(selected, packed)
    fingerprint = dict(test_ids=sorted(r['study_id'] for r in selected if r['dev_or_test']=='test'),
                       val_by_fold={str(f): sorted(r['study_id'] for r in selected if r['dev_or_test']=='dev' and r['fold']==f) for f in range(8)})
    (args.output/'split_reference.json').write_text(json.dumps(fingerprint, indent=2))
    (args.output/'split_used.csv').write_bytes(args.split.read_bytes())
    (args.output/'brainiac_run.json').write_text(json.dumps(dict(
        encoder='BrainIAC frozen final patch tokens', trainer=str(trainer),
        trainer_sha256=hashlib.sha256(trainer.read_bytes()).hexdigest(),
        split_sha256=hashlib.sha256(args.split.read_bytes()).hexdigest(),
        token_source=str(args.tokens.resolve()), settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}), indent=2))
    cmd = [sys.executable, str(Path(__file__).with_name('attentive_search_adapter.py').resolve()),
           '--trainer-source', str(trainer), '--head-depth', str(args.head_depth),
           '--head-norm', str(args.head_norm), '--batch-size', str(args.batch_size), '--embeddings', str(packed), '--output-dir', str(args.output.resolve()),
           '--mcinc-mode', 'all', '--num-folds', '8', '--seeds', '42']
    for key in ('split_seed','epochs','lr','weight_decay','hidden_dim','drop_rate','patience','grad_clip'):
        cmd += ['--'+key.replace('_','-'), str(getattr(args,key))]
    if args.device:
        cmd += ['--device', args.device]
    if args.dev_only:
        cmd += ['--dev-only']
    env = dict(os.environ)
    env['PYTHONPATH'] = str(args.neurovfm_root.resolve()) + os.pathsep + env.get('PYTHONPATH','')
    subprocess.run(cmd, env=env, check=True)
    if not args.dev_only:
        result = torch.load(args.output/'seed_42/results.pt', map_location='cpu', weights_only=False)
        test = [r for r in selected if r['dev_or_test']=='test']
        if len(test) != len(result['test_probs']) or [r['label'] for r in test] != result['test_labels']:
            raise ValueError('Test prediction ordering/labels do not match input')
        with (args.output/'test_predictions.csv').open('w',newline='') as f:
            w=csv.writer(f); w.writerow(['subject_id','label','probability'])
            w.writerows((r['study_id'],r['label'],prob) for r,prob in zip(test,result['test_probs']))
        negatives = [(r,prob) for r,prob in zip(test,result['test_probs']) if r['label']==0]
        metrics = {k:v for k,v in result.items() if k.startswith('ensemble_test_') or k in ('ci_lower','ci_upper')}
        metrics['ensemble_test_specificity'] = sum(prob < .5 for _,prob in negatives)/len(negatives)
        (args.output/'test_metrics.json').write_text(json.dumps(metrics,indent=2))


if __name__ == '__main__':
    main()

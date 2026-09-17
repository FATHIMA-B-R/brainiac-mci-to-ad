"""Optuna HPO with a head-width search; frozen tokens and development-only CV."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

SPACE = {'lr': [1e-5, 3e-4], 'weight_decay': [1e-4, .1],
         'hidden_dim': [32, 64, 128, 256], 'drop_rate': [0., .1, .2, .3, .5],
         'head_depth': [1, 2, 3], 'head_norm': [0, 1], 'batch_size': [4, 8, 16]}


def initial_candidates():
    # Baseline first, then all 12 width/depth combinations under one fixed setup.
    baseline = dict(lr=3e-4, weight_decay=.05, hidden_dim=256, drop_rate=.1,
                    head_depth=1, head_norm=0, batch_size=4)
    candidates = [baseline]
    for width in SPACE['hidden_dim']:
        for depth in SPACE['head_depth']:
            candidates.append(dict(baseline, lr=1e-4, hidden_dim=width, head_depth=depth))
    # Isolate weaker/stronger regularization, normalization and batching on a small head.
    base = dict(baseline, lr=1e-4, hidden_dim=64)
    candidates.extend([dict(base, weight_decay=.001, drop_rate=0.),
                       dict(base, weight_decay=.1, drop_rate=.3),
                       dict(base, head_norm=1), dict(base, batch_size=8)])
    return candidates


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def score_summary(path):
    data = json.loads(path.read_text())
    folds = data.get('fold_val_metrics', [])
    if data.get('dev_only') is not True or len(folds) != 8:
        raise ValueError('Expected development-only results for eight folds')
    values = [float(f['auc']) for f in folds]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('Invalid fold AUROC')
    score = sum(values) / 8
    if not math.isclose(score, float(data['mean_cv_val_auc']), abs_tol=1e-8):
        raise ValueError('Mean AUROC does not match fold results')
    return score


def command(args, params, output, development=True):
    cmd = [sys.executable, str(Path(__file__).with_name('train_brainiac_attentive.py').resolve()),
           '--tokens', str(args.tokens.resolve()), '--split', str(args.split.resolve()),
           '--neurovfm-root', str(args.neurovfm_root.resolve()), '--output', str(output.resolve()),
           '--split-seed', str(args.split_seed), '--epochs', str(args.epochs),
           '--patience', str(args.patience), '--grad-clip', '1.0']
    for k, v in params.items():
        cmd += ['--' + k.replace('_', '-'), str(v)]
    if args.device:
        cmd += ['--device', args.device]
    if development:
        cmd += ['--dev-only']
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('tokens', 'split', 'neurovfm-root', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--trials', type=int, default=32, help='Total trial budget, including previous and failed trials')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--split-seed', type=int, default=0)
    p.add_argument('--search-seed', type=int, default=42)
    p.add_argument('--device', default=None)
    args = p.parse_args()
    if args.trials < 1 or args.epochs < 1 or args.patience < 0:
        p.error('Trials/epochs must be positive; patience must be nonnegative')
    import optuna
    trainer = args.neurovfm_root / 'examples/train_attentive_probe_cv.py'
    sources = [args.tokens, args.split, trainer,
               args.neurovfm_root/'neurovfm/models/mil.py',
               Path(__file__).with_name('train_brainiac_attentive.py'),
               Path(__file__).with_name('attentive_search_adapter.py'), Path(__file__)]
    fingerprint = dict(inputs={str(f.resolve()): digest(f) for f in sources},
                       space=SPACE, epochs=args.epochs, patience=args.patience,
                       split_seed=args.split_seed, search_seed=args.search_seed,
                       device=args.device, python=sys.version, optuna=optuna.__version__)
    args.output.mkdir(parents=True, exist_ok=True)
    reference = args.output/'search_config.json'
    if reference.exists():
        if json.loads(reference.read_text()) != fingerprint:
            raise ValueError('Search inputs/code/settings changed. Use a new output directory.')
    else:
        if any(args.output.iterdir()):
            raise ValueError('Output is nonempty without search provenance; use a new directory')
        reference.write_text(json.dumps(fingerprint, indent=2))
    study = optuna.create_study(study_name='attentive_head', direction='maximize',
        storage='sqlite:///' + str((args.output/'study.sqlite3').resolve()), load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.search_seed, n_startup_trials=5),
        pruner=optuna.pruners.NopPruner())
    if not study.trials:
        for parameters in initial_candidates():
            study.enqueue_trial(parameters)
    # Resume completed trials, but never silently treat an interrupted training run as complete.
    if any(t.state == optuna.trial.TrialState.RUNNING for t in study.trials):
        raise RuntimeError('A trial is RUNNING (active or interrupted). Do not launch two searches in one directory. '
                           'If interrupted, preserve this study and start a new output directory.')

    def export():
        study.trials_dataframe().to_csv(args.output/'trials.csv', index=False)
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not complete:
            return
        best = study.best_trial
        (args.output/'best_params.json').write_text(json.dumps(dict(
            trial=best.number, mean_development_auroc=best.value, parameters=best.params,
            note='Selection score, not an independent performance estimate. Test not evaluated.'), indent=2))
        final = command(args, best.params, args.output/'final_test', development=False)
        (args.output/'run_selected_final.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n' + shlex.join(final) + '\n')

    def objective(trial):
        params = dict(lr=trial.suggest_float('lr', *SPACE['lr'], log=True),
                      weight_decay=trial.suggest_float('weight_decay', *SPACE['weight_decay'], log=True),
                      hidden_dim=trial.suggest_categorical('hidden_dim', SPACE['hidden_dim']),
                      drop_rate=trial.suggest_categorical('drop_rate', SPACE['drop_rate']),
                      head_depth=trial.suggest_categorical('head_depth', SPACE['head_depth']),
                      head_norm=trial.suggest_categorical('head_norm', SPACE['head_norm']),
                      batch_size=trial.suggest_categorical('batch_size', SPACE['batch_size']))
        out = args.output/f'trial_{trial.number:04d}'
        with (args.output/f'trial_{trial.number:04d}.log').open('w') as log:
            subprocess.run(command(args, params, out), stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = out/'seed_42/dev_summary.json'
        value = score_summary(summary)
        diagnostics = json.loads(summary.read_text())
        fold_scores = [float(f['auc']) for f in diagnostics['fold_val_metrics']]
        trial.set_user_attr('val_auc_sd', (sum((v-value)**2 for v in fold_scores)/8)**.5)
        trial.set_user_attr('folds_below_chance', sum(v < .5 for v in fold_scores))
        trial.set_user_attr('best_epochs', [r['best_epoch'] for r in diagnostics['fold_training_logs']])
        trial.set_user_attr('output_directory', str(out.resolve()))
        # Each wrapper packs the same source tokens; remove only its redundant generated copy.
        (out/'brainiac_tokens_with_splits.pt').unlink(missing_ok=True)
        return value

    finished_or_started = sum(t.state != optuna.trial.TrialState.WAITING for t in study.trials)
    remaining = max(0, args.trials - finished_or_started)
    try:
        if remaining:
            study.optimize(objective, n_trials=remaining, n_jobs=1,
                           callbacks=[lambda study, trial: export()])
    finally:
        export()
    print(f'Search saved in {args.output}. Held-out test was not evaluated.')
    print('Review best_params.json and trials.csv before explicitly running run_selected_final.sh.')


if __name__ == '__main__':
    main()

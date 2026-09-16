"""Frozen BrainIAC feature extraction and paired MCI conversion evaluation."""
import argparse
import csv
import json
from pathlib import Path
import sys


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def manifest(args):
    rows = read_csv(args.cohort)
    ids = [r['subject_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate cohort subject IDs')
    output = []
    for r in rows:
        label = int(r['label'])
        if label not in (0, 1):
            raise ValueError('Labels must be 0 or 1')
        group = 'MCIc' if label else 'MCInc'
        subject = r['subject_id']
        if Path(subject).name != subject:
            raise ValueError(f'Invalid subject ID: {subject}')
        candidates = [args.root / group / f'{subject}{suffix}.nii.gz' for suffix in ('_0000', '')]
        matches = [p for p in candidates if p.is_file()]
        if len(matches) != 1:
            raise ValueError(f'{subject}: expected exactly one processed image, found {matches}. Check filename mapping.')
        output.append(dict(subject_id=subject, label=label, image_path=str(matches[0].resolve())))
    write_csv(args.output, ['subject_id', 'label', 'image_path'], output)
    print(f'Saved {len(output)} verified image paths to {args.output}')


def extract(args):
    import numpy as np
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from dataset import get_validation_transform
    from load_brainiac import load_brainiac
    rows = read_csv(args.manifest)
    if not rows or len({r['subject_id'] for r in rows}) != len(rows):
        raise ValueError('Manifest must be nonempty with unique subject IDs')
    for r in rows:
        if not Path(r['image_path']).is_file() or int(r['label']) not in (0, 1):
            raise ValueError(f'Invalid manifest row: {r}')
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    model = load_brainiac(str(args.checkpoint), device).eval()
    transform = get_validation_transform()
    output = []
    with torch.inference_mode():
        for i, r in enumerate(rows):
            image = transform({'image': r['image_path']})['image'].unsqueeze(0).to(device)
            if args.tokens:
                # Use the complete final token sequence; model(image) selects token 0.
                tokens = model.backbone(image)[0].squeeze(0).detach().float().cpu()
                if tokens.ndim != 2 or tokens.shape[0] < 2 or tokens.shape[1] != 768 or not torch.isfinite(tokens).all():
                    raise ValueError(f'Invalid token sequence for {r["subject_id"]}: {tokens.shape}')
                output.append(dict(study_id=r['subject_id'], label=int(r['label']), embedding=tokens))
                print(f'[{i+1}/{len(rows)}] {r["subject_id"]}: {list(tokens.shape)}', flush=True)
                continue
            features = model(image).detach().cpu().numpy().reshape(-1)
            if features.shape != (768,) or not np.isfinite(features).all():
                raise ValueError(f'Invalid features for {r["subject_id"]}')
            output.append(dict(subject_id=r['subject_id'], label=int(r['label']),
                               **{f'Feature_{j}': float(v) for j, v in enumerate(features)}))
            print(f'[{i+1}/{len(rows)}] {r["subject_id"]}', flush=True)
    if args.tokens:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output, args.output)
        print(f'Saved frozen token sequences to {args.output}')
        return
    write_csv(args.output, ['subject_id', 'label'] + [f'Feature_{j}' for j in range(768)], output)
    print(f'Saved features with IDs to {args.output}')


def evaluate(args):
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, confusion_matrix, accuracy_score
    import joblib
    import sklearn
    import hashlib

    if not args.c_values or any(not np.isfinite(c) or c <= 0 for c in args.c_values):
        raise ValueError('C values must be finite and positive')
    features = read_csv(args.features)
    if not features or len({r['subject_id'] for r in features}) != len(features):
        raise ValueError('Feature IDs must be nonempty and unique')
    indexed = {r['subject_id']: r for r in features}
    fields = sorted((k for k in features[0] if k.startswith('Feature_')), key=lambda k: int(k.split('_')[1]))
    if not fields:
        raise ValueError('No Feature_* columns found')
    splits = read_csv(args.split)
    ids = [r['subject_id'] for r in splits]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Split IDs must be nonempty and unique')
    for r in splits:
        sid = r['subject_id']
        if sid not in indexed or int(r['label']) != int(indexed[sid]['label']):
            raise ValueError(f'Missing features or label mismatch: {sid}')
        if int(r['label']) not in (0, 1):
            raise ValueError('Expected labels 0/1')
        if r['dev_or_test'] not in ('dev', 'test'):
            raise ValueError('Expected dev/test partitions')
        fold = int(r['val_fold'])
        if (r['dev_or_test'] == 'test' and fold != -1) or (r['dev_or_test'] == 'dev' and fold < 0):
            raise ValueError(f'Invalid fold assignment: {sid}')
    dev = [r for r in splits if r['dev_or_test'] == 'dev']
    test = [r for r in splits if r['dev_or_test'] == 'test']
    if not dev or not test:
        raise ValueError('Both development and test partitions are required')
    X = np.array([[float(indexed[r['subject_id']][k]) for k in fields] for r in dev])
    y = np.array([int(r['label']) for r in dev])
    folds = np.array([int(r['val_fold']) for r in dev])
    unique = sorted(set(folds.tolist()))
    if unique != list(range(8)):
        raise ValueError(f'Expected eight validation folds 0..7; found {unique}')
    if not np.isfinite(X).all():
        raise ValueError('Nonfinite development features')
    for fold in unique:
        if len(set(y[folds == fold])) != 2 or len(set(y[folds != fold])) != 2:
            raise ValueError(f'Both labels required in train/validation fold {fold}')
    args.output.mkdir(parents=True, exist_ok=False)

    def model(c):
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight='balanced',
                             solver='lbfgs', max_iter=3000, random_state=42))

    # Promote nonconvergence to an error rather than reporting unreliable results.
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('error', category=ConvergenceWarning)
    tuning = []
    for c in sorted(set(args.c_values)):
        scores = []
        for fold in unique:
            m = model(c).fit(X[folds != fold], y[folds != fold])
            scores.append(roc_auc_score(y[folds == fold], m.predict_proba(X[folds == fold])[:, 1]))
        tuning.append(dict(C=c, mean_validation_auroc=float(np.mean(scores)), sd_validation_auroc=float(np.std(scores))))
    best = max(tuning, key=lambda r: r['mean_validation_auroc'])['C']
    write_csv(args.output / 'development_tuning.csv', list(tuning[0]), tuning)
    meta = dict(selected_C=best, training_seed=42, class_weight='balanced', threshold=0.5,
                sklearn_version=sklearn.__version__, development_subjects=len(dev), test_subjects=len(test),
                split_sha256=hashlib.sha256(args.split.read_bytes()).hexdigest(),
                features_sha256=hashlib.sha256(args.features.read_bytes()).hexdigest(),
                test_evaluated=not args.dev_only,
                note='Validation scores used for selection are development results, not independent performance estimates.')
    (args.output / 'run.json').write_text(json.dumps(meta, indent=2))
    write_csv(args.output / 'split_used.csv', list(splits[0]), splits)
    if args.dev_only:
        print(f'Development tuning complete. Selected C={best}; test not evaluated.')
        return
    Xt = np.array([[float(indexed[r['subject_id']][k]) for k in fields] for r in test])
    yt = np.array([int(r['label']) for r in test])
    if not np.isfinite(Xt).all() or len(set(yt)) != 2:
        raise ValueError('Test features must be finite and both labels present')
    predictions = []
    for fold in unique:
        m = model(best).fit(X[folds != fold], y[folds != fold])
        joblib.dump(m, args.output / f'fold_{fold}.joblib')
        predictions.append(m.predict_proba(Xt)[:, 1])
    p = np.mean(predictions, axis=0)
    tn, fp, fn, tp = confusion_matrix(yt, p >= .5, labels=[0, 1]).ravel()
    metrics = dict(auroc=float(roc_auc_score(yt, p)), auprc=float(average_precision_score(yt, p)),
                   balanced_accuracy=float(balanced_accuracy_score(yt, p >= .5)),
                   accuracy=float(accuracy_score(yt, p >= .5)), sensitivity=float(tp/(tp+fn)),
                   specificity=float(tn/(tn+fp)), threshold=0.5, n_test=len(test))
    (args.output / 'test_metrics.json').write_text(json.dumps(metrics, indent=2))
    output = [dict(subject_id=r['subject_id'], label=int(r['label']), probability=float(p[i]),
                   **{f'fold_{f}_probability': float(predictions[j][i]) for j, f in enumerate(unique)}) for i, r in enumerate(test)]
    write_csv(args.output / 'test_predictions.csv', list(output[0]), output)
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('manifest')
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(func=manifest)
    p = sub.add_parser('extract')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default=None)
    p.add_argument('--tokens', action='store_true', help='Save all final patch tokens as .pt for attentive probing')
    p.set_defaults(func=extract)
    p = sub.add_parser('evaluate')
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New directory; existing directories are rejected')
    p.add_argument('--c-values', type=float, nargs='+', default=[0.001, 0.01, 0.1, 1.0])
    p.add_argument('--dev-only', action='store_true')
    p.set_defaults(func=evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

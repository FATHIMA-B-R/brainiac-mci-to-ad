"""Attach actual NeuroVFM test/fold membership to a labeled cohort CSV."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.cohort.open() as handle:
        rows = list(csv.DictReader(handle))
    ids = [r['subject_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate cohort subject IDs')
    reference = json.loads(args.reference.read_text())
    assignments = {}

    def add(subjects, partition, fold):
        for subject in subjects:
            if subject in assignments:
                raise ValueError(f'Duplicate/overlapping reference subject: {subject}')
            assignments[subject] = (partition, fold)

    if not reference['test_ids'] or len(reference['val_by_fold']) < 2:
        raise ValueError('Expected held-out test subjects and at least two validation folds')
    add(reference['test_ids'], 'test', -1)
    for fold, subjects in reference['val_by_fold'].items():
        if int(fold) < 0 or not subjects:
            raise ValueError('Expected nonempty folds with nonnegative integer indices')
        add(subjects, 'dev', int(fold))
    if set(ids) != set(assignments):
        raise ValueError(
            f'Cohort/reference mismatch: missing from reference={sorted(set(ids)-set(assignments))}; '
            f'extra in reference={sorted(set(assignments)-set(ids))}'
        )
    if args.output.resolve() in (args.cohort.resolve(), args.reference.resolve()):
        raise ValueError('Output must not overwrite an input')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['subject_id', 'label', 'class_name', 'dev_or_test', 'val_fold'])
        writer.writeheader()
        for row in rows:
            partition, fold = assignments[row['subject_id']]
            writer.writerow({k: row[k] for k in ('subject_id', 'label', 'class_name')} | {'dev_or_test': partition, 'val_fold': fold})
    print(f'Saved {len(rows)} subjects with exact reference assignments to {args.output}')


if __name__ == '__main__':
    main()

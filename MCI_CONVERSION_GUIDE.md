# BrainIAC frozen-feature MCI conversion comparison

Run from the BrainIAC repository root with your existing environment activated.
Labels are converter (MCIc)=1 and non-converter (MCInc)=0. This workflow uses the
repository's foundation backbone, not its pretrained healthy-control/MCI head.
It uses the existing validation transforms and backbone feature readout unchanged.
That readout selects output token index 0; this workflow does not change pooling.

## 1. Build the image manifest after both classes finish preprocessing

```bash
python scripts/mci_conversion.py manifest \
  --cohort comparison_splits/cohort_88_251.csv \
  --root /workspace/ad_detection/brainiac_data/nifti_processed \
  --output comparison_splits/brainiac_manifest.csv
```

This requires one final image per subject, named SUBJECT_0000.nii.gz or
SUBJECT.nii.gz in MCIc/ or MCInc/. Missing or ambiguous matches stop the run.
It does not select temporary registered images or skull masks. If your converted
filenames differ, supply a verified manifest with subject_id,label,image_path;
do not guess a subject-to-scan mapping. Confirm these are the same MRI visits
used for NeuroVFM, and visually check preprocessing quality.

## 2. Extract all 339 subjects once, inside the GPU container

```bash
python scripts/mci_conversion.py extract \
  --manifest comparison_splits/brainiac_manifest.csv \
  --checkpoint src/checkpoints/brainiac.ckp \
  --output comparison_splits/brainiac_features_with_ids.csv
```

Change the checkpoint path to its actual location and spelling. This processes
one scan at a time, preserves subject IDs, applies no training augmentation, and
does not override scheduler GPU visibility. It saves output after all scans pass.
The existing model loader requires checkpoint keys prefixed with backbone. and
loads them strictly. Use the trusted foundation checkpoint you downloaded.

## 3. Check the split CSVs

Required inputs (already generated on HPC using apply_reference.py):

- splits_88_88_split21.csv
- splits_88_88_split42.csv
- splits_88_132_split21.csv
- splits_88_132_split42.csv
- splits_88_251.csv

Each needs subject_id,label,class_name,dev_or_test,val_fold. Test fold is -1;
development folds are 0..7. No new subject splits are generated. Each experiment
rejects duplicate IDs, missing features, label mismatches, and invalid fold values.
The source split is copied into its results directory, and input hashes are saved.

## 4. Development-only tuning (recommended first)

```bash
bash scripts/run_mci_experiments.sh \
  comparison_splits/brainiac_features_with_ids.csv \
  results/mci_linear_development --dev-only
```

The classifier is StandardScaler followed by balanced-class L2 logistic regression,
with seed 42 and C candidates 0.001, 0.01, 0.1, 1. Scaling and class weights are
fitted using training subjects only. C is selected by mean validation AUROC across
eight folds; ties favor the smaller C. Validation scores used to select C are not
independent performance estimates. Nonconverged fits stop the run.

## 5. Final evaluation after fixing the development protocol

```bash
bash scripts/run_mci_experiments.sh \
  comparison_splits/brainiac_features_with_ids.csv \
  results/mci_linear_final
```

This repeats the same development selection and fits eight fold models with the
selected C. The test probability is their arithmetic mean. Threshold metrics use
0.5. Do not revise settings based on these test results. Existing result directories
are rejected to avoid overwriting earlier experiments. For an individual run or a
different development C grid, use `python scripts/mci_conversion.py evaluate --help`.

Per experiment outputs:

- development_tuning.csv: C selection scores
- run.json: selected settings, software version, source hashes
- split_used.csv: the exact input assignments
- fold_0.joblib through fold_7.joblib: fitted scaler and classifier per fold
- test_predictions.csv: IDs, labels, per-fold and ensemble probabilities
- test_metrics.json: AUROC, AUPRC, balanced accuracy, accuracy, sensitivity, specificity

The last three output types are only produced in final evaluation mode.
Compare with the NeuroVFM run using the exact corresponding split reference.
For a representation comparison, run the same linear-probe protocol on NeuroVFM
features too. Comparison with NeuroVFM's attentive head is a complete-pipeline
comparison, because the heads differ. Split21 and split42 test sets can overlap;
do not treat their results as independent replicates or choose a split by test score.

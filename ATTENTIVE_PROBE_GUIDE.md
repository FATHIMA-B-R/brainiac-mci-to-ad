# Frozen BrainIAC tokens → eight-fold attentive probe → held-out ensemble

This is the requested attentive workflow. The earlier linear-probe scripts remain
available but are not used here. The BrainIAC encoder is never optimized. Only the
NeuroVFM ClassifyThenAggregate probe is trained, using your existing NeuroVFM
trainer directly. Run in a Python environment supporting both repositories;
the NeuroVFM trainer's existing dependencies must be installed in that environment.

From the BrainIAC repository root, after preprocessing:

```bash
python scripts/mci_conversion.py manifest \
  --cohort comparison_splits/cohort_88_251.csv \
  --root /workspace/ad_detection/brainiac_data/nifti_processed \
  --output comparison_splits/brainiac_manifest.csv

python scripts/mci_conversion.py extract --tokens \
  --manifest comparison_splits/brainiac_manifest.csv \
  --checkpoint src/checkpoints/brainiac.ckp \
  --output comparison_splits/brainiac_tokens.pt

bash scripts/run_attentive_experiments.sh \
  comparison_splits/brainiac_tokens.pt \
  /workspace/ad_detection/neurovfm-mci-ad \
  results/brainiac_attentive
```

Adjust the checkpoint's actual spelling/path. The token export uses the complete
final MONAI ViT output (normally 216 x 768 with the repository's 96-cubed input and
16-cubed patches), not the original wrapper's token-index-zero vector. One vector
per subject is not accepted as an attentive token bag. Extract only once.

The launcher runs all five saved split CSVs with eight folds and training seed 42.
It uses --mcinc-mode all ONLY after selecting and assigning every subject from
the supplied split CSV; thus it does not generate new cohort partitions. It writes
an expected split_reference.json before training, which the NeuroVFM trainer checks.
For full339, split-seed metadata is 0 (unspecified); actual folds are preserved.

Defaults: epochs=30, lr=3e-4, weight_decay=0.05, hidden_dim=256, drop_rate=0.1,
patience=10, grad_clip=1. Match these to the actual NeuroVFM run being compared.
They are current trainer defaults, not verified historical run settings. Append
options to the launcher to change settings globally; for per-experiment settings,
use scripts/train_brainiac_attentive.py directly with --tokens, --split,
--neurovfm-root and --output.

The trainer uses training-fold class weights, validation AUROC checkpoint selection,
and averages eight selected probes' held-out probabilities. There is no encoder
fine-tuning. Optional --dev-only runs omit held-out testing; choose a separate new
output directory for a subsequent final run. Never select settings using test scores.

Outputs include the existing trainer's seed_42 checkpoints/results/training logs,
the exact split reference, source split CSV, trainer hash/settings, plus
subject-level test_predictions.csv and test_metrics.json (final runs only).
AUROC bootstrap confidence intervals come from the existing NeuroVFM trainer.

This reuses the current HPC NeuroVFM training implementation. If historical runs
used different code or hyperparameters, align them before claiming identical
training protocols. Token extraction requires a real checkpoint and processed MRI;
local checks cannot validate HPC GPU compatibility or historical model parity.

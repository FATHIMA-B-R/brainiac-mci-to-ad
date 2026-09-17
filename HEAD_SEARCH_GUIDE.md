# Automated attentive-head search

Branch: codex/attentive-head-search. This adds Optuna TPE hyperparameter search
and a small architecture search over head width (64/128/256), not full NAS over
new network topologies. BrainIAC weights and saved tokens remain unchanged.
The existing NeuroVFM eight-fold trainer runs with --dev-only for EVERY trial.
The objective is mean selected validation AUROC across all eight supplied folds.
Training seed stays 42. Early stopping stays enabled, with fixed patience 10.
No trial pruning across folds: each successful configuration receives all eight.

From the BrainIAC repository root:

```bash
python -m pip install -r requirements-search.txt
python -u scripts/search_attentive_head.py \
  --tokens comparison_splits/brainiac_tokens.pt \
  --split comparison_splits/splits_88_88_split21.csv \
  --neurovfm-root /workspace/ad_detection/neurovfm-mci-ad \
  --output results/head_search/88_88_split21 \
  --split-seed 21 --trials 12
```

Search space: learning rate 1e-5..3e-4 (log), weight decay 1e-4..0.1 (log),
hidden dimension 64/128/256, dropout 0/0.1/0.2/0.3. The existing baseline is
queued first. Twelve trials mean 96 fold-training runs. Epoch limit is 30.
Use tmux. Check trial_0000.log etc. for progress. Trial checkpoints are retained;
redundant per-trial packed token copies are removed after successful scoring.

Repeat for each split CSV with a distinct output directory. Specify split-seed
21 or 42 for the four named experiments; leave it unspecified for full339.
It is metadata only: exact supplied subject assignments are always preserved.

Outputs: persistent study.sqlite3, trials.csv, best_params.json, search_config.json,
per-trial logs/checkpoints, and run_selected_final.sh. No held-out prediction is
computed by search. To explicitly evaluate the selected head after freezing your
protocol:

```bash
bash results/head_search/88_88_split21/run_selected_final.sh
```

That script retrains the selected configuration using the same eight development
folds, then evaluates the ensemble through the existing wrapper. It rejects an
existing final_test directory. Checkpoint selection uses validation data only.

Rerunning search resumes its stored completed trials; --trials is the TOTAL
budget, not additional trials. Increase the budget to extend a completed study.
Inputs, source code, settings, and Optuna version must match its fingerprint.
Sampler state is not persisted, so resumed suggestions need not match an
uninterrupted run. Failed trials consume budget and errors stop the command;
fix environmental failures before restarting. A RUNNING trial after interruption
requires a new output directory; no partial training resume is implemented.
Do not run two processes on the same study directory. Keep SQLite on storage
supporting reliable file locking (check HPC policy for network filesystems).

Preserve the original matched-settings baseline. Selected development scores are
optimistic selection results, not independent evaluation. The already-inspected
test set cannot become untouched again: report further evaluation transparently
as exploratory. For tuned encoder comparisons use equal search spaces/budgets
and identical per-experiment folds for both encoders. Do not pick split seeds
based on their test scores. Different split seeds can have overlapping subjects.

## Expanded small-data search

The current default is **32 trials per experiment** (256 fold fits). The first
17 are predefined: original baseline, all 12 width/depth combinations at lr=1e-4,
and four small-head checks for weak/strong regularization, normalization, and
batch size. Remaining trials use TPE suggestions. All queued configurations are
only covered if your budget is at least 17. Use `--trials 32` in the command above
instead of 12 to use the expanded budget. Do not launch all five studies before
checking the first one's runtime and disk usage.

Current search space supersedes the earlier narrow description:

| Parameter | Values |
|---|---|
| Attention width / MLP hidden width | 32, 64, 128, 256 |
| MLP hidden layer count | 1, 2, 3 |
| Learning rate | 1e-5 to 3e-4, logarithmic |
| Weight decay | 1e-4 to 0.1, logarithmic |
| Head dropout | 0, 0.1, 0.2, 0.3, 0.5 |
| Existing head LayerNorm | Off / on |
| Training batch size | 4, 8, 16 |

Depth changes the token-classification MLP, not the encoder or attention block
count. Dropout uses the existing head's dropout locations; it does not add dropout
after every MLP hidden layer. Normalization applies the head's existing norm_attn
and norm_mlp layers. Batch-size changes affect training only. The separate adapter
loads the original trainer and configures its model factory/data loader; the
NeuroVFM repository is not edited. Baseline architecture settings remain 256/1,
normalization off, batch size 4.

Safeguards: fixed folds and train seed, fold-specific training class weights,
gradient clipping at 1, validation-only checkpoint selection, early stopping,
finite token checks, no encoder updates, no test-based selection. CV fold standard
deviation, below-chance fold count, and selected epochs are recorded as trial
attributes. All folds are run; difficult folds are never dropped. Selection is
still mean validation AUROC, not a score adjusted after seeing test results.

A larger search can overfit the validation folds. More trials are not automatically
better. Keep a fixed budget and freeze the selected protocol before testing.
This search does not guarantee removal of over/underfitting or exhaustive coverage.
As a development diagnostic, compare with the existing frozen-feature linear
baseline; if all heads perform weakly, investigate labels, scan identity, feature
variation, preprocessing and representation suitability before adding complexity.
Training-history AUROC is the existing online training statistic (dropout active,
weights changing within an epoch); do not interpret it as directly comparable to
validation-mode AUROC. This adapter does not change that logging definition.

For a clean confirmation, assess shortlisted settings on additional development
training seeds with a predeclared selection rule before test evaluation. The
current implementation fixes training seed 42; it does not automate multi-seed
confirmation or nested CV. Cross-validation used repeatedly for selection is not
an unbiased generalization estimate. Already observed test results remain exploratory.

## Next run: only new tapered heads

Run this AFTER the previous search finishes and after transferring these updates.
Do not replace scripts while an older search is launching subsequent trials.
Use a new output directory; old studies remain untouched.

```bash
python -u scripts/search_attentive_head.py \
  --tokens comparison_splits/brainiac_tokens.pt \
  --split comparison_splits/splits_88_132_split21.csv \
  --neurovfm-root /workspace/ad_detection/neurovfm-mci-ad \
  --output results/head_search_tapered/88_132_split21 \
  --split-seed 21 --tapered-only --device cuda
```

This queues EXACTLY ten new configurations, without the old baseline/equal-width
trials or adaptive suggestions. Total budget defaults to 10 and cannot exceed 10.
Two hidden layers: 256,128; 256,64; 256,32; 128,64; 128,32; 64,32.
Three hidden layers: 256,128,64; 256,128,32; 256,64,32; 128,64,32.
These exhaust strictly decreasing sequences of length 2/3 from 32/64/128/256.

Each uses lr=1e-4, weight_decay=.05, dropout=.1, normalization off, training
batch size 4, and attention width equal to its first hidden width. The MLP has a
final one-unit output beyond the listed hidden layers. Existing head-depth syntax
remains supported; explicit --mlp-dims overrides it. The adaptive combined search
also supports explicit layouts, with attention width a separate parameter.

Ten trials = 80 fold fits, with frozen tokens, existing fold assignments, seed 42,
30-epoch limit, patience 10. No test evaluation occurs. Child logs now flush
without waiting for their output buffer to fill. Completed trials in this NEW
study are skipped on rerun; interrupted RUNNING trials retain the earlier recovery
limitation. best_params.json selects among these ten only, not the previous study.
Compare its development results with the previous study before choosing finalists.
Existing run_selected_final.sh would evaluate this study's winner, not a combined
winner; do not run it until the cross-study development selection is complete.

### Cross tapered heads with dropout

Use `--tapered-only --tapered-dropouts 0.1 0.2 0.3` to queue all ten tapered
architectures at each dropout: 30 unique configurations / 240 fold fits. Do not
specify --trials unless intentionally limiting this list: the default automatically
covers the full requested grid, with no subsequent adaptive/equal-width trials.
Dropout is the outer loop, so trials 0–9 use 0.1, 10–19 use 0.2, and 20–29 use 0.3.
All other settings remain as specified for the tapered-only run above.

Use a new directory such as `results/head_search_tapered_dropout/88_132_split21`.
If the ten dropout-0.1 tapered trials already finished, request only
`--tapered-dropouts 0.2 0.3` (20 trials / 160 fold fits) to avoid repeating them.
This does not import or auto-deduplicate results from other study directories.
An already running search does not adopt these changes; wait before updating HPC
scripts. Expanded studies require a new directory because the queue and provenance
are different. Encoder tokens do not need re-extraction.

### Fixed optimizer settings (current behavior)

Learning rate and weight decay are now fixed for EVERY queued and adaptive
configuration: lr=0.0003 and weight_decay=0.05. These are the original attentive
trainer's defaults, not scikit-learn defaults or AdamW's library defaults.
This supersedes earlier search ranges and tapered lr=0.0001 described above.
The cosine learning-rate schedule remains active; 0.0003 is the starting LR.
Both fixed values are retained in trial parameters and final-test commands.
Use a new output directory (e.g. head_search_tapered_fixed/88_132_split21) since
optimizer settings changed. Old results are preserved but are different experiments.
The dropout-0.1 configurations from the old tapered lr=0.0001 run are not equivalent
to dropout-0.1 at the new starting LR; include them when doing the fixed-LR comparison.

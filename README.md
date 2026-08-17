# ECH2O Scorched Earth

Reproducible CPU- and CUDA-capable ConvGRU and ConvLSTM experiments for daily,
full-resolution ECH2O bbox outputs. The fixed-window baseline consumes ordered
daily forcing rasters and 13 static covariates, then predicts soil moisture,
AM/PM skin temperature, and AM/PM PLC at the target grid's native resolution.

This repository contains code and lightweight configuration only. Raw ECH2O
rasters, NetCDFs, model checkpoints, QA screens, manifests, predictions, and
figures are deliberately excluded from Git.

## Requirements

Python 3.11+ is recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project needs four regional static sources in addition to each site's
`Spatial/*.asc` files. Export their paths before running scripts (or adapt the
defaults in `src/data/static_contract.py`):

```bash
export ECH2O_TWI_PATH=/path/to/twi.tif
export ECH2O_FAC_PATH=/path/to/fac.vrt
export ECH2O_PSST_PATH=/path/to/psst.tif
export ECH2O_WBDEF_PATH=/path/to/wbdef.tif
```

Set the raw ECH2O root once as well:

```bash
export ECH2O_DATA_ROOT=/path/to/ech2o_ai
```

Copy `.env.example` as a local reference if useful; `.env` is never committed.
Every data-building script accepts this explicit path as
`--data-root "$ECH2O_DATA_ROOT"`, which prevents any hidden machine-specific
path from entering an experiment.

## Data contract

Read [the Phase 1 schema report](docs/phase1_schema_report.md) before using a
new data copy. Targets in `SITE-YYYY_subdaily.nc` use a water-year axis, so
target band zero is **October 1 of YYYY-1**. In contrast, forcing TIFF bands
are calendar-year indexed: band zero is **January 1 of YYYY**. The sources are
joined only by ISO calendar date—not by shared band number—and the supported
training overlap is January 1 through September 30 of YYYY. October--December
target days are retained in QA reports but deliberately excluded from modeling.
Static ASCII rasters are interpreted as EPSG:5070 and must satisfy the
documented one-cell inset relationship.

## Source QA before any split or normalization

Each newly transferred or regenerated source copy must pass this workflow
before it is used to build a train/validation split. It is intentionally
separate from model evaluation: these checks examine raw inputs and raw ECH2O
targets, never predictions. The commands below use the current calendar-aware
contract and audit the usable **January--September** overlap.

1. **Inspect source bundles.** Validate the expected files, target variables,
   static-grid/target-grid relationship, CRS assumptions, and the calendar
   join contract. The schema report is the authoritative list of source-valid
   bundles.

2. **Screen each day.** Create one CSV per source-valid site. This verifies
   the target/forcing ISO-date join, six-forcing availability, target masks,
   grid alignment, and forcing validity over the entire target-support bbox.
   It records target and forcing band indices independently.

3. **Audit raw targets by target variable.** Score each target band separately
   for suspicious seams, edge jumps, corner jumps, blockiness, and high-
   frequency artifacts. `tskin_am` and `tskin_pm` are evaluated more
   sensitively than PLC or soil moisture, whose real spatial gradients can be
   sharp. Narrow, elongated bboxes are marked low-confidence rather than
   automatically rejected.

4. **Build a conservative review queue.** Apply the versioned manual override
   file only after checking it against the new source copy. Automated signals
   create `keep`, `review`, or candidate-exclusion actions; they do not silently
   remove whole sites. Review timelines retain the flagged target indices and
   ISO dates.

5. **Summarize forcing-bbox continuity.** This is a pure forcing check: it
   does not evaluate target values or artifacts. A clean day requires all six
   forcing rasters to be available and valid across the full bbox. The summary
   reports edge-NA counts and the longest contiguous clean run, which is the
   direct eligibility diagnostic for fixed windows and stateful Jan--Sep replay.

6. **Create split-ready decisions.** Combine source-contract status, target
   artifact triage, and daily forcing QA into site- and target-level decision
   tables. Resolve the review queue before constructing the persisted spatial
   split; fit normalization only after that split is frozen.

Example, using a named QA run directory:

```bash
QA_RUN=artifacts/reports/full_source_jan_sep_training_qa_corrected_v3
DAILY_DIR=artifacts/manifests/phase2_daily_calendar_forcing_corrected_v3
SCHEMA=artifacts/schema_reports/phase1_calendar_forcing_corrected_v3.json

python scripts/inspect_phase1_schema.py \
  --data-root "$ECH2O_DATA_ROOT" --output "$SCHEMA"

python scripts/run_phase2_screen.py \
  --schema-report "$SCHEMA" --data-root "$ECH2O_DATA_ROOT" \
  --output-dir "$DAILY_DIR"

python scripts/audit_manifest_target_artifacts.py \
  --schema-report "$SCHEMA" --data-root "$ECH2O_DATA_ROOT" \
  --output-dir "$QA_RUN" --months 1 2 3 4 5 6 7 8 9 --workers 6

python scripts/build_target_artifact_triage.py \
  --daily-audit "$QA_RUN/manifest_target_artifact_daily.csv" \
  --overrides configs/data/target_artifact_manual_overrides_v1.csv \
  --output "$QA_RUN/target_artifact_triage.csv"

python scripts/summarize_forcing_bbox_qa.py \
  --daily-dir "$DAILY_DIR" --output "$QA_RUN/forcing_bbox_qa_summary.csv"

python scripts/build_training_qa_decisions.py \
  --schema-report "$SCHEMA" --triage "$QA_RUN/target_artifact_triage.csv" \
  --daily-dir "$DAILY_DIR" --output-dir "$QA_RUN"

python scripts/build_training_qa_review.py \
  --site-decisions "$QA_RUN/training_qa_site_decisions.csv" \
  --target-decisions "$QA_RUN/training_qa_target_decisions.csv" \
  --triage "$QA_RUN/target_artifact_triage.csv" \
  --daily-artifact-audit "$QA_RUN/manifest_target_artifact_daily.csv" \
  --output-dir "$QA_RUN"
```

The resulting review package contains:

- `training_qa_site_decisions.csv`: source-contract and forcing-day status per site;
- `training_qa_target_decisions.csv`: the action for each site/target pair;
- `training_qa_site_target_artifact_stats.csv`: per-target median, p95, and
  maximum artifact metrics for filtering and visual review;
- `training_qa_include_candidates.csv`, `training_qa_review_sites.csv`, and
  `training_qa_exclude_sites.csv`: concise site-level worklists;
- `target_artifact_candidate_days.csv`: complete target-index/date timelines;
- `forcing_bbox_qa_summary.csv`: Jan--Sep forcing completeness, edge-NA
  diagnostics, and longest clean contiguous run.

Keep the generated QA run directory with the associated source-data version;
it is an input to splitting, not an interchangeable model artifact.

The complete expected directory structure is in
[the input-folder schema](docs/input_folder_schema.md) and
[`configs/data/input_layout_v1.yaml`](configs/data/input_layout_v1.yaml).
At minimum, the raw data root looks like:

```text
/path/to/ech2o_ai/SITE_ID/
  prcp_YYYY.tif ... rmax_YYYY.tif
  Spatial/*.asc
  SITE_ID-YYYY_subdaily.nc
```

## Minimal workflow

All commands run from the repository root.

1. Inspect the data and produce the schema report.

   ```bash
   python scripts/inspect_phase1_schema.py \
     --data-root "$ECH2O_DATA_ROOT" \
     --output artifacts/schema_reports/phase1_schema.json
   ```

2. Rebuild every daily screen under the v2 date contract. This records both
   `target_time_index` and `forcing_time_index`; October--December rows must
   show `forcing_available=False`. Every v2 fixed-window manifest then persists
   `target_time_index`, `forcing_start_index`, and `forcing_end_index`; the
   dataset recomputes and asserts those indices from the ISO dates before it
   reads a raster. This deliberately rejects all historical v1 manifests.

   ```bash
   python scripts/run_phase2_screen.py \
     --schema-report artifacts/schema_reports/phase1_schema.json \
     --data-root "$ECH2O_DATA_ROOT" \
     --output-dir artifacts/manifests/phase2_daily_v2 \
     --overwrite
   ```

3. Construct a new persisted site-level split. The full split
   keeps the external panel out of both train and validation.

   ```bash
   python scripts/build_full_spatial_split.py \
     --daily-dir artifacts/manifests/phase2_daily_v2 \
     --data-root "$ECH2O_DATA_ROOT" \
     --external-manifest artifacts/manifests/external_panel_v1/manifest.csv \
     --output-dir artifacts/manifests/full75_val25_jun_sep_v1
   ```

4. Fit normalization using `dev_train` only and train the bounded ConvGRU.

   ```bash
   python scripts/fit_dev_normalization.py \
     --manifest artifacts/manifests/full75_val25_jun_sep_v1/manifest.csv \
     --output artifacts/normalization/full75_val25_jun_sep_v1.json

   python scripts/train_dev_cpu.py \
     --manifest artifacts/manifests/full75_val25_jun_sep_v1/manifest.csv \
     --normalization artifacts/normalization/full75_val25_jun_sep_v1.json \
     --checkpoint artifacts/checkpoints/full75_val25_jun_sep_bounded_cpu.pt \
     --epochs 15 --threads 32 --seed 20260803 \
     --enforce-physical-bounds --validation-split dev_spatial_val
   ```

The best checkpoint is selected strictly by the site-disjoint validation loss;
the external panel is reserved for final testing and GeoTIFF export.

### CPU or GPU runtime

All main training commands accept `--device auto|cpu|cuda|cuda:N`. `auto` is
the default: it selects the first visible NVIDIA GPU when the installed PyTorch
build has CUDA support, otherwise it runs on CPU exactly as before. The script
name `train_dev_cpu.py` is retained for command-line compatibility; it is not
CPU-only. Training checkpoints are portable between CPU and CUDA machines.

For a GPU run, use the same reproducible manifest and normalization artifact,
but increase batch size only after a short benchmark confirms it fits GPU
memory. `--threads` still controls CPU-side raster preparation, not GPU work.

```bash
python scripts/train_dev_cpu.py \
  --manifest artifacts/manifests/full75_val25_jun_sep_v1/manifest.csv \
  --normalization artifacts/normalization/full75_val25_jun_sep_v1.json \
  --checkpoint artifacts/checkpoints/full75_val25_lstm_seq90_gpu.pt \
  --epochs 15 --batch-size 16 --threads 8 --interop-threads 1 \
  --device cuda --recurrent-cell lstm --enforce-physical-bounds \
  --validation-split dev_spatial_val
```

Verify the installed environment before a large run:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA GPU')"
```

If this reports `False`, install a CUDA-enabled PyTorch build appropriate for
the target machine; the CPU-only environment on Odin will correctly remain on
CPU. The water-year trainer supports the identical `--device cuda` option.

## Fixed-window model selection

The next controlled experiment compares 30-, 60-, and 90-day ConvGRU and
ConvLSTM models on the same persisted 75/25 spatial split.  The exact matrix,
selection rule, report contents, and commands are documented in
[the fixed-window model-selection plan](docs/fixed_window_model_selection_plan.md).

### ConvGRU versus ConvLSTM

Use the same manifest, normalization artifact, batch size, validation split,
and seed for a controlled recurrent-cell comparison. The fixed-window LSTM
starts with a zero state at the beginning of every independently evaluated
30/60/90-day sequence; its gates decide which information to retain *within*
that declared lookback. It must never carry a hidden state between shuffled
samples or sites.

```bash
python scripts/train_dev_cpu.py \
  --manifest artifacts/manifests/full75_val25_jun_sep_v1/manifest.csv \
  --normalization artifacts/normalization/full75_val25_jun_sep_v1.json \
  --checkpoint artifacts/checkpoints/full75_val25_lstm_seq30_b16_cpu.pt \
  --epochs 15 --batch-size 16 --threads 32 --interop-threads 1 \
  --seed 20260803 --recurrent-cell lstm --enforce-physical-bounds \
  --validation-split dev_spatial_val
```

`--recurrent-cell lstm` together with `--enforce-physical-bounds` creates a
true bounded ConvLSTM checkpoint. Checkpoint metadata records
`recurrent_cell: lstm`; verify this before interpreting a run.

#### Continuing fixed-window models

The fixed 30-, 60-, and 90-day checkpoints save both model weights and AdamW
state. Resume the validation-selected `*_best.pt` checkpoint into a new path;
in `--resume-from` mode, `--epochs` means additional epochs. Use the same
site-disjoint validation loss for early stopping, rather than training loss.

```bash
python scripts/train_dev_cpu.py \
  --manifest artifacts/manifests/full75_val25_jun_sep_seq90_v1/manifest.csv \
  --normalization artifacts/normalization/full75_val25_jun_sep_seq90_v1.json \
  --resume-from artifacts/checkpoints/full75_val25_lstm_seq90_b16_cpu_best.pt \
  --checkpoint artifacts/checkpoints/full75_val25_lstm_seq90_b16_e60_cpu.pt \
  --epochs 50 --learning-rate 2e-4 --batch-size 16 --threads 32 \
  --interop-threads 1 --recurrent-cell lstm --enforce-physical-bounds \
  --validation-split dev_spatial_val \
  --early-stopping-patience 8 --min-delta 0.0005
```

Use the corresponding `seq30` or `seq60` manifest and checkpoint names for
the other two controlled continuations. `--early-stopping-patience 8` means
eight consecutive validation epochs without at least `0.0005` improvement in
normalized validation Huber loss; set it to `0` to disable the cap.

### Stateful Jan--September ConvGRU and ConvLSTM

The stateful experiment starts on January 1 and carries the recurrent state
through September 30 of the water-year end calendar year. It resets only at a
site/water-year boundary. For LSTM,
that state is the hidden state plus cell state; for GRU, it is the hidden state.
`--full-bptt` keeps the full Jan--Sep graph connected and performs one optimizer
update per site-year. Without it, the state is still carried forward but its
gradient is detached at each `--chunk-days` boundary (stateful TBPTT).

| Temporal protocol | ConvGRU | ConvLSTM |
| --- | --- | --- |
| Fixed 30 / 60 / 90 days | supported | supported |
| Stateful Jan--Sep (TBPTT) | supported | supported |
| Full Jan--Sep BPTT | supported | supported |

This makes the fair long-memory experiment explicit: use the same Jan--Sep
manifest, normalization, chunk size, seed, loss months, and BPTT mode; vary
only `--recurrent-cell`.

```bash
python scripts/train_water_year_recurrent.py \
  --manifest artifacts/manifests/lstm10_water_year_v1/manifest.csv \
  --normalization artifacts/normalization/lstm10_water_year_v1.json \
  --checkpoint artifacts/checkpoints/lstm10_water_year_full_bptt_cpu.pt \
  --epochs 100 --chunk-days 0 --full-bptt --threads 32 \
  --recurrent-cell lstm
```

`--chunk-days 0` is the strictest full-BPTT protocol: one unchunked Jan--Sep
forward/backward pass per site-year. A positive `--chunk-days` only controls
the implementation loop in full-BPTT mode; it does not detach the recurrent
state or truncate gradients.

Run the matched ConvGRU experiment by changing only the recurrent-cell flag and
checkpoint name:

```bash
python scripts/train_water_year_recurrent.py \
  --manifest artifacts/manifests/lstm10_water_year_v1/manifest.csv \
  --normalization artifacts/normalization/lstm10_water_year_v1.json \
  --checkpoint artifacts/checkpoints/gru10_water_year_full_bptt_cpu.pt \
  --epochs 100 --chunk-days 0 --full-bptt --threads 32 \
  --recurrent-cell gru
```

For the stateful-TBPTT comparison, omit `--full-bptt` from both commands. The
same prediction script replays either checkpoint: model metadata chooses the
GRU or LSTM cell automatically.

#### Continuing a water-year run

Use `--resume-from` with the prior best checkpoint and a new destination path.
`--epochs` means additional epochs in this mode. Current historical checkpoints
contain model weights but not AdamW state, so their first continuation is a
weight warm-start with a fresh optimizer; continuation checkpoints written by
the current trainer save AdamW state and can subsequently resume exactly.

Use validation—not training—loss for stopping. Training loss can continue to
decline while performance on site-disjoint validation sites has stopped
improving. `--early-stopping-patience` caps a run after that many consecutive
validation epochs fail to improve by at least `--min-delta`; `0` (the default)
disables the cap. The best checkpoint is always retained independently of the
last checkpoint.

```bash
python scripts/train_water_year_recurrent.py \
  --manifest artifacts/manifests/full75_val25_water_year_v1/manifest.csv \
  --normalization artifacts/normalization/full75_val25_water_year_v1.json \
  --resume-from artifacts/checkpoints/full75_val25_gru_water_year_full_bptt_cpu_best.pt \
  --checkpoint artifacts/checkpoints/full75_val25_gru_water_year_full_bptt_e30_cpu.pt \
  --epochs 15 --learning-rate 2e-4 --chunk-days 0 --full-bptt \
  --recurrent-cell gru --threads 32 \
  --early-stopping-patience 8 --min-delta 0.0005
```

## Validation diagnostics and homelab site

The validation reports are exhaustive across eligible June--September
site-date rows and valid pixels. The fixed-window dashboard compares model,
month, state, target, and metric. The separate pixel-strata explorer aggregates
pixels in fixed elevation, TPI, TWI, and climatic-water-deficit quartiles.

```bash
python scripts/build_model_comparison_html.py ...
python scripts/build_pixel_strata_html.py ...
python scripts/build_pixel_joint_heatmap_html.py ...
python scripts/publish_dashboard_site.py --reports-dir artifacts/reports --site-dir site
python -m http.server 8080 --directory site
```

The generated `site/` directory is intentionally ignored by Git. It contains
only the HTML reports selected for the homelab deployment; source data,
checkpoints, GeoTIFFs, and other artifacts remain private. See
[homelab deployment notes](docs/homelab_dashboard.md).

### Metric interpretation

All validation metrics compare prediction and ECH2O truth pixel-by-pixel after
excluding nodata and target-QA-invalid pixels. MAE is the mean absolute pixel
error; RMSE is the square root of mean squared pixel error; and bias is mean
prediction minus observation. Correlation, R-squared, and SD ratio are pooled
pixel-pair statistics. SD ratio is `SD(prediction) / SD(observation)`; values
below one indicate smoother-than-observed predictions.

Pixel-strata reports group those same pixels by fixed split-wide static
quantiles. The joint heatmap uses climatic-water-deficit × TPI bins to locate
terrain–climate regimes where a target fails. These are association diagnostics,
not causal attribution; always inspect the displayed pixel count and follow up
with maps or time series for suspicious cells.

## CPU batching and throughput

The model is CPU-oriented and each sample remains a complete site bbox:

```text
[batch, 30 days, 6 forcing channels, height, width]
```

Batching does **not** crop spatial patches, shorten the 30-day sequence, mix
hidden states across samples, or change the sites/dates seen in an epoch. A
recurrent state is independently reset for every sequence. One epoch always
visits every manifest row exactly once.

With `--batch-size N` greater than one, the loader groups shuffled target dates
from the same site into a batch. That uses their common bbox dimensions to
avoid padding small and large sites together. It is an execution optimization:
the CPU processes several complete sequences concurrently using more RAM.

The tradeoff is optimizer behavior, not coverage. For `S` training sequences,
an epoch has approximately `S / N` gradient updates rather than `S` singleton
updates. Compare candidate batch sizes by validation loss, not by epoch number
alone.

Before a full run, measure real hardware throughput without writing a
checkpoint:

```bash
python scripts/train_dev_cpu.py \
  --manifest artifacts/manifests/full75_val25_jun_sep_v1/manifest.csv \
  --normalization artifacts/normalization/full75_val25_jun_sep_v1.json \
  --checkpoint /tmp/unused.pt \
  --benchmark-batches 8 \
  --batch-size 1 \
  --threads 32 \
  --interop-threads 1 \
  --enforce-physical-bounds
```

Repeat with `--batch-size 2`, `4`, `8`, or `16` and compare the reported
`samples_per_second`. Do not assume that all available CPU cores are fastest;
test a few `--threads` values. Omit `--benchmark-batches` for a full run and
use the selected `--batch-size`, `--threads`, and `--interop-threads` values.

## Repository layout

- `src/data`: date, QA, static-grid, manifest, and normalization contracts
- `src/models`: ConvGRU and full-resolution multitask decoder
- `src/training`: masked losses
- `scripts`: reproducible command-line workflows
- `configs`: explicit channel/date/grid contract
- `docs`: schema and scientific decisions
- `artifacts`: local generated outputs, ignored by Git

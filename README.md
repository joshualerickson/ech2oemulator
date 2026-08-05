# ECH2O Scorched Earth

Reproducible CPU-oriented ConvGRU and ConvLSTM experiments for daily,
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
new data copy. The critical date rule is that a file named `SITE-YYYY` begins
on **October 1 of YYYY-1**, not January 1. Static ASCII rasters are interpreted
as EPSG:5070 and must satisfy the documented one-cell inset relationship.

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

2. Run daily QA, then construct a persisted site-level split. The full split
   keeps the external panel out of both train and validation.

   ```bash
   python scripts/build_full_spatial_split.py \
     --daily-dir artifacts/manifests/phase2_daily \
     --data-root "$ECH2O_DATA_ROOT" \
     --external-manifest artifacts/manifests/external_panel_v1/manifest.csv \
     --output-dir artifacts/manifests/full75_val25_jun_sep_v1
   ```

3. Fit normalization using `dev_train` only and train the bounded ConvGRU.

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

### Stateful water-year ConvGRU and ConvLSTM

The water-year experiment starts at October 1 and carries the recurrent state
through the site-year. It resets only at a site/water-year boundary. For LSTM,
that state is the hidden state plus cell state; for GRU, it is the hidden state.
`--full-bptt` keeps the full Oct--Sep graph connected and performs one optimizer
update per site-year. Without it, the state is still carried forward but its
gradient is detached at each `--chunk-days` boundary (stateful TBPTT).

| Temporal protocol | ConvGRU | ConvLSTM |
| --- | --- | --- |
| Fixed 30 / 60 / 90 days | supported | supported |
| Stateful Oct--Sep (TBPTT) | supported | supported |
| Full-year BPTT | supported | supported |

This makes the fair long-memory experiment explicit: use the same water-year
manifest, normalization, chunk size, seed, loss months, and BPTT mode; vary
only `--recurrent-cell`.

```bash
python scripts/train_water_year_recurrent.py \
  --manifest artifacts/manifests/lstm10_water_year_v1/manifest.csv \
  --normalization artifacts/normalization/lstm10_water_year_v1.json \
  --checkpoint artifacts/checkpoints/lstm10_water_year_full_bptt_cpu.pt \
  --epochs 100 --chunk-days 30 --full-bptt --threads 32 \
  --recurrent-cell lstm
```

`--chunk-days` only controls the implementation loop in full-BPTT mode; it
does not detach the recurrent state or truncate gradients.

Run the matched ConvGRU experiment by changing only the recurrent-cell flag and
checkpoint name:

```bash
python scripts/train_water_year_recurrent.py \
  --manifest artifacts/manifests/lstm10_water_year_v1/manifest.csv \
  --normalization artifacts/normalization/lstm10_water_year_v1.json \
  --checkpoint artifacts/checkpoints/gru10_water_year_full_bptt_cpu.pt \
  --epochs 100 --chunk-days 30 --full-bptt --threads 32 \
  --recurrent-cell gru
```

For the stateful-TBPTT comparison, omit `--full-bptt` from both commands. The
same prediction script replays either checkpoint: model metadata chooses the
GRU or LSTM cell automatically.

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

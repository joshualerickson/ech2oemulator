# ECH2O Scorched Earth

Reproducible CPU-oriented ConvGRU experiments for daily, full-resolution ECH2O
bbox outputs. The baseline consumes 30 consecutive daily forcing rasters and
13 static covariates, then predicts soil moisture, AM/PM skin temperature, and
AM/PM PLC at the target grid's native resolution.

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

Copy `.env.example` as a local reference if useful; `.env` is never committed.

## Data contract

Read [the Phase 1 schema report](docs/phase1_schema_report.md) before using a
new data copy. The critical date rule is that a file named `SITE-YYYY` begins
on **October 1 of YYYY-1**, not January 1. Static ASCII rasters are interpreted
as EPSG:5070 and must satisfy the documented one-cell inset relationship.

Raw data root example:

```text
/path/to/ech2o_ai/SITE_ID/
  prcp_YYYY.tif ... rmax_YYYY.tif
  Spatial/*.asc
  SITE_ID-YYYY_subdaily.nc
```

## Minimal workflow

All commands run from the repository root. Replace `/path/to/ech2o_ai` with
the raw data root on the current machine.

1. Inspect the data and produce the schema report.

   ```bash
   python scripts/inspect_phase1_schema.py \
     --data-root /path/to/ech2o_ai \
     --output artifacts/schema_reports/phase1_schema.json
   ```

2. Run daily QA, then construct a persisted site-level split. The full split
   keeps the external panel out of both train and validation.

   ```bash
   python scripts/build_full_spatial_split.py \
     --daily-dir artifacts/manifests/phase2_daily \
     --data-root /path/to/ech2o_ai \
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

## Repository layout

- `src/data`: date, QA, static-grid, manifest, and normalization contracts
- `src/models`: ConvGRU and full-resolution multitask decoder
- `src/training`: masked losses
- `scripts`: reproducible command-line workflows
- `configs`: explicit channel/date/grid contract
- `docs`: schema and scientific decisions
- `artifacts`: local generated outputs, ignored by Git

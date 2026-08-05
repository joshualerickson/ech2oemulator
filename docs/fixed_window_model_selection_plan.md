# Fixed-window ConvGRU / ConvLSTM model-selection plan

## Decision to make

Select a fixed temporal lookback for each recurrent cell using the already
persisted 75/25 site-level spatial split.  The held-out external panel remains
untouched during selection.

This is a controlled comparison of the recurrent cell and temporal context,
not a test of the stateful water-year models.  The latter is tracked as a
separate long-memory experiment.

## Immutable comparison conditions

- Reference site split: `artifacts/manifests/full75_val25_jun_sep_v1`.
- Training dates and validation dates: June through September only.
- Validation: the `dev_spatial_val` sites; never use the seven external-panel
  sites to choose a model.
- Targets, static channels, decoder, loss, physical bounds, seed, optimizer,
  and CPU batch settings remain unchanged.
- Each candidate receives exactly 15 epochs, batch size 16, 32 compute
  threads, and one interop thread.
- Fit a separate normalization artifact from that candidate's `dev_train`
  rows only.  This avoids leaking validation values and correctly reflects the
  different histories included at 30, 60, and 90 days.

## Candidate grid

| Run ID | Recurrent cell | Lookback | Status |
| --- | --- | ---: | --- |
| `gru_seq30` | ConvGRU | 30 days | rerun/complete to the controlled 15-epoch endpoint |
| `gru_seq60` | ConvGRU | 60 days | planned |
| `gru_seq90` | ConvGRU | 90 days | planned |
| `lstm_seq30` | ConvLSTM | 30 days | planned true ConvLSTM rerun |
| `lstm_seq60` | ConvLSTM | 60 days | planned |
| `lstm_seq90` | ConvLSTM | 90 days | planned |

The previously saved `lstm10_jun_sep_bounded` fixed-window checkpoint is not
included: it was created before `train_dev_cpu.py` preserved
`--recurrent-cell lstm` together with physical output bounds, so it is a GRU
checkpoint despite its filename.  Water-year checkpoints explicitly record
`recurrent_cell: lstm` and are valid LSTM experiments.

## Manifest construction

Use `build_fixed_window_comparison_manifest.py` for 60 and 90 days.  It copies
the site-to-split assignment from the 30-day reference manifest rather than
re-drawing a split.  Any site unable to supply a continuous longer history is
listed in its manifest summary; this attrition must be reported alongside the
metrics.

## Model selection and report

For every candidate, retain:

1. per-epoch train and validation normalized Huber loss;
2. the validation-best checkpoint;
3. validation metrics across all June--September rows, overall and grouped by
   target, state, and site: MAE, RMSE, bias, correlation, R-squared, and
   spatial standard-deviation ratio;
4. the manifest and normalization identifiers, seed, model configuration,
   batch settings, epoch time, and parameter count.

Rank candidates by validation loss.  Interpret the ranking only after checking
the per-target report: reject a nominal winner that materially worsens PLC,
soil-moisture bounds, systematic bias, or spatial SD ratio.  The seven-site
external panel is then evaluated once for the chosen GRU and chosen LSTM; it
is not a tuning set.

## After the 15-epoch screen

1. Select the most promising lookback within GRU and within LSTM.
2. Continue each selected checkpoint with `--resume`, evaluating validation
   loss every epoch and stopping after a documented plateau/patience rule.
3. Compare the two final fixed-window winners on the untouched external panel,
   including GeoTIFF diagnostics for representative sites and regimes.
4. Only then decide whether a longer stateful water-year LSTM provides enough
   additional value to justify its substantially higher training cost.


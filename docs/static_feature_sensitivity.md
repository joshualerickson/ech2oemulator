# Static feature sensitivity

This workflow describes **how the trained emulator responds** when one static
input changes while the site, date, and complete 90-day dynamic forcing sequence
are held fixed. It is an explanation diagnostic, not an estimate of a causal
vegetation treatment effect: static features were learned from observational,
correlated landscapes.

The first experiment focuses on the two canopy features in the final model:

- `vcf`: vegetation/canopy cover fraction, constrained to `[0, 1]`;
- `lai`: leaf area index, constrained to `>= 0`.

The selected static value is shifted by -1 and +1 training standard deviation
at every valid pixel. Changes that would leave its physical range are clipped,
and the output records the mean *effective* change after clipping. The other 12
static channels and all forcings remain unchanged.

Run a CPU pilot on 16 evenly site-distributed dates:

```bash
/home/josh.erickson/miniconda3/envs/echo/bin/python \
  scripts/explore_static_sensitivity.py \
  --manifest artifacts/manifests/final_all463_calendar_v3_seq90/manifest.csv \
  --split final_fit_train \
  --checkpoint artifacts/checkpoints/final_all463_calendar_v3_seq90_lstm_e33.pt \
  --normalization artifacts/normalization/final_all463_calendar_v3_seq90.json \
  --output-dir artifacts/feature_exploration/vcf_lai_pilot \
  --features vcf lai --delta-z -1 1 --max-samples 16 --threads 24
```

The sampler rotates its selected site/date examples across June, July, August,
and September (roughly equally in each complete pass), then repeats sites only
when more samples are requested than distinct sites.

Outputs:

- `per_sample_effects.csv`: one row per sampled site/date, feature,
  perturbation, and target. `mean_prediction_change` is the spatial mean of
  `(counterfactual prediction - baseline prediction)` over valid pixels.
- `summary.csv`: distribution of that per-sample mean response across the
  selected site/date contexts.
- `run.json`: exact inputs and interpretation statement.

Interpret feature-response sign and magnitude alongside the effective input
change and site/date context. A positive response for one target does not imply
that mechanically increasing canopy in the real landscape will produce that
response; it describes the emulator’s learned conditional relationship.

## Joint canopy dashboard

The static site includes **Canopy sensitivity**, a compact interactive view of
the joint VCF--LAI experiment. It intentionally publishes aggregated summaries,
not raw inputs, raster predictions, or the per-pixel counterfactual table.

The page uses grouped monthly bars. Do not add monthly values together: each
bar is one month’s mean site/date response in the selected target’s native
units. `dem_q*`, `tpi_q*`, `twi_q*`, and `wbdef_q*` are global
training-pixel quartiles, so they retain pixel-level spatial context but mix
within-site and between-site variation.

The public view presents four selectable canopy configurations:

- coupled low (`VCF -1 SD`, `LAI -1 SD`);
- coupled high (`VCF +1 SD`, `LAI +1 SD`);
- VCF high / LAI low (`VCF +1 SD`, `LAI -1 SD`); and
- VCF low / LAI high (`VCF -1 SD`, `LAI +1 SD`).

This is deliberate: VCF and LAI are very strongly correlated in the final-fit
training pixels. The mixed configurations are shown as individual model
diagnostics, not as ecological results. The page intentionally does not show
calculated cross-contrast or interaction metrics.

Build the page after a completed joint run:

```bash
/home/josh.erickson/miniconda3/envs/echo/bin/python \
  scripts/build_static_sensitivity_html.py \
  --input artifacts/feature_exploration/joint_vcf_lai_64_stratified/joint_canopy_per_sample_strata.csv \
  --strata-json artifacts/feature_exploration/final_all463_static_strata.json \
  --output artifacts/reports/static_feature_sensitivity.html

/home/josh.erickson/miniconda3/envs/echo/bin/python \
  scripts/publish_dashboard_site.py \
  --reports-dir artifacts/reports --site-dir site
```

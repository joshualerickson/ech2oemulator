# Example: using the three reference sites for artifact QA

This is a worked example, not a production exclusion threshold. It uses the
pre-rerun raw NetCDF benchmark outputs and must be recalibrated against the
corrected source data plus a broader clean-control sample.

## What each score means

For one target field on one NetCDF band:

- `row_seam_ratio` is the strongest horizontal boundary jump divided by the
  typical neighboring-pixel jump in that field.
- `column_seam_ratio` is the same calculation for vertical boundaries.
- `edge_gradient_ratio` is the strongest of the top, bottom, left, and right
  boundary jumps divided by the typical internal jump.
- `corner_jump_ratio` compares each corner with its one-pixel-inward diagonal
  neighbor. It is deliberately sensitive to isolated corner defects.

Ratios are target-specific. In particular, nearly uniform PLC fields can have
a very small typical-gradient denominator, so a large PLC ratio is a review
signal, not automatic evidence of an artifact.

## Reference behavior from the benchmark

| Site | Role | Tskin AM seam p50 / p95 | Tskin PM seam p50 / p95 | Interpretation |
| --- | --- | ---: | ---: | --- |
| `CA3982012144020181108` | known severe | 6.57 / 14.96 | 6.31 / 15.59 | Persistent Tskin artifact; exclude those two target channels until repaired. |
| `CA4156412340420210801` | known moderate | 3.41 / 10.92 | 5.57 / 13.87 | Candidate/review case. Its known severe Tskin band is 171; later bands after 250 require targeted review. |
| `CA4198012316420170811` | mostly good comparison | 2.02 / 2.86 | 1.91 / 2.54 | Retain Tskin. Address occasional corner defects with a spatial mask rather than discarding the site. |

The comparison site has no Tskin days with `row_seam_ratio >= 8`; the severe
site has 132 AM and 136 PM days at or above that value. This makes the score
useful as one Tskin screening feature, but it is not a universal cutoff.

## Example decisions

### A. Persistent site-target contamination

Use a *site-target summary* to identify fields that are consistently bad:

```text
example persistent-Tskin review rule
  candidate if:
    row_seam_ratio_p50 >= 6
    AND row_seam_ratio_p95 >= 12
    AND count(row_seam_ratio >= 10) >= 80 days
```

`CA398...` satisfies this pattern for both Tskin targets. Its practical action
is:

```text
artifact_action = exclude_target_channel_all_days
site_id = CA3982012144020181108
targets = [tskin_am, tskin_pm]
```

Do not infer that soil moisture or PLC need the same action. They require their
own target-specific review.

### B. Day-level candidate review

For a site such as `CA415...`, retain every raw score, then create a review
queue rather than immediately excluding every day:

```text
candidate day if:
  target in [tskin_am, tskin_pm]
  AND row_seam_ratio >= 8

priority review if:
  target_time_index == 171
  OR target_time_index >= 250
```

The output must retain both zero-based `target_time_index` and the derived
water-year target date. For this site, band 171 is 2021-03-21 and band 250 is
2021-06-08. After visual confirmation, set one of:

```text
exclude_target_day
exclude_target_channel_all_days
no_action
```

### C. Localized corners or edges

`CA419...` demonstrates why site-level exclusion is too blunt. Its Tskin seam
behavior is clean, while the corner score identifies isolated possible defects.
The correct next action is a pixel-level mask:

```text
if a corner pixel is visually/algorithmically confirmed bad:
  valid_mask[target, y, x] = False
else:
  retain the rest of that target field and site-day
```

The final production artifact screen should write an explicit per-target mask
or mask reference. It must not convert masked pixels to zero, and its training
and validation metric code must use the same mask.

## Recommended rollout after the corrected rerun

1. Recompute these benchmark reports on the repaired versions of all three
   sites.
2. Add at least 20 visually clean controls spanning terrain, state, and target.
3. Compare score distributions by target, using absolute jump magnitude as
   well as ratios for near-flat PLC fields.
4. Freeze a versioned artifact policy and persist its flags/masks in the daily
   QA artifact before any split or normalization step.
5. Keep site-level train/validation assignment independent of artifact flags;
   only eligible target days/pixels enter a persisted manifest.

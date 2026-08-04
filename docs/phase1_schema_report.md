# Phase 1 schema report — ECH2O bbox sources

Inspection date: 2026-07-30  
Source root: `/mnt/alpheus1/zholden/ech2o_projects/ech2o_ai`

## Result

The source contains 518 site/bbox directories. 511 meet the baseline source
contract and are eligible for manifest construction; seven are quarantined,
not repaired or resampled.

Every valid bundle contains six daily forcing stacks, 21 static ASCII rasters,
and five target variables in one subdaily NetCDF. Targets use EPSG:5070 at
240 m, while the bbox dimensions vary by site (target width 11–319; height
10–341). A batching strategy must therefore use padding/collation or
fixed-size model patches; it may not assume one raster shape for all sites.

## Calendar contract (required)

The filename year is the **water-year end year**. NetCDF `time` is unusable
(`units = "unknown"` in the inspected schema), so dates are derived as:

```text
site-YYYY_subdaily.nc band 0 = (YYYY - 1)-10-01
band i = band 0 + i calendar days
```

All joins and manifest rows will use derived ISO calendar dates, never a raw
band number or an assumed January 1 start. Coverage is 2011-10-01 through
2024-09-30: 387 365-step and 124 366-step eligible site-years.

## Channel contract

Dynamic order: `prcp`, `srad`, `tmin`, `tmax`, `rmin`, `rmax`.

Target order: `soilmoisture`, `tskin_am`, `tskin_pm`, `plc_am`, `plc_pm`.
The target fill value is `-1.175494e38`; forcing GeoTIFF nodata is NaN; static
ASCII nodata is `-9999`. Values are to become masks, never zeros.

All 21 static source files are present at every site:
`BClambda`, `ClimZone`, `age`, `albedo`, `ba`, `dem`, `hgt`, `ksat`, `lai`,
`lai_postfire`, `lai_uni`, `ntr`, `poros`, `psi_ae`, `root`, `slope`,
`soildepth`, `theta_r`, `vcf`, `vcf_postfire`, and `vcf_uni`.
The approved baseline selection inherited from the existing project is the
ordered 13-channel set `vcf`, `theta_r`, `BClambda`, `poros`, `psi_ae`, `dem`,
`keff` (resolved from `ksat`), `lai`, `twi`, `fac`, `tpi`, `psst`, `wbdef`.
`lai_postfire` and `vcf_postfire` are intentionally excluded.

## Spatial contract

For 516 sites, each forcing bbox is exactly two cells wider and taller than
the target and the target is its one-cell inset. No clipping is needed:
the canonical target-support window is an index crop of the already supplied
bbox. It must be validated as an exact grid relationship, not obtained through
unrecorded reprojection.

The static ASCII files carry coordinates and 240 m cell size but no CRS tag.
They are approved as EPSG:5070. The loader still asserts their transform
matches the forcing bbox and records the inherited CRS decision in the
manifest.

Two sites break the standard spatial contract and are excluded:
`AZ3150111115820230701` and `NV3957611991320200627`.

## Target masks

The full target-mask scan covered 1,287,390,963 target cells per channel:

| Target | Valid fraction |
| --- | ---: |
| soilmoisture | 99.2173% |
| tskin_am | 95.9670% |
| tskin_pm | 95.9670% |
| plc_am | 100.0000% |
| plc_pm | 100.0000% |

## Quarantined site-years

Five 2024 site-years have 366 target steps but only 274 forcing bands and are
excluded: `AZ3323511112720240901`, `AZ3343911144120240910`,
`AZ3382111178820240518`, `OR4435711775920240711`, and
`WA4667112099220240723`.

## Sequence and split proposal

For a contiguous 30-day baseline, the 511 eligible site-years yield 171,820
candidate sequences before later target-quality screening. Proposed persisted
experiments are: (1) site-disjoint spatial holdout, stratified by state;
(2) temporal holdout within every retained site, with a 30-day embargo at each
boundary; and (3) their intersection for spatiotemporal holdout. Split IDs and
training-only normalization statistics must be stored before training.

## Approved second-stage QA policy

The selected plausibility masks are PLC 0–100%, soil moisture 0–1, and Tskin
−10–70 °C. Values outside these bounds will be masked and logged by site/day;
they will never be clipped. The one-cell target inset and EPSG:5070 inheritance
for ASCII statics are approved.

Detailed machine-readable results, including every site and mask count, are in
`artifacts/schema_reports/phase1_schema.json`.

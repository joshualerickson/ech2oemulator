# Input folder schema

Set one raw-data root and pass it to every data-building command using
`--data-root "$ECH2O_DATA_ROOT"`. The complete machine-readable contract is
[`configs/data/input_layout_v1.yaml`](../configs/data/input_layout_v1.yaml).

```text
$ECH2O_DATA_ROOT/
├── AZ.../                              # one site/bbox directory
│   ├── prcp_YYYY.tif                   # daily forcing stack
│   ├── srad_YYYY.tif
│   ├── tmin_YYYY.tif
│   ├── tmax_YYYY.tif
│   ├── rmin_YYYY.tif
│   ├── rmax_YYYY.tif
│   ├── SITE_ID-YYYY_subdaily.nc        # five target variables
│   └── Spatial/
│       ├── dem.asc
│       ├── vcf.asc
│       ├── theta_r.asc
│       └── ...
└── CA.../
```

`YYYY` is the **water-year end**, and raster/NetCDF band zero is October 1 of
the preceding year. The validation script checks this layout, target names,
daily band counts, the forcing/target one-cell inset, and static alignment:

```bash
python scripts/inspect_phase1_schema.py \
  --data-root "$ECH2O_DATA_ROOT" \
  --output artifacts/schema_reports/phase1_schema.json
```

The bbox-specific ASCII files provide local terrain/soil/vegetation values.
The regional TWI, flow accumulation, PSST, and climatic-deficit rasters live
outside this root. Configure them with the four `ECH2O_*_PATH` variables in
`.env.example`; the loader reprojects them to the target EPSG:5070 grid.

# Homelab diagnostics website

The scientific dashboard source and publishing script stay in this repository.
The generated website is local deployment output, not a Git-tracked model
artifact.  It contains only self-contained HTML diagnostics; raw ECH2O inputs,
GeoTIFFs, NetCDF files, checkpoints, and manifests remain private.

After generating the desired reports, stage the static site:

```bash
/home/josh.erickson/miniconda3/envs/echo/bin/python \
  scripts/publish_dashboard_site.py \
  --reports-dir artifacts/reports \
  --site-dir site
```

Quick local preview:

```bash
python -m http.server 8080 --directory site
```

Then browse to `http://HOSTNAME:8080`.

For a persistent homelab deployment, point an existing Caddy, nginx, or Apache
static-file site at the absolute `site/` directory. Regenerate the dashboards
and rerun the publish command after an experiment completes; there is no
database or application server to maintain.

Do not publish `artifacts/` directly: it includes private or large experiment
products outside the intended HTML reports.

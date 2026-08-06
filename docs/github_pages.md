# Public diagnostics site

The repository can publish a static GitHub Pages site from `public/`. It is a
deliberately restricted release surface: it contains only the generated HTML
diagnostic reports and landing-page interpretation text.

It must never include raw ECH2O rasters, NetCDF files, daily forcings, target
arrays, model checkpoints, model-ready tensors, or private filesystem paths.

## Publishing an updated dashboard

After producing the desired local reports, rebuild the committed public bundle:

```bash
/home/josh.erickson/miniconda3/envs/echo/bin/python \
  scripts/publish_dashboard_site.py \
  --reports-dir artifacts/reports \
  --site-dir public

git add public .github/workflows/deploy-pages.yml docs/github_pages.md
git commit -m "Publish updated diagnostics dashboard"
git push
```

The GitHub Actions workflow deploys `public/` on pushes to `main`. In the
repository settings, set **Pages → Build and deployment → Source** to
**GitHub Actions** once. The project site will then be served at the repository
owner's GitHub Pages URL (for this repository, normally
`https://joshualerickson.github.io/ech2oemulator/`).

The local `site/` directory remains ignored and is still suitable for homelab
hosting. `public/` is the intentional, reviewable public snapshot.

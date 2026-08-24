#!/usr/bin/env python3
"""Stage generated HTML diagnostics as a small static homelab website."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPORTS = (
    ("Recurrent model comparison", "fixed_window_model_comparison.html"),
    ("Pixel terrain and climate strata", "pixel_strata_explorer.html"),
    ("Joint climatic-deficit × TPI heatmap", "pixel_joint_heatmap.html"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=Path("artifacts/reports"))
    parser.add_argument("--site-dir", type=Path, default=Path("site"))
    args = parser.parse_args()
    args.site_dir.mkdir(parents=True, exist_ok=True)
    published: list[tuple[str, str]] = []
    for title, filename in REPORTS:
        source = args.reports_dir / filename
        if not source.exists():
            continue
        shutil.copy2(source, args.site_dir / filename)
        published.append((title, filename))
    if not published:
        raise FileNotFoundError(f"No expected HTML reports found in {args.reports_dir}")
    links = "\n".join(f'<li><a href="{filename}">{title}</a></li>' for title, filename in published)
    (args.site_dir / "index.html").write_text(
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ECH2O emulator diagnostics</title>
<style>body{font:16px system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#17212b;line-height:1.55}a{color:#1769aa}li{margin:12px 0}.muted{color:#52616d}.note{background:#eef4f8;border-left:4px solid #1769aa;padding:12px 16px}table{border-collapse:collapse}td,th{border:1px solid #d7dee5;padding:7px 10px;text-align:left}.nav{display:flex;gap:10px;flex-wrap:wrap;padding:0 0 22px;border-bottom:1px solid #d7dee5;margin-bottom:26px}.nav a{color:#17212b;text-decoration:none;padding:7px 11px;border-radius:999px;background:#edf2f7;font-weight:650}.nav a.active,.nav a:hover{background:#1769aa;color:#fff}</style>
<nav class="nav"><a class="active" href="index.html">Overview</a><a href="fixed_window_model_comparison.html">Model comparison</a><a href="pixel_strata_explorer.html">Terrain &amp; climate strata</a><a href="pixel_joint_heatmap.html">Deficit × TPI</a></nav><h1>ECH2O recurrent emulator diagnostics</h1>
<p class="muted">Public, static validation diagnostics for the current ConvLSTM experiments predicting daily spatial ECH2O outputs.</p>
<h2>What is published</h2><p>Interactive summaries of pixel-level held-out validation metrics, terrain/climate error strata, and model comparisons. No raw rasters, targets, forcing inputs, site files, checkpoints, or model-ready tensors are published.</p>
<h2>Current protocol</h2><table><tr><th>Temporal protocol</th><th>Published model</th></tr><tr><td>Fixed 30, 60, or 90-day forcing lookback</td><td>ConvLSTM</td></tr><tr><td>Stateful Jan.–Sep. full BPTT</td><td>ConvLSTM</td></tr></table>
<div class="note"><strong>Data and comparison note.</strong> Targets use a water-year axis and daily forcings use a calendar-year axis; all modeling joins by ISO date over the shared Jan.–Sep. period. The corrected full-bbox QA retained 477 candidate sites, of which 463 have uninterrupted daily forcings; seven external sites are held out and the persisted modeling split contains 343 training and 114 spatial-validation sites. Fixed-window models share that identical split. Full-BPTT replays the same sites from Jan. 1 through Sep. 30, so it is a long-memory comparison rather than a different data cohort.</div>
<h2>Current reading</h2><p>The dashboard shows only the current corrected-calendar ConvLSTM checkpoints. Select a month, state, target, and metric to compare fixed lookbacks directly; treat the full-BPTT result as the explicit long-memory protocol. All displayed scores are pooled over valid prediction/target pixel pairs, never random pixel samples or averages of site scores.</p>
<h2>Interactive reports</h2><ul>"""
        + links
        + "</ul><p class=\"muted\">Source code, model protocol, and reproducibility documentation are maintained in the accompanying GitHub repository.</p></html>\n"
    )
    print({"site_dir": str(args.site_dir), "reports": [filename for _, filename in published]})


if __name__ == "__main__":
    main()

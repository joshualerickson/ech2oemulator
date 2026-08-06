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
<style>body{font:16px system-ui,sans-serif;max-width:900px;margin:48px auto;padding:0 20px;color:#17212b;line-height:1.55}a{color:#1769aa}li{margin:12px 0}.muted{color:#52616d}.note{background:#eef4f8;border-left:4px solid #1769aa;padding:12px 16px}table{border-collapse:collapse}td,th{border:1px solid #d7dee5;padding:7px 10px;text-align:left}</style>
<h1>ECH2O recurrent emulator diagnostics</h1>
<p class="muted">Public, static validation diagnostics for ConvGRU and ConvLSTM experiments predicting daily spatial ECH2O outputs.</p>
<h2>What is published</h2><p>Interactive summaries of pixel-level held-out validation metrics, terrain/climate error strata, and model comparisons. No raw rasters, targets, forcing inputs, site files, checkpoints, or model-ready tensors are published.</p>
<h2>Protocols represented</h2><table><tr><th>Protocol</th><th>Models</th></tr><tr><td>Fixed 30, 60, or 90-day lookback</td><td>ConvGRU and ConvLSTM</td></tr><tr><td>Stateful Oct.–Sep. full BPTT</td><td>ConvGRU and ConvLSTM</td></tr></table>
<div class="note"><strong>Interpretation note.</strong> Fixed-window reports use the full persisted 75/25 spatial split. Full-water-year BPTT reports use the continuity-valid subset (290 training and 97 validation sites), so compare models most directly within the same temporal protocol and cohort.</div>
<h2>Current reading</h2><p>In the current fixed-window grid, the 90-day ConvLSTM is the strongest overall result. The full-water-year BPTT experiments are retained as an explicit long-memory diagnostic rather than treated as a direct winner over the fixed-window cohort.</p>
<h2>Interactive reports</h2><ul>"""
        + links
        + "</ul><p class=\"muted\">Source code, model protocol, and reproducibility documentation are maintained in the accompanying GitHub repository.</p></html>\n"
    )
    print({"site_dir": str(args.site_dir), "reports": [filename for _, filename in published]})


if __name__ == "__main__":
    main()

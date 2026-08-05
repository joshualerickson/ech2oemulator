#!/usr/bin/env python3
"""Stage generated HTML diagnostics as a small static homelab website."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPORTS = (
    ("Fixed-window model comparison", "fixed_window_model_comparison.html"),
    ("Pixel terrain and climate strata", "pixel_strata_explorer.html"),
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
        "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>ECH2O emulator diagnostics</title>"
        "<style>body{font:16px system-ui,sans-serif;max-width:760px;margin:60px auto;padding:0 20px;color:#17212b}"
        "a{color:#1769aa}li{margin:14px 0}.muted{color:#52616d}</style>"
        "<h1>ECH2O emulator diagnostics</h1>"
        "<p class=\"muted\">Static, locally generated validation diagnostics. "
        "No raw rasters, targets, checkpoints, or model inputs are published here.</p>"
        f"<ul>{links}</ul></html>\n"
    )
    print({"site_dir": str(args.site_dir), "reports": [filename for _, filename in published]})


if __name__ == "__main__":
    main()

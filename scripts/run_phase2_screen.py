#!/usr/bin/env python3
"""Resumably screen every source-valid site-year into one daily QA CSV per site."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
if __package__ in {None, ""}: sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.phase2_qc import iter_daily_screen_rows
from src.data.paths import configured_data_root

def write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--schema-report",type=Path,required=True); p.add_argument("--data-root",type=Path,default=configured_data_root(),help="Raw input root; defaults to ECH2O_DATA_ROOT.")
    p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--site-offset",type=int,default=0); p.add_argument("--site-limit",type=int)
    p.add_argument("--max-cell-days",type=int,default=800000,help="Maximum H*W*days per read; lower for constrained hosts.")
    p.add_argument("--overwrite",action="store_true")
    a=p.parse_args()
    if a.data_root is None: p.error("set ECH2O_DATA_ROOT or pass --data-root")
    schema=json.loads(a.schema_report.read_text())
    sites=[s for s in schema['sites'] if not s['issues']][a.site_offset:]
    if not sites:
        raise ValueError(
            "Schema report contains zero source-valid site bundles. "
            "Do not create a daily-screen artifact or downstream split; verify --data-root points to the ECH2O site directory containing *_subdaily.nc files."
        )
    if a.site_limit is not None: sites=sites[:a.site_limit]
    for n,s in enumerate(sites,1):
        site_id=str(s['site_id']); wy=int(s['water_year_end']); output=a.output_dir/f'{site_id}_{wy}_daily_screen.csv'
        if output.exists() and not a.overwrite:
            print(f'[{n}/{len(sites)}] exists {output.name}', flush=True); continue
        cells=int(s['target_grid']['width'])*int(s['target_grid']['height']); steps=int(s['target_band_counts']['soilmoisture'])
        block=max(1,min(steps,a.max_cell_days//cells)); rows=[]
        for start in range(0,steps,block):
            rows.extend(iter_daily_screen_rows(a.data_root/site_id,wy,start,min(start+block,steps)))
        if len(rows)!=steps: raise AssertionError(f'{site_id}: expected {steps} days, got {len(rows)}')
        write(output,rows); print(f'[{n}/{len(sites)}] wrote {output.name}',flush=True)
if __name__=='__main__': main()

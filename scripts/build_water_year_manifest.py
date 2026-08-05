#!/usr/bin/env python3
"""Collapse a sequence manifest to one Oct--Sep stateful LSTM row per site."""
from __future__ import annotations
import argparse,csv,json
from datetime import date,timedelta
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--source-manifest',type=Path,required=True);p.add_argument('--daily-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 with a.source_manifest.open(newline='') as f: source=list(csv.DictReader(f))
 unique={}
 for r in source: unique.setdefault((r['split'],r['site_id']),r)
 rows=[];excluded=[]
 for (split,site),r in sorted(unique.items()):
  paths=list(a.daily_dir.glob(f'{site}_*_daily_screen.csv'))
  if len(paths)!=1: raise FileNotFoundError(f'Need one daily screen for {site}')
  with paths[0].open(newline='') as f: daily=list(csv.DictReader(f))
  days=[date.fromisoformat(x['date']) for x in daily]
  if any(b-a!=timedelta(days=1) for a,b in zip(days,days[1:])) or not all(x['forcing_valid']=='True' for x in daily): excluded.append({'site_id':site,'split':split,'reason':'forcing_gap_or_invalid_day'});continue
  rows.append({**{k:r[k] for k in ('split','site_id','bbox_id','water_year_end','dynamic_sequence_source','static_source','target_source')},'start_date':daily[0]['date'],'end_date':daily[-1]['date'],'daily_screen_source':str(paths[0])})
 a.output_dir.mkdir(parents=True,exist_ok=True);fields=list(rows[0])
 with (a.output_dir/'manifest.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 if not rows:raise ValueError('No source-valid full water years')
 summary={'source_manifest':str(a.source_manifest),'sites_by_split':{s:sum(r['split']==s for r in rows) for s in sorted({r['split'] for r in rows})},'excluded_sites':excluded,'water_year_rows':len(rows)};(a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
if __name__=='__main__':main()

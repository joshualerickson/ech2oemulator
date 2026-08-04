#!/usr/bin/env python3
"""Build the final contiguous sequence manifest from persisted daily QA files."""
from __future__ import annotations
import argparse,csv,json
from datetime import date,timedelta
from pathlib import Path
def main() -> None:
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--daily-dir',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--sequence-length',type=int,default=30); a=p.parse_args()
 rows=[]
 for path in sorted(a.daily_dir.glob('*_daily_screen.csv')):
  with path.open(newline='') as f: daily=list(csv.DictReader(f))
  by={date.fromisoformat(r['date']):r for r in daily}
  for target,record in by.items():
   history=[by.get(target-timedelta(days=i)) for i in range(a.sequence_length-1,-1,-1)]
   if None in history or not all(r['forcing_valid']=='True' for r in history) or record['target_qa_valid']!='True': continue
   rows.append({'site_id':record['site_id'],'bbox_id':record['bbox_id'],'water_year_end':record['water_year_end'],'start_date':history[0]['date'],'end_date':record['date'],'target_date':record['date'],'sequence_length':a.sequence_length})
 if not rows: raise ValueError('No eligible sequences')
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 print(json.dumps({'manifest_row_count':len(rows),'site_count':len(set(r['site_id'] for r in rows))},indent=2))
if __name__=='__main__': main()

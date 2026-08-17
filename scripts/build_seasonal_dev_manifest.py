#!/usr/bin/env python3
"""Restrict an existing development site split to explicit target months."""
from __future__ import annotations
import argparse,csv,json
from datetime import date,timedelta
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.temporal_contract import TEMPORAL_CONTRACT
def main():
 p=argparse.ArgumentParser(); p.add_argument('--dev-summary',type=Path,required=True); p.add_argument('--daily-dir',type=Path,required=True); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--months',type=int,nargs='+',default=[6,7,8,9,10]); p.add_argument('--sequence-length',type=int,default=30); a=p.parse_args()
 s=json.loads(a.dev_summary.read_text()); groups={'dev_train':s['train_sites'],'dev_spatial_val':s['spatial_val_sites'],'dev_spatial_test':s['spatial_test_sites']}; rows=[]
 for split,sites in groups.items():
  for site in sites:
   paths=list(a.daily_dir.glob(f'{site}_*_daily_screen.csv'))
   if len(paths)!=1: raise FileNotFoundError(f'Need one completed daily screen for {site}')
   with paths[0].open(newline='') as f: daily=list(csv.DictReader(f))
   by={date.fromisoformat(r['date']):r for r in daily}; wy=daily[0]['water_year_end']; root=a.data_root/site
   for target,record in sorted(by.items()):
    if target.month not in a.months: continue
    hist=[by.get(target-timedelta(days=i)) for i in range(a.sequence_length-1,-1,-1)]
    if None in hist or not all(r['forcing_valid']=='True' for r in hist) or record['target_qa_valid']!='True': continue
    rows.append({'split':split,'site_id':site,'bbox_id':site,'water_year_end':wy,'temporal_contract':TEMPORAL_CONTRACT,'start_date':hist[0]['date'],'end_date':record['date'],'target_date':record['date'],'sequence_length':a.sequence_length,'target_time_index':record['target_time_index'],'forcing_start_index':hist[0]['forcing_time_index'],'forcing_end_index':record['forcing_time_index'],'dynamic_sequence_source':str(root),'static_source':str(root/'Spatial'),'target_source':str(root/f'{site}-{wy}_subdaily.nc')})
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 print({'months':a.months,'rows_by_split':{x:sum(r['split']==x for r in rows) for x in groups}})
if __name__=='__main__': main()

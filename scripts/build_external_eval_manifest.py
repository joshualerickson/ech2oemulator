#!/usr/bin/env python3
"""Create a multi-site prediction manifest from completed daily QA screens."""
from __future__ import annotations
import argparse,csv
from datetime import date,timedelta
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.temporal_contract import TEMPORAL_CONTRACT
def main():
 p=argparse.ArgumentParser(); p.add_argument('--daily-dir',type=Path,required=True); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--site-ids',nargs='+',required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--sequence-length',type=int,default=30); a=p.parse_args(); rows=[]
 if a.sequence_length<1:p.error('--sequence-length must be positive')
 for site in a.site_ids:
  path=next(a.daily_dir.glob(f'{site}_*_daily_screen.csv'))
  with path.open(newline='') as f: daily=list(csv.DictReader(f))
  by={date.fromisoformat(r['date']):r for r in daily}; wy=daily[0]['water_year_end']; root=a.data_root/site
  for target,r in by.items():
   h=[by.get(target-timedelta(days=i)) for i in range(a.sequence_length-1,-1,-1)]
   if None in h or not all(x['forcing_valid']=='True' for x in h) or r['target_qa_valid']!='True': continue
   rows.append({'split':'dev_spatial_test','site_id':site,'bbox_id':site,'water_year_end':wy,'temporal_contract':TEMPORAL_CONTRACT,'start_date':h[0]['date'],'end_date':r['date'],'target_date':r['date'],'sequence_length':a.sequence_length,'target_time_index':r['target_time_index'],'forcing_start_index':h[0]['forcing_time_index'],'forcing_end_index':r['forcing_time_index'],'dynamic_sequence_source':str(root),'static_source':str(root/'Spatial'),'target_source':str(root/f'{site}-{wy}_subdaily.nc')})
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print({'sites':len(a.site_ids),'sequence_length':a.sequence_length,'rows':len(rows)})
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Build a prediction-only manifest for one screened site outside training."""
from __future__ import annotations
import argparse,csv
from datetime import date,timedelta
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.temporal_contract import TEMPORAL_CONTRACT
def main():
 p=argparse.ArgumentParser(); p.add_argument('--daily-screen',type=Path,required=True); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--sequence-length',type=int,default=30); a=p.parse_args()
 with a.daily_screen.open(newline='') as f: daily=list(csv.DictReader(f))
 by={date.fromisoformat(r['date']):r for r in daily}; rows=[]; site=daily[0]['site_id']; wy=daily[0]['water_year_end']; root=a.data_root/site
 for target,record in sorted(by.items()):
  history=[by.get(target-timedelta(days=i)) for i in range(a.sequence_length-1,-1,-1)]
  if None in history or not all(r['forcing_valid']=='True' for r in history) or record['target_qa_valid']!='True': continue
  rows.append({'split':'dev_spatial_test','site_id':site,'bbox_id':site,'water_year_end':wy,'temporal_contract':TEMPORAL_CONTRACT,'start_date':history[0]['date'],'end_date':record['date'],'target_date':record['date'],'sequence_length':a.sequence_length,'target_time_index':record['target_time_index'],'forcing_start_index':history[0]['forcing_time_index'],'forcing_end_index':record['forcing_time_index'],'dynamic_sequence_source':str(root),'static_source':str(root/'Spatial'),'target_source':str(root/f'{site}-{wy}_subdaily.nc')})
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 print({'site_id':site,'rows':len(rows),'output':str(a.output)})
if __name__=='__main__': main()

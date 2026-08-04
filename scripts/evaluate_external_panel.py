#!/usr/bin/env python3
"""Score a normalized checkpoint by site/date/target and write triage tables."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
if __package__ in {None,""}:sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from src.data.sequence_dataset import SequenceDataset
from src.data.phase2_qc import DYNAMIC_CHANNELS,TARGET_CHANNELS
from src.data.static_contract import STATIC_CHANNELS
from src.data.transforms import standardize
from src.models.multitask_model import model_from_checkpoint,load_model_state
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--normalization',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--split',default='dev_spatial_test');p.add_argument('--month',type=int,default=8);p.add_argument('--max-days-per-site',type=int,default=1,help='0 evaluates every selected day.');a=p.parse_args()
 stats=json.loads(a.normalization.read_text())['groups'];ds=SequenceDataset(a.manifest,a.split,cache_site_arrays=True);state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);m=model_from_checkpoint(state);load_model_state(m,state);m.eval(); seen={}; rows=[]
 with torch.no_grad():
  for i,r in enumerate(ds.rows):
   if int(r['target_date'][5:7])!=a.month or (a.max_days_per_site and seen.get(r['site_id'],0)>=a.max_days_per_site):continue
   seen[r['site_id']]=seen.get(r['site_id'],0)+1;s=ds[i]; x=standardize(torch.nan_to_num(s['dynamic_sequence']).unsqueeze(0),stats['dynamic'],DYNAMIC_CHANNELS,2);c=standardize(torch.nan_to_num(s['current_forcing']).unsqueeze(0),stats['dynamic'],DYNAMIC_CHANNELS,1);z=standardize(torch.nan_to_num(s['static_stack']).unsqueeze(0),stats['static'],STATIC_CHANNELS,1);q=m(x,c,z)[0]
   mean=torch.tensor([stats['target'][n]['mean'] for n in TARGET_CHANNELS]).view(-1,1,1);std=torch.tensor([stats['target'][n]['std'] for n in TARGET_CHANNELS]).view(-1,1,1);q=(q*std+mean).numpy();y=s['target_stack'].numpy();mask=s['valid_mask'].numpy()
   for j,n in enumerate(TARGET_CHANNELS):
    v=mask[j];e=q[j][v]-y[j][v]; rows.append({'site_id':r['site_id'],'state':r['site_id'][:2],'date':r['target_date'],'target':n,'mae':float(np.mean(abs(e))),'rmse':float(np.sqrt(np.mean(e*e))),'bias':float(np.mean(e)),'sd_ratio':float(np.std(q[j][v])/max(np.std(y[j][v]),1e-6))})
 a.output_dir.mkdir(parents=True,exist_ok=True)
 with (a.output_dir/'per_site_date_target.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary={n:{k:float(np.mean([r[k] for r in rows if r['target']==n])) for k in ('mae','rmse','bias','sd_ratio')} for n in TARGET_CHANNELS};(a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

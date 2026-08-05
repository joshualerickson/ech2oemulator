#!/usr/bin/env python3
"""Fit stateful-LSTM scaling from train sites; forcings use full Oct--Sep history."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
if __package__ in {None,''}:sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.water_year_dataset import WaterYearDataset
from src.data.phase2_qc import DYNAMIC_CHANNELS,TARGET_CHANNELS
from src.data.static_contract import STATIC_CHANNELS
from src.data.transforms import RunningStats,save_stats
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();ds=WaterYearDataset(a.manifest,'dev_train');groups={k:[RunningStats() for _ in names] for k,names in {'dynamic':DYNAMIC_CHANNELS,'static':STATIC_CHANNELS,'target':TARGET_CHANNELS}.items()}
 for i,s in enumerate(ds,1):
  for c,x in enumerate(groups['dynamic']):x.update(s['dynamic'][:,c])
  for c,x in enumerate(groups['static']):x.update(s['static'][c])
  for c,x in enumerate(groups['target']):x.update(s['target'][:,c][s['loss_days']])
  print(f'fitted site {i}/{len(ds)}',flush=True)
 save_stats(a.output,groups,{'dynamic':DYNAMIC_CHANNELS,'static':STATIC_CHANNELS,'target':TARGET_CHANNELS});print(f'wrote {a.output}')
if __name__=='__main__':main()

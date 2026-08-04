#!/usr/bin/env python3
"""Fit train-only normalization from dev-train samples; never read val/test rows."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
if __package__ in {None,""}: sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.sequence_dataset import SequenceDataset
from src.data.phase2_qc import DYNAMIC_CHANNELS,TARGET_CHANNELS
from src.data.static_contract import STATIC_CHANNELS
from src.data.transforms import RunningStats,save_stats
def main() -> None:
 p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--max-samples',type=int,default=0); a=p.parse_args()
 ds=SequenceDataset(a.manifest,'dev_train',cache_site_arrays=True); n=len(ds) if not a.max_samples else min(len(ds),a.max_samples)
 groups={k:[RunningStats() for _ in names] for k,names in {'dynamic':DYNAMIC_CHANNELS,'static':STATIC_CHANNELS,'target':TARGET_CHANNELS}.items()}
 seen_static=set()
 for i in range(n):
  sample=ds[i]
  for c,stat in enumerate(groups['dynamic']): stat.update(sample['dynamic_sequence'][:,c])
  for c,stat in enumerate(groups['target']): stat.update(sample['target_stack'][c])
  if sample['site_id'] not in seen_static:
   for c,stat in enumerate(groups['static']): stat.update(sample['static_stack'][c])
   seen_static.add(sample['site_id'])
  if (i+1)%100==0: print(f'fitted {i+1}/{n}',flush=True)
 save_stats(a.output,groups,{'dynamic':DYNAMIC_CHANNELS,'static':STATIC_CHANNELS,'target':TARGET_CHANNELS}); print(f'wrote {a.output}')
if __name__=='__main__': main()

#!/usr/bin/env python3
"""Load and batch real dev-manifest sequences without training a model."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
if __package__ in {None, ""}: sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch.utils.data import DataLoader, Subset
from src.data.sequence_dataset import SequenceDataset, pad_collate
def main() -> None:
 p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--split',default='dev_train'); p.add_argument('--batch-size',type=int,default=2); p.add_argument('--max-samples',type=int,default=4); a=p.parse_args()
 dataset=SequenceDataset(a.manifest,a.split)
 indices=[]; seen=set()
 for index,row in enumerate(dataset.rows):
  if row['site_id'] not in seen:
   indices.append(index); seen.add(row['site_id'])
  if len(indices)==a.max_samples: break
 subset=Subset(dataset,indices)
 batch=next(iter(DataLoader(subset,batch_size=a.batch_size,collate_fn=pad_collate,num_workers=0)))
 assert batch['dynamic_sequence'].ndim==5 and batch['target_stack'].shape[1]==5 and batch['static_stack'].shape[1]==len(dataset._static_cache[next(iter(dataset._static_cache))])
 print({key: tuple(value.shape) for key,value in batch.items() if hasattr(value,'shape')})
if __name__=='__main__': main()

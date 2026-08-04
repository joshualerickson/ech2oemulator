#!/usr/bin/env python3
from __future__ import annotations
import argparse,sys
from pathlib import Path
if __package__ in {None,""}: sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader,Subset
from src.data.sequence_dataset import SequenceDataset,pad_collate
from src.models.multitask_model import ConvGRUMultitask
from src.training.losses import masked_huber
def finite(x): return torch.nan_to_num(x,nan=0.0,posinf=0.0,neginf=0.0)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--epochs',type=int,default=1); p.add_argument('--max-samples',type=int,default=4); a=p.parse_args()
 ds=SequenceDataset(a.manifest,'dev_train'); loader=DataLoader(Subset(ds,range(min(len(ds),a.max_samples))),batch_size=1,collate_fn=pad_collate,num_workers=0)
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=ConvGRUMultitask().to(device); optim=torch.optim.AdamW(model.parameters(),lr=1e-3); losses=[]
 model.train()
 for _ in range(a.epochs):
  for b in loader:
   pred=model(finite(b['dynamic_sequence']).to(device),finite(b['current_forcing']).to(device),finite(b['static_stack']).to(device)); loss=masked_huber(pred,b['target_stack'].to(device),b['valid_mask'].to(device).bool()); optim.zero_grad(); loss.backward(); optim.step(); losses.append(float(loss.detach().cpu()))
 a.output.parent.mkdir(parents=True,exist_ok=True); torch.save({'model_state':model.state_dict(),'losses':losses,'parameter_count':sum(p.numel() for p in model.parameters())},a.output)
 print({'device':str(device),'steps':len(losses),'first_loss':losses[0],'last_loss':losses[-1],'parameters':sum(p.numel() for p in model.parameters())})
if __name__=='__main__': main()

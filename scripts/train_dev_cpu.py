#!/usr/bin/env python3
"""Resumable normalized CPU training for the ConvGRU dev baseline."""
from __future__ import annotations
import argparse,json,random,sys,time
from collections import defaultdict
from pathlib import Path
if __package__ in {None,""}: sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader, Sampler
from src.data.sequence_dataset import SequenceDataset,pad_collate
from src.data.phase2_qc import DYNAMIC_CHANNELS,TARGET_CHANNELS
from src.data.static_contract import STATIC_CHANNELS
from src.data.transforms import standardize
from src.models.multitask_model import ConvGRUMultitask,load_model_state
from src.training.losses import masked_huber
from src.data.target_qc import TARGET_PLAUSIBILITY_RANGES
from src.utils.runtime import resolve_device,to_device


class SiteBatchSampler(Sampler[list[int]]):
 """Batch dates from one bbox, avoiding padding of variable site shapes."""
 def __init__(self,dataset:SequenceDataset,batch_size:int,seed:int,shuffle:bool)->None:
  self.batch_size=batch_size; self.seed=seed; self.shuffle=shuffle; self.epoch=0; self.groups=defaultdict(list)
  for index,row in enumerate(dataset.rows): self.groups[row['site_id']].append(index)
 def __iter__(self):
  rng=random.Random(self.seed+self.epoch); self.epoch+=1; batches=[]
  for indices in self.groups.values():
   current=indices.copy()
   if self.shuffle: rng.shuffle(current)
   batches.extend(current[start:start+self.batch_size] for start in range(0,len(current),self.batch_size))
  if self.shuffle: rng.shuffle(batches)
  yield from batches
 def __len__(self)->int: return sum((len(indices)+self.batch_size-1)//self.batch_size for indices in self.groups.values())


def site_loader(dataset:SequenceDataset,batch_size:int,seed:int,shuffle:bool,pin_memory:bool=False)->DataLoader:
 """Use ordinary shuffled singleton batches or full-bbox site batches."""
 if batch_size==1: return DataLoader(dataset,batch_size=1,shuffle=shuffle,collate_fn=pad_collate,num_workers=0,pin_memory=pin_memory,generator=torch.Generator().manual_seed(seed))
 return DataLoader(dataset,batch_sampler=SiteBatchSampler(dataset,batch_size,seed,shuffle),collate_fn=pad_collate,num_workers=0,pin_memory=pin_memory)


def main():
 p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--normalization',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--epochs',type=int,default=5); p.add_argument('--threads',type=int,default=32); p.add_argument('--interop-threads',type=int,default=1); p.add_argument('--device',default='auto',help='auto (default), cpu, cuda, or cuda:N.'); p.add_argument('--batch-size',type=int,default=1,help='>1 batches same-site dates without spatial padding.'); p.add_argument('--benchmark-batches',type=int,default=0,help='Run this many train batches, report throughput, and exit without a checkpoint.'); p.add_argument('--recurrent-cell',choices=('gru','lstm'),default='gru',help='Use lstm only for a controlled ConvGRU comparison.'); p.add_argument('--seed',type=int,default=20260803); p.add_argument('--enforce-physical-bounds',action='store_true',help='Constrain soil moisture to [0,1] and PLC to [0,100] in the model head.'); p.add_argument('--validation-split',default=None,help='Optional site-level manifest split, e.g. dev_spatial_val.'); p.add_argument('--validation-every',type=int,default=1); p.add_argument('--validation-max-samples',type=int,default=0,help='0 validates every row of the split.'); p.add_argument('--resume',action='store_true'); a=p.parse_args()
 if a.batch_size<1 or a.interop_threads<1: p.error('--batch-size and --interop-threads must be positive')
 torch.set_num_threads(a.threads); torch.set_num_interop_threads(a.interop_threads); torch.manual_seed(a.seed); device=resolve_device(a.device); stats=json.loads(a.normalization.read_text())['groups']; ds=SequenceDataset(a.manifest,'dev_train',cache_site_arrays=True); loader=site_loader(ds,a.batch_size,a.seed,True,pin_memory=device.type=='cuda')
 val_loader=None
 if a.validation_split:
  val_ds=SequenceDataset(a.manifest,a.validation_split,cache_site_arrays=True)
  if a.validation_max_samples: val_ds.rows=val_ds.rows[:a.validation_max_samples]
  val_loader=site_loader(val_ds,a.batch_size,a.seed,False,pin_memory=device.type=='cuda')
 # Keep the requested recurrent cell when output bounds are also enabled.
 # Omitting the default GRU key preserves compatibility with earlier GRU
 # checkpoints when resuming them.
 model_config={} if a.recurrent_cell=='gru' else {'recurrent_cell':a.recurrent_cell}
 if a.enforce_physical_bounds:
  constrained={'soilmoisture','plc_am','plc_pm'}; lower=[]; upper=[]
  for name in TARGET_CHANNELS:
   if name in constrained:
    raw_lower,raw_upper=TARGET_PLAUSIBILITY_RANGES[name]; mean,std=stats['target'][name]['mean'],stats['target'][name]['std']; lower.append((raw_lower-mean)/std); upper.append((raw_upper-mean)/std)
   else: lower.append(0.0); upper.append(0.0)
  model_config.update({'output_lower':lower,'output_upper':upper})
 model=ConvGRUMultitask(**model_config).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3); start=0; losses=[]; epoch_metrics=[]; best_validation_loss=float('inf')
 if a.resume and a.checkpoint.exists():
  s=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
  if s.get('model_config',{})!=model_config: raise ValueError('Checkpoint model configuration differs from requested output-bound configuration')
  load_model_state(model,s); opt.load_state_dict(s['optimizer_state']); start=s['epoch']; losses=s['losses']; epoch_metrics=s.get('epoch_metrics',[]); best_validation_loss=s.get('best_validation_loss',float('inf'))
 if a.benchmark_batches:
  model.train(); started=time.perf_counter(); sample_count=0; measured=[]
  for number,b in enumerate(loader,1):
   x=to_device(standardize(torch.nan_to_num(b['dynamic_sequence']),stats['dynamic'],DYNAMIC_CHANNELS,2),device); cur=to_device(standardize(torch.nan_to_num(b['current_forcing']),stats['dynamic'],DYNAMIC_CHANNELS,1),device); sta=to_device(standardize(torch.nan_to_num(b['static_stack']),stats['static'],STATIC_CHANNELS,1),device); target=to_device(standardize(b['target_stack'],stats['target'],TARGET_CHANNELS,1),device); mask=to_device(b['valid_mask'].bool(),device)
   loss=masked_huber(model(x,cur,sta),target,mask); opt.zero_grad(); loss.backward(); opt.step(); measured.append(float(loss.detach())); sample_count+=len(b['site_id'])
   if number>=a.benchmark_batches: break
  seconds=time.perf_counter()-started; print({'benchmark_batches':len(measured),'samples':sample_count,'seconds':seconds,'samples_per_second':sample_count/seconds,'mean_loss':sum(measured)/len(measured),'batch_size':a.batch_size,'threads':a.threads,'interop_threads':a.interop_threads,'device':str(device)},flush=True); return
 for epoch in range(start,a.epochs):
  epoch_started=time.perf_counter()
  epoch_losses=[]
  for b in loader:
   x=to_device(standardize(torch.nan_to_num(b['dynamic_sequence']),stats['dynamic'],DYNAMIC_CHANNELS,2),device); cur=to_device(standardize(torch.nan_to_num(b['current_forcing']),stats['dynamic'],DYNAMIC_CHANNELS,1),device); sta=to_device(standardize(torch.nan_to_num(b['static_stack']),stats['static'],STATIC_CHANNELS,1),device); target=to_device(standardize(b['target_stack'],stats['target'],TARGET_CHANNELS,1),device); mask=to_device(b['valid_mask'].bool(),device)
   loss=masked_huber(model(x,cur,sta),target,mask); opt.zero_grad(); loss.backward(); opt.step(); value=float(loss.detach()); losses.append(value); epoch_losses.append(value)
  validation_loss=None
  if val_loader is not None and (epoch+1)%a.validation_every==0:
   model.eval(); validation_losses=[]
   with torch.no_grad():
    for b in val_loader:
     x=to_device(standardize(torch.nan_to_num(b['dynamic_sequence']),stats['dynamic'],DYNAMIC_CHANNELS,2),device); cur=to_device(standardize(torch.nan_to_num(b['current_forcing']),stats['dynamic'],DYNAMIC_CHANNELS,1),device); sta=to_device(standardize(torch.nan_to_num(b['static_stack']),stats['static'],STATIC_CHANNELS,1),device); target=to_device(standardize(b['target_stack'],stats['target'],TARGET_CHANNELS,1),device); mask=to_device(b['valid_mask'].bool(),device); validation_losses.append(float(masked_huber(model(x,cur,sta),target,mask)))
   validation_loss=sum(validation_losses)/len(validation_losses); model.train()
  metric={'epoch':epoch+1,'train_mean_loss':sum(epoch_losses)/len(epoch_losses),'validation_mean_loss':validation_loss}; epoch_metrics.append(metric)
  is_best=validation_loss is not None and validation_loss<best_validation_loss
  if is_best: best_validation_loss=validation_loss
  payload={'model_state':model.state_dict(),'optimizer_state':opt.state_dict(),'epoch':epoch+1,'losses':losses,'epoch_metrics':epoch_metrics,'best_validation_loss':best_validation_loss,'parameter_count':sum(x.numel() for x in model.parameters()),'normalization':str(a.normalization),'model_config':model_config,'runtime_config':{'batch_size':a.batch_size,'threads':a.threads,'interop_threads':a.interop_threads,'device':str(device)},'seed':a.seed}; a.checkpoint.parent.mkdir(parents=True,exist_ok=True); torch.save(payload,a.checkpoint)
  if is_best: torch.save(payload,a.checkpoint.with_name(a.checkpoint.stem+'_best'+a.checkpoint.suffix))
  print({'epoch':epoch+1,'mean_loss':metric['train_mean_loss'],'validation_mean_loss':validation_loss,'last_batch_loss':epoch_losses[-1],'steps':len(losses),'epoch_seconds':time.perf_counter()-epoch_started,'device':str(device)},flush=True)
if __name__=='__main__': main()

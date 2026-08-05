#!/usr/bin/env python3
"""Stateful Oct--Sep ConvGRU/ConvLSTM training with TBPTT or full water-year BPTT."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
if __package__ in {None,''}:sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from src.data.water_year_dataset import WaterYearDataset
from src.data.phase2_qc import DYNAMIC_CHANNELS,TARGET_CHANNELS
from src.data.static_contract import STATIC_CHANNELS
from src.data.target_qc import TARGET_PLAUSIBILITY_RANGES
from src.data.transforms import standardize
from src.models.multitask_model import ConvGRUMultitask
from src.training.losses import masked_huber
def model_config(stats,recurrent_cell):
 lower=[];upper=[]
 for n in TARGET_CHANNELS:
  if n in {'soilmoisture','plc_am','plc_pm'}:
   lo,hi=TARGET_PLAUSIBILITY_RANGES[n];m,s=stats['target'][n]['mean'],stats['target'][n]['std'];lower.append((lo-m)/s);upper.append((hi-m)/s)
  else:lower.append(0.);upper.append(0.)
 return {'recurrent_cell':recurrent_cell,'output_lower':lower,'output_upper':upper}
def prepare_site(s,stats):
 x=standardize(torch.nan_to_num(s['dynamic']).unsqueeze(0),stats['dynamic'],DYNAMIC_CHANNELS,2);z=standardize(torch.nan_to_num(s['static']).unsqueeze(0),stats['static'],STATIC_CHANNELS,1);y=standardize(s['target'].unsqueeze(0),stats['target'],TARGET_CHANNELS,2);valid=s['valid'].unsqueeze(0).bool();days=s['loss_days'];state=None;losses=[]
 return x,z,y,valid,days

def site_loss_tbptt(model,s,stats,chunk,optim=None):
 """Thirty-day TBPTT baseline: detach state and update after each scored chunk."""
 x,z,y,valid,days=prepare_site(s,stats);state=None;losses=[]
 for start in range(0,x.shape[1],chunk):
  end=min(x.shape[1],start+chunk);pred,state=model.forward_stateful_chunks(x[:,start:end],z,state);mask=valid[:,start:end]&days[start:end].view(1,-1,1,1,1)
  if mask.any():
   loss=masked_huber(pred,y[:,start:end],mask);losses.append(float(loss.detach()))
   if optim:optim.zero_grad();loss.backward();optim.step()
  state=tuple(v.detach() for v in state) if isinstance(state,tuple) else state.detach()
 return sum(losses)/len(losses)

def site_loss_full_bptt(model,s,stats,chunk,optim=None):
 """Backpropagate summer loss through the uninterrupted Oct--Sep state path.

 Chunks only limit the Python-side decoding loop; their recurrent states remain
 connected.  One optimizer step is made only after the complete site-year.
 """
 x,z,y,valid,days=prepare_site(s,stats);state=None;loss_sum=None;count=0
 for start in range(0,x.shape[1],chunk):
  end=min(x.shape[1],start+chunk);pred,state=model.forward_stateful_chunks(x[:,start:end],z,state);mask=valid[:,start:end]&days[start:end].view(1,-1,1,1,1);mask=mask & torch.isfinite(y[:,start:end])
  if mask.any():
   current=F.huber_loss(pred[mask],y[:,start:end][mask],delta=1.0,reduction='sum')
   loss_sum=current if loss_sum is None else loss_sum+current;count+=int(mask.sum())
 if loss_sum is None or count==0: raise ValueError(f"{s['site_id']} has no valid June--September targets")
 loss=loss_sum/count
 if optim:
  optim.zero_grad();loss.backward();optim.step()
 return float(loss.detach())
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--normalization',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--epochs',type=int,default=15);p.add_argument('--chunk-days',type=int,default=30);p.add_argument('--full-bptt',action='store_true',help='Keep the Oct--Sep graph connected; one update per site-year.');p.add_argument('--recurrent-cell',choices=('gru','lstm'),default='lstm',help='Use gru for the water-year ConvGRU comparison.');p.add_argument('--threads',type=int,default=32);p.add_argument('--seed',type=int,default=20260804);a=p.parse_args();torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.manual_seed(a.seed);stats=json.loads(a.normalization.read_text())['groups'];train=WaterYearDataset(a.manifest,'dev_train');val=WaterYearDataset(a.manifest,'dev_spatial_val');config=model_config(stats,a.recurrent_cell);model=ConvGRUMultitask(**config);opt=torch.optim.AdamW(model.parameters(),lr=1e-3);best=float('inf');loss_fn=site_loss_full_bptt if a.full_bptt else site_loss_tbptt
 for epoch in range(1,a.epochs+1):
  started=time.perf_counter();model.train();tl=[loss_fn(model,s,stats,a.chunk_days,opt) for s in train];model.eval()
  with torch.no_grad():vl=[loss_fn(model,s,stats,a.chunk_days) for s in val]
  metric={'epoch':epoch,'train_mean_loss':sum(tl)/len(tl),'validation_mean_loss':sum(vl)/len(vl),'epoch_seconds':time.perf_counter()-started};payload={'model_state':model.state_dict(),'epoch':epoch,'model_config':config,'normalization':str(a.normalization),'stateful_protocol':{'start':'water_year_october_1','chunk_days':a.chunk_days,'loss_months':[6,7,8,9],'backpropagation':'full_october_to_september' if a.full_bptt else 'truncated_at_chunk_boundaries'},'metrics':metric};a.checkpoint.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,a.checkpoint)
  if metric['validation_mean_loss']<best:best=metric['validation_mean_loss'];torch.save(payload,a.checkpoint.with_name(a.checkpoint.stem+'_best'+a.checkpoint.suffix))
  print(metric,flush=True)
if __name__=='__main__':main()

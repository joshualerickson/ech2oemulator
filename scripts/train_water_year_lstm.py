#!/usr/bin/env python3
"""Stateful Jan--Sep ConvGRU/ConvLSTM training with TBPTT or full BPTT."""
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
from src.utils.runtime import resolve_device,to_device
def model_config(stats,recurrent_cell):
 lower=[];upper=[]
 for n in TARGET_CHANNELS:
  if n in {'soilmoisture','plc_am','plc_pm'}:
   lo,hi=TARGET_PLAUSIBILITY_RANGES[n];m,s=stats['target'][n]['mean'],stats['target'][n]['std'];lower.append((lo-m)/s);upper.append((hi-m)/s)
  else:lower.append(0.);upper.append(0.)
 return {'recurrent_cell':recurrent_cell,'output_lower':lower,'output_upper':upper}
def prepare_site(s,stats,device):
 x=to_device(standardize(torch.nan_to_num(s['dynamic']).unsqueeze(0),stats['dynamic'],DYNAMIC_CHANNELS,2),device);z=to_device(standardize(torch.nan_to_num(s['static']).unsqueeze(0),stats['static'],STATIC_CHANNELS,1),device);y=to_device(standardize(s['target'].unsqueeze(0),stats['target'],TARGET_CHANNELS,2),device);valid=to_device(s['valid'].unsqueeze(0).bool(),device);days=to_device(s['loss_days'],device);state=None;losses=[]
 return x,z,y,valid,days

def site_loss_tbptt(model,s,stats,chunk,optim=None,device=torch.device('cpu')):
 """Thirty-day TBPTT baseline: detach state and update after each scored chunk."""
 x,z,y,valid,days=prepare_site(s,stats,device);chunk=x.shape[1] if chunk==0 else chunk;state=None;losses=[]
 for start in range(0,x.shape[1],chunk):
  end=min(x.shape[1],start+chunk);pred,state=model.forward_stateful_chunks(x[:,start:end],z,state);mask=valid[:,start:end]&days[start:end].view(1,-1,1,1,1)
  if mask.any():
   loss=masked_huber(pred,y[:,start:end],mask);losses.append(float(loss.detach()))
   if optim:optim.zero_grad();loss.backward();optim.step()
  state=tuple(v.detach() for v in state) if isinstance(state,tuple) else state.detach()
 return sum(losses)/len(losses)

def site_loss_full_bptt(model,s,stats,chunk,optim=None,device=torch.device('cpu')):
 """Backpropagate summer loss through the uninterrupted Jan--Sep state path.

 Chunks only limit the Python-side decoding loop; their recurrent states remain
 connected.  One optimizer step is made only after the complete site-year.
 """
 x,z,y,valid,days=prepare_site(s,stats,device);chunk=x.shape[1] if chunk==0 else chunk;state=None;loss_sum=None;count=0
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
 p=argparse.ArgumentParser()
 p.add_argument('--manifest',type=Path,required=True);p.add_argument('--normalization',type=Path,required=True)
 p.add_argument('--checkpoint',type=Path,required=True,help='Destination for the latest continuation checkpoint.')
 p.add_argument('--resume-from',type=Path,default=None,help='Load model weights and, when available, AdamW state from this checkpoint.')
 p.add_argument('--epochs',type=int,default=15,help='Additional epochs to run after --resume-from.')
 p.add_argument('--early-stopping-patience',type=int,default=0,help='Stop after this many consecutive validation epochs without a meaningful improvement; 0 disables early stopping.')
 p.add_argument('--min-delta',type=float,default=0.0,help='Required absolute validation-loss decrease to reset early-stopping patience.')
 p.add_argument('--chunk-days',type=int,default=30,help='Days decoded per loop; use 0 for one unchunked Jan--Sep forward/backward pass.')
 p.add_argument('--full-bptt',action='store_true',help='Keep the Jan--Sep graph connected; one update per site-year.')
 p.add_argument('--recurrent-cell',choices=('gru','lstm'),default='lstm',help='Use gru for the water-year ConvGRU comparison.')
 p.add_argument('--learning-rate',type=float,default=1e-3,help='AdamW learning rate; use a smaller value for a weights-only warm-start.')
 p.add_argument('--threads',type=int,default=32);p.add_argument('--device',default='auto',help='auto (default), cpu, cuda, or cuda:N.');p.add_argument('--seed',type=int,default=20260804);a=p.parse_args()
 if a.early_stopping_patience<0 or a.min_delta<0:p.error('--early-stopping-patience and --min-delta must be non-negative')
 torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.manual_seed(a.seed);device=resolve_device(a.device)
 stats=json.loads(a.normalization.read_text())['groups'];train=WaterYearDataset(a.manifest,'dev_train');val=WaterYearDataset(a.manifest,'dev_spatial_val')
 config=model_config(stats,a.recurrent_cell);model=ConvGRUMultitask(**config).to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate)
 start_epoch=0;best=float('inf');non_improving_epochs=0;resume_note=None
 if a.resume_from:
  previous=torch.load(a.resume_from,map_location='cpu',weights_only=False)
  previous_config=previous.get('model_config',{})
  if previous_config.get('recurrent_cell','gru')!=a.recurrent_cell:raise ValueError('Resume checkpoint recurrent cell does not match --recurrent-cell')
  model.load_state_dict(previous['model_state'])
  start_epoch=int(previous.get('epoch',0));best=float(previous.get('best_validation_loss',previous.get('metrics',{}).get('validation_mean_loss',best)))
  non_improving_epochs=int(previous.get('non_improving_epochs',0))
  if 'optimizer_state' in previous:opt.load_state_dict(previous['optimizer_state']);resume_note='model_and_adamw_state'
  else:resume_note='model_weights_only_fresh_adamw'
  print({'resume_from':str(a.resume_from),'start_epoch':start_epoch,'resume_mode':resume_note,'prior_best_validation_loss':best,'prior_non_improving_epochs':non_improving_epochs},flush=True)
 loss_fn=site_loss_full_bptt if a.full_bptt else site_loss_tbptt
 for epoch in range(start_epoch+1,start_epoch+a.epochs+1):
  started=time.perf_counter();model.train();tl=[loss_fn(model,s,stats,a.chunk_days,opt,device) for s in train];model.eval()
  with torch.no_grad():vl=[loss_fn(model,s,stats,a.chunk_days,None,device) for s in val]
  metric={'epoch':epoch,'train_mean_loss':sum(tl)/len(tl),'validation_mean_loss':sum(vl)/len(vl),'epoch_seconds':time.perf_counter()-started}
  improved=metric['validation_mean_loss']<best-a.min_delta
  if improved:best=metric['validation_mean_loss'];non_improving_epochs=0
  else:non_improving_epochs+=1
  payload={'model_state':model.state_dict(),'optimizer_state':opt.state_dict(),'epoch':epoch,'model_config':config,'normalization':str(a.normalization),'resume_from':str(a.resume_from) if a.resume_from else None,'resume_mode':resume_note,'runtime_config':{'device':str(device),'threads':a.threads},'early_stopping':{'patience':a.early_stopping_patience,'min_delta':a.min_delta},'best_validation_loss':best,'non_improving_epochs':non_improving_epochs,'stateful_protocol':{'temporal_contract':'target_water_year_forcing_calendar_year_v2','start':'calendar_year_january_1','end':'calendar_year_september_30','chunk_days':a.chunk_days,'sequence_execution':'single_full_jan_to_sep_pass' if a.chunk_days==0 else 'chronological_chunks','loss_months':[6,7,8,9],'backpropagation':'full_january_to_september' if a.full_bptt else 'truncated_at_chunk_boundaries'},'metrics':metric}
  a.checkpoint.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,a.checkpoint)
  if improved:torch.save(payload,a.checkpoint.with_name(a.checkpoint.stem+'_best'+a.checkpoint.suffix))
  print({**metric,'best_validation_loss':best,'non_improving_epochs':non_improving_epochs,'device':str(device)},flush=True)
  if a.early_stopping_patience and non_improving_epochs>=a.early_stopping_patience:
   print({'early_stopping':'triggered','epoch':epoch,'best_validation_loss':best,'patience':a.early_stopping_patience,'min_delta':a.min_delta},flush=True);break
if __name__=='__main__':main()

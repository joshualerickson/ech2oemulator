"""Train-only, persisted per-channel standardization."""
from __future__ import annotations
import json
from pathlib import Path
import torch

class RunningStats:
    def __init__(self) -> None: self.count=0; self.mean=0.; self.m2=0.
    def update(self, values: torch.Tensor) -> None:
        values=values[torch.isfinite(values)].double()
        if not values.numel(): return
        n=int(values.numel()); mean=float(values.mean()); m2=float(((values-mean)**2).sum()); total=self.count+n; delta=mean-self.mean
        self.mean+=delta*n/total; self.m2+=m2+delta*delta*self.count*n/total; self.count=total
    def result(self) -> dict[str,float|int]: return {'count':self.count,'mean':self.mean,'std':max((self.m2/max(self.count-1,1))**.5,1e-6)}

def save_stats(path:Path, groups:dict[str,list[RunningStats]], names:dict[str,tuple[str,...]], metadata:dict[str,object]|None=None) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    payload={'groups':{group:{name:stat.result() for name,stat in zip(names[group],stats)} for group,stats in groups.items()}}
    if metadata:
        payload['metadata']=metadata
    path.write_text(json.dumps(payload,indent=2))

def standardize(x:torch.Tensor, group:dict[str,dict[str,float|int]], names:tuple[str,...], dim:int) -> torch.Tensor:
    shape=[1]*x.ndim; shape[dim]=len(names)
    mean=torch.tensor([group[n]['mean'] for n in names],dtype=x.dtype,device=x.device).reshape(shape)
    std=torch.tensor([group[n]['std'] for n in names],dtype=x.dtype,device=x.device).reshape(shape)
    return (x-mean)/std

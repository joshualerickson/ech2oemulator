#!/usr/bin/env python3
"""Render observed, predicted, and residual maps from one holdout NPZ export."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 d=np.load(a.input); pred=d['prediction']; obs=d['observation']; mask=d['valid_mask']; names=d['target_names']
 fig,axes=plt.subplots(len(names),3,figsize=(10,3*len(names)))
 for i,name in enumerate(names):
  o=np.where(mask[i],obs[i],np.nan); q=np.where(mask[i],pred[i],np.nan); r=q-o; lim=np.nanpercentile(np.abs(r),99)
  for ax,data,title,cmap in ((axes[i,0],o,'observed','viridis'),(axes[i,1],q,'predicted','viridis'),(axes[i,2],r,'residual','coolwarm')):
   im=ax.imshow(data,cmap=cmap,vmin=-lim if title=='residual' else None,vmax=lim if title=='residual' else None); ax.set_title(f'{name}: {title}'); ax.set_axis_off(); fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
 fig.tight_layout(); a.output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(a.output,dpi=150); print(a.output)
if __name__=='__main__': main()

#!/usr/bin/env python3
"""Export held-out predictions, observations, and masks as compressed arrays."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
if __package__ in {None,""}: sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import rasterio
from src.data.sequence_dataset import SequenceDataset
from src.models.multitask_model import model_from_checkpoint,load_model_state
from src.data.phase2_qc import DYNAMIC_CHANNELS,TARGET_CHANNELS
from src.data.static_contract import STATIC_CHANNELS
from src.data.transforms import standardize
def inverse(x,group,names):
 mean=torch.tensor([group[n]['mean'] for n in names],dtype=x.dtype).view(-1,1,1); std=torch.tensor([group[n]['std'] for n in names],dtype=x.dtype).view(-1,1,1)
 return x*std+mean
def write_tif(path, values, source, names, nodata=-9999.0):
 with rasterio.open(f'NETCDF:{source}:soilmoisture') as ref:
  profile=ref.profile.copy(); profile.pop('blockxsize',None); profile.pop('blockysize',None); profile.update(driver='GTiff',count=len(names),dtype='float32',nodata=nodata,compress='deflate',tiled=False)
  with rasterio.open(path,'w',**profile) as dst:
   for band,(name,value) in enumerate(zip(names,values),1):
    dst.write(np.where(np.isfinite(value),value,nodata).astype('float32'),band); dst.set_band_description(band,str(name))
def main():
 p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--normalization',type=Path,required=True); p.add_argument('--site-id',required=True); p.add_argument('--month',type=int,default=8); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--max-days',type=int,default=0); a=p.parse_args()
 ds=SequenceDataset(a.manifest,'dev_spatial_test'); indices=[i for i,r in enumerate(ds.rows) if r['site_id']==a.site_id and int(r['target_date'][5:7])==a.month]
 if a.max_days: indices=indices[:a.max_days]
 if not indices: raise ValueError('No matching held-out rows')
 state=torch.load(a.checkpoint,map_location='cpu',weights_only=False); stats=json.loads(a.normalization.read_text())['groups']; model=model_from_checkpoint(state); load_model_state(model,state); model.eval(); a.output_dir.mkdir(parents=True,exist_ok=True)
 with torch.no_grad():
  for i in indices:
   s=ds[i]; dynamic=standardize(torch.nan_to_num(s['dynamic_sequence']).unsqueeze(0),stats['dynamic'],DYNAMIC_CHANNELS,2); current=standardize(torch.nan_to_num(s['current_forcing']).unsqueeze(0),stats['dynamic'],DYNAMIC_CHANNELS,1); static=standardize(torch.nan_to_num(s['static_stack']).unsqueeze(0),stats['static'],STATIC_CHANNELS,1); pred=inverse(model(dynamic,current,static)[0],stats['target'],TARGET_CHANNELS).numpy()
   names=np.array(TARGET_CHANNELS); obs=s['target_stack'].numpy(); mask=s['valid_mask'].numpy(); base=a.output_dir/f"{s['site_id']}_{s['target_date']}"; np.savez_compressed(base.with_suffix('.npz'),prediction=pred,observation=obs,valid_mask=mask,target_names=names)
   write_tif(base.with_name(base.name+'_prediction.tif'),np.where(mask,pred,np.nan),ds.rows[i]['target_source'],names)
   write_tif(base.with_name(base.name+'_observation.tif'),obs,ds.rows[i]['target_source'],names)
   write_tif(base.with_name(base.name+'_residual.tif'),np.where(mask,pred-obs,np.nan),ds.rows[i]['target_source'],names)
   write_tif(base.with_name(base.name+'_valid_mask.tif'),mask.astype('float32'),ds.rows[i]['target_source'],names,nodata=0.0)
 print({'site_id':a.site_id,'days_written':len(indices),'output_dir':str(a.output_dir)})
if __name__=='__main__': main()

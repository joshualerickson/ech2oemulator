#!/usr/bin/env python3
"""Replay an Oct--Sep stateful ConvGRU or ConvLSTM and export GIS-ready predictions."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
if __package__ in {None,''}:sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,rasterio,torch
from src.data.water_year_dataset import WaterYearDataset
from src.data.phase2_qc import DYNAMIC_CHANNELS,TARGET_CHANNELS
from src.data.static_contract import STATIC_CHANNELS
from src.data.transforms import standardize
from src.models.multitask_model import model_from_checkpoint,load_model_state
def inverse(x,stats):
 m=torch.tensor([stats[n]['mean'] for n in TARGET_CHANNELS]).view(1,1,-1,1,1);s=torch.tensor([stats[n]['std'] for n in TARGET_CHANNELS]).view(1,1,-1,1,1);return x*m*0+x*s+m
def write(path,values,source,nodata=-9999.):
 with rasterio.open(f'NETCDF:{source}:soilmoisture') as d:profile=d.profile.copy()
 profile.pop('blockxsize',None);profile.pop('blockysize',None);profile.update(driver='GTiff',count=5,dtype='float32',nodata=nodata,compress='deflate',tiled=False)
 with rasterio.open(path,'w',**profile) as d:
  for i,(n,v) in enumerate(zip(TARGET_CHANNELS,values),1):d.write(np.where(np.isfinite(v),v,nodata).astype('float32'),i);d.set_band_description(i,n)
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--normalization',type=Path,required=True);p.add_argument('--site-id',required=True);p.add_argument('--month',type=int,default=8);p.add_argument('--max-days',type=int,default=1);p.add_argument('--chunk-days',type=int,default=30);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();stats=json.loads(a.normalization.read_text())['groups'];ds=WaterYearDataset(a.manifest,'dev_spatial_test');s=next(x for x in ds if x['site_id']==a.site_id);state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);m=model_from_checkpoint(state);load_model_state(m,state);m.eval();x=standardize(torch.nan_to_num(s['dynamic']).unsqueeze(0),stats['dynamic'],DYNAMIC_CHANNELS,2);z=standardize(torch.nan_to_num(s['static']).unsqueeze(0),stats['static'],STATIC_CHANNELS,1);parts=[];hidden=None
 with torch.no_grad():
  for start in range(0,x.shape[1],a.chunk_days):q,hidden=m.forward_stateful_chunks(x[:,start:start+a.chunk_days],z,hidden);parts.append(q)
 pred=inverse(torch.cat(parts,1),stats['target'])[0].numpy();obs=s['target'].numpy();valid=s['valid'].numpy();a.output_dir.mkdir(parents=True,exist_ok=True);written=0
 for i,day in enumerate(s['dates']):
  if int(day[5:7])!=a.month:continue
  if a.max_days and written>=a.max_days:break
  base=a.output_dir/f'{a.site_id}_{day}';write(base.with_name(base.name+'_prediction.tif'),np.where(valid[i],pred[i],np.nan),s['source']);write(base.with_name(base.name+'_observation.tif'),obs[i],s['source']);write(base.with_name(base.name+'_residual.tif'),np.where(valid[i],pred[i]-obs[i],np.nan),s['source']);write(base.with_name(base.name+'_valid_mask.tif'),valid[i].astype('float32'),s['source'],0.);written+=1
 print({'site_id':a.site_id,'days_written':written,'output_dir':str(a.output_dir)})
if __name__=='__main__':main()

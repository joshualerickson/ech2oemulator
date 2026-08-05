"""One full chronological water year per site for stateful recurrent training."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np,rasterio,torch,xarray as xr
from torch.utils.data import Dataset
from src.data.phase2_qc import DYNAMIC_CHANNELS,TARGET_CHANNELS
from src.data.static_contract import Grid,load_selected_static_stack
from src.data.target_qc import mask_implausible
class WaterYearDataset(Dataset):
 def __init__(self,manifest,split):
  with Path(manifest).open(newline='') as f:self.rows=[r for r in csv.DictReader(f) if r['split']==split]
  if not self.rows:raise ValueError(f'No {split} rows')
  self.cache={};self.static={}
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i];site=r['site_id'];wy=int(r['water_year_end'])
  if site not in self.cache:
   root=Path(r['dynamic_sequence_source']);dyn=[]
   for n in DYNAMIC_CHANNELS:
    with rasterio.open(root/f'{n}_{wy}.tif') as d:dyn.append(d.read(out_dtype='float32')[:,1:-1,1:-1])
   with xr.open_dataset(r['target_source'],engine='h5netcdf',decode_cf=False,mask_and_scale=False) as d:tar=[np.asarray(d[n].values,dtype=np.float32) for n in TARGET_CHANNELS]
   with Path(r['daily_screen_source']).open(newline='') as f:daily=list(csv.DictReader(f))
   self.cache[site]=(dyn,tar,daily)
  dyn,tar,daily=self.cache[site]
  if site not in self.static:
   with rasterio.open(f"NETCDF:{r['target_source']}:soilmoisture") as d:grid=Grid(d.width,d.height,d.transform,d.crs)
   self.static[site]=torch.from_numpy(load_selected_static_stack(Path(r['static_source']),grid))
  masked=[mask_implausible(v,n) for n,v in zip(TARGET_CHANNELS,tar)]
  target=np.stack(masked,1);valid=np.isfinite(target);loss_days=np.array([x['target_qa_valid']=='True' and int(x['date'][5:7]) in (6,7,8,9) for x in daily])
  return {'site_id':site,'dates':[x['date'] for x in daily],'source':r['target_source'],'dynamic':torch.from_numpy(np.stack(dyn,1)),'static':self.static[site],'target':torch.from_numpy(target),'valid':torch.from_numpy(valid),'loss_days':torch.from_numpy(loss_days)}

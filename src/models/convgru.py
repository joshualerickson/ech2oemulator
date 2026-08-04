from __future__ import annotations
import torch
from torch import nn

class ConvGRUCell(nn.Module):
    def __init__(self, input_channels:int, hidden_channels:int) -> None:
        super().__init__(); self.hidden_channels=hidden_channels
        self.gates=nn.Conv2d(input_channels+hidden_channels,2*hidden_channels,3,padding=1)
        self.candidate=nn.Conv2d(input_channels+hidden_channels,hidden_channels,3,padding=1)
    def forward(self,x:torch.Tensor,h:torch.Tensor)->torch.Tensor:
        update,reset=self.gates(torch.cat((x,h),1)).chunk(2,1); update,reset=torch.sigmoid(update),torch.sigmoid(reset)
        candidate=torch.tanh(self.candidate(torch.cat((x,reset*h),1)))
        return (1-update)*h+update*candidate

class ConvGRU(nn.Module):
    def __init__(self,input_channels:int,hidden_channels:int)->None:
        super().__init__(); self.cell=ConvGRUCell(input_channels,hidden_channels); self.hidden_channels=hidden_channels
    def forward(self,sequence:torch.Tensor)->torch.Tensor:
        b,_,_,h,w=sequence.shape; state=sequence.new_zeros((b,self.hidden_channels,h,w))
        for step in sequence.unbind(1): state=self.cell(step,state)
        return state

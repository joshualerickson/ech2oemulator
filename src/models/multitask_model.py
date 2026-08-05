from __future__ import annotations
import torch
from torch import nn
from src.models.convgru import ConvGRU
from src.models.convlstm import ConvLSTM

class ResidualBlock(nn.Module):
    def __init__(self,channels:int)->None:
        super().__init__(); self.body=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1),nn.GroupNorm(4,channels),nn.GELU(),nn.Conv2d(channels,channels,3,padding=1),nn.GroupNorm(4,channels)); self.act=nn.GELU()
    def forward(self,x:torch.Tensor)->torch.Tensor: return self.act(x+self.body(x))

class ConvGRUMultitask(nn.Module):
    def __init__(self,dynamic_channels:int=6,static_channels:int=13,targets:int=5,hidden_channels:int=32,decoder_channels:int=32,recurrent_cell:str='gru',output_lower:list[float]|None=None,output_upper:list[float]|None=None)->None:
        super().__init__()
        if recurrent_cell not in {'gru','lstm'}: raise ValueError(f'Unsupported recurrent_cell {recurrent_cell!r}')
        self.recurrent_cell=recurrent_cell; encoder_type=ConvGRU if recurrent_cell=='gru' else ConvLSTM; self.encoder=encoder_type(dynamic_channels,hidden_channels); self.stem=nn.Conv2d(hidden_channels+dynamic_channels+static_channels,decoder_channels,3,padding=1); self.trunk=nn.Sequential(ResidualBlock(decoder_channels),ResidualBlock(decoder_channels)); self.heads=nn.ModuleList([nn.Conv2d(decoder_channels,1,1) for _ in range(targets)])
        if (output_lower is None) != (output_upper is None): raise ValueError('Specify both output_lower and output_upper, or neither')
        if output_lower is None:
            output_lower=[0.0]*targets; output_upper=[0.0]*targets; bounded=[False]*targets
        else:
            if len(output_lower)!=targets or len(output_upper)!=targets: raise ValueError('Output bounds must match target count')
            bounded=[lower<upper for lower,upper in zip(output_lower,output_upper)]
        # Bounds are in normalized target units.  They constrain only requested
        # channels; other heads retain an unconstrained linear output.
        self.register_buffer('output_lower',torch.tensor(output_lower,dtype=torch.float32).view(1,-1,1,1)); self.register_buffer('output_upper',torch.tensor(output_upper,dtype=torch.float32).view(1,-1,1,1)); self.register_buffer('bounded_outputs',torch.tensor(bounded,dtype=torch.bool).view(1,-1,1,1))
    def forward(self,sequence:torch.Tensor,current:torch.Tensor,static:torch.Tensor)->torch.Tensor:
        hidden=self.encoder(sequence); return self.decode(hidden,current,static)
    def decode(self,hidden:torch.Tensor,current:torch.Tensor,static:torch.Tensor)->torch.Tensor:
        """Apply the unchanged full-resolution decoder to a recurrent state."""
        features=self.trunk(self.stem(torch.cat((hidden,current,static),1))); raw=torch.cat([head(features) for head in self.heads],1)
        bounded=self.output_lower+(self.output_upper-self.output_lower)*torch.sigmoid(raw)
        return torch.where(self.bounded_outputs,bounded,raw)
    def forward_lstm_chunks(self,sequence:torch.Tensor,static:torch.Tensor,state:tuple[torch.Tensor,torch.Tensor]|None=None)->tuple[torch.Tensor,tuple[torch.Tensor,torch.Tensor]]:
        """Decode every LSTM step while carrying state across chronological chunks."""
        if self.recurrent_cell!='lstm': raise ValueError('Stateful chunks require recurrent_cell=lstm')
        hidden_steps,next_state=self.encoder.forward_stateful(sequence,state,return_sequence=True)
        predictions=[self.decode(hidden_steps[:,step],sequence[:,step],static) for step in range(sequence.shape[1])]
        return torch.stack(predictions,1),next_state


def model_from_checkpoint(checkpoint:dict)->ConvGRUMultitask:
    """Construct either legacy unbounded or new bounded-head checkpoints."""
    return ConvGRUMultitask(**checkpoint.get('model_config',{}))


def load_model_state(model:ConvGRUMultitask, checkpoint:dict)->None:
    """Load a checkpoint while accepting bound buffers absent from legacy files."""
    result=model.load_state_dict(checkpoint['model_state'],strict=False)
    tolerated={'output_lower','output_upper','bounded_outputs'}
    if set(result.missing_keys)-tolerated or set(result.unexpected_keys)-tolerated:
        raise RuntimeError(f'Unexpected checkpoint/model mismatch: {result}')

import torch
import torch.nn as nn
import re
import copy

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

def build_vision_projectors(config, num_of_kvs=4,version='cross_merge'):
    return Vision_projectors(config, num_of_kvs,version)
class Vision_projectors(nn.Module):
    def __init__(self, config, num_of_kvs=4,version='cross_merge'):
        super().__init__()
        self.version=version
        self.num_of_kvs=num_of_kvs
        projector_type = getattr(config, 'mm_projector_type', 'linear')
        
        if projector_type == 'linear':
            if version.find("merge")!=-1:
                self.models= nn.Linear(config.mm_hidden_size, config.hidden_size)
            elif version.find("sequence")!=-1:
                self.models=  nn.ModuleList([nn.Linear(config.mm_hidden_size, config.hidden_size) for _ in range(num_of_kvs+1)])
            elif version.find("channel")!=-1:
                self.models=  nn.Linear(config.mm_hidden_size*int(num_of_kvs+1), config.hidden_size)
            return
        
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            mulxc=int(num_of_kvs+1) if version.find("channel")!=-1 else 1
            modules = [nn.Linear(config.mm_hidden_size*mulxc, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            module= nn.Sequential(*modules)
            if version.find("sequence")!=-1:
                self.models= nn.ModuleList([copy.deepcopy(module) for _ in range(2)])
            else:
                self.models=module
            return
        
        if projector_type == 'identity':
            self.models=  IdentityMap()
            return

        raise ValueError(f'Unknown projector type: {projector_type}')
    def forward(self,x):
        if isinstance(x,list) and self.version.find("sequence")!=-1:
            y=torch.sum(torch.stack(x[1:],dim=1),dim=1)
            y=torch.cat([model(x_i) for model,x_i in zip(self.models,[x[0],y])],dim=1)
        elif isinstance(x,list):
            y_list=torch.stack([model(x_i)for model,x_i in zip(self.models,x)])
            y=torch.sum(y_list,dim=0)
        else:
            if self.version.find("sequence")==-1:
                y=self.models(x)
            else:  
                y=torch.cat([model(x_i) for model,x_i in zip(self.models,torch.chunk(x,int(self.num_of_kvs+1),dim=1))],dim=1)
        return y

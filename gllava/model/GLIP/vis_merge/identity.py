import torch
import torch.utils.checkpoint
from torch import nn
import math
import numpy as np
import math
import torch.nn.functional as F



class CrossAttention(nn.Module):

    def __init__(self, q_dim, kv_dim, hidden_dim, num_heads, attention_bias=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // self.num_heads

        if (self.head_dim * self.num_heads) != self.hidden_dim:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads (got `hidden_dim`: {self.hidden_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Sequential(nn.LayerNorm(q_dim), nn.Linear(q_dim, self.num_heads * self.head_dim, bias=attention_bias))
        self.k_proj = nn.Sequential(nn.LayerNorm(kv_dim), nn.Linear(kv_dim, self.num_heads * self.head_dim, bias=attention_bias))
        self.v_proj = nn.Sequential(nn.LayerNorm(kv_dim), nn.Linear(kv_dim, self.num_heads * self.head_dim, bias=attention_bias))
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, q_dim, bias=attention_bias)

    def forward(
        self,
        vision_latents, queries, attention_mask=None
    ):
        
        bsz, q_len, _ = queries.size()
        bsz, v_len, _ = vision_latents.size()

        query_states = self.q_proj(queries)
        key_states = self.k_proj(vision_latents)
        value_states = self.v_proj(vision_latents)

        query_states = query_states.view(bsz, -1, q_len,self.num_heads, self.head_dim).permute(0,2,3,1,4).view(bsz, q_len*self.num_heads, -1, self.head_dim)
        key_states = key_states.view(bsz, -1, q_len,self.num_heads, self.head_dim).permute(0,2,3,1,4).view(bsz, q_len*self.num_heads, -1, self.head_dim)
        value_states =value_states.view(bsz, -1, q_len,self.num_heads, self.head_dim).permute(0,2,3,1,4).view(bsz, q_len*self.num_heads, -1, self.head_dim)



        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        # scale_factor = 1 / math.sqrt(query_states.size(-1)) 
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            # scale_factor=scale_factor
        )#(bsz, q_len*self.num_heads, 1, self.head_dim)

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output
    

class AggregationBlock(nn.Module):
    def __init__(self, attention, q_dim, kv_dim, hidden_dim, num_heads, attention_bias=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // self.num_heads

        if (self.head_dim * self.num_heads) != self.hidden_dim:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads (got `hidden_dim`: {self.hidden_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.attention = attention
        if attention:
            self.attention_layer = CrossAttention(q_dim, kv_dim, hidden_dim, num_heads, attention_bias)
        else:
            self.attention_layer = MLP(kv_dim, q_dim, q_dim,norm=True)        

    def forward(
        self,
        vision_latents, queries, attention_mask=None
    ):
        if self.attention:
            queries = self.attention_layer(vision_latents, queries, attention_mask)
        else:
            queries = self.attention_layer(vision_latents)

        return queries



class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out,norm=False):
        super().__init__() 
        self.linear_1 = nn.Linear(d_in, d_hidden, bias=False)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(d_hidden, d_out, bias=False)
        if norm: self.norm = nn.LayerNorm(d_hidden)
        
        

    def forward(self, x):
        if hasattr(self,"norm"):
            return self.linear_2(self.norm(self.act(self.linear_1(x))))
        return self.linear_2(self.act(self.linear_1(x)))


class VisionAggregationLayer(nn.Module):
    def __init__(self, q_dim=1024, context_dim=256, num_of_kvs=4, hidden_dim = 1024):
        super().__init__()
        num_heads = 16
        self.num_of_kvs = num_of_kvs
        
        # self.proj_in = nn.Linear(q_dim, hidden_dim, bias=False)

        self.proj_out = MLP(hidden_dim, hidden_dim, q_dim)

        self.norm = nn.LayerNorm(hidden_dim)

        if self.num_of_kvs > 1:
            self.proj_context = nn.Linear(context_dim, context_dim, bias=False)#context_dim=256
            self.norm_context = nn.LayerNorm(context_dim)
            self.norm_quy = nn.LayerNorm(q_dim)
            self.weight_mlp = MLP(q_dim+context_dim, hidden_dim, 1)

        for i in range(self.num_of_kvs):
            setattr(self, "aggregate_{}".format(i), AggregationBlock(False, hidden_dim, context_dim, hidden_dim, num_heads))
    def forward(
        self,
        queries,
        context_feature,
        vision_latents_list,mul4=False
    ):

        # residual = queries #clip 
        # vision_latents_list#GLIP
        if self.num_of_kvs > 1:
            context_feature = self.proj_context(context_feature)#global features
            combination_weight = (self.weight_mlp(torch.cat([self.norm_quy(queries.mean(1,True)).expand(-1,context_feature.size(1),-1), self.norm_context(context_feature)], -1))).sigmoid() #.softmax(1) .sigmoid()# B *num_tower*1
            combination_weight = combination_weight.unsqueeze(-1) #B * num_tower*1*1
        else:
            combination_weight = 1

        # queries = self.proj_in(queries)

        vision_latents_list = vision_latents_list[:self.num_of_kvs]

        aggregated_vision_latents_list = []
        for i, vision_latents in enumerate(vision_latents_list):
            aggregated_vision_latents_list.append(getattr(self, "aggregate_{}".format(i))(vision_latents, queries))
        if not mul4:
            aggregated_vision_latents = torch.stack(aggregated_vision_latents_list, 1)

            aggregated_vision_latents = aggregated_vision_latents * combination_weight

            aggregated_vision_latents  = self.norm(aggregated_vision_latents)

            aggregated_vision_latents = self.proj_out(aggregated_vision_latents)
        else:
            # aggregated_vision_latents = [fea*coe.squeeze(-1) for fea, coe in zip(aggregated_vision_latents_list,combination_weight.chunk(self.num_of_kvs,dim=1))]
            # aggregated_vision_latents_list = [self.proj_out(self.norm(fea)) for fea in aggregated_vision_latents_list]
            aggregated_vision_latents_list = [self.proj_out(self.norm(fea+queries))  for fea in aggregated_vision_latents_list]#+queries
            aggregated_vision_latents = [fea*coe.squeeze(-1) for fea, coe in zip(aggregated_vision_latents_list,combination_weight.chunk(self.num_of_kvs,dim=1))]

        return aggregated_vision_latents 

class ORi_VisionToken(nn.Module):
    def __init__(self, q_dim=1024, context_dim=256, vision_hidden_size=1024, num_of_kvs=4,version='identify_channel',num_of_layers=1):
        super().__init__()
        self.version=version
        self.layers = nn.ModuleList([VisionAggregationLayer(q_dim, context_dim, num_of_kvs,hidden_dim=vision_hidden_size) for idx in range(num_of_layers)])
    def forward(self, queries,*vision_latents_list):
        scalars=[1 for _ in range(self.layers[0].num_of_kvs)]
        dtype=vision_latents_list[0][0].dtype
        query_size=int(math.sqrt(queries.size(1)))
        if len(scalars)==3:
            vision_latents_list=list(vision_latents_list[0])
            vision_latents_list.pop(-1) 
        elif len(scalars)==2:
            vision_latents_list=list(vision_latents_list[0])
            vision_latents_list.pop(1) 
            vision_latents_list.pop(2) 
        elif len(scalars)==1:
            vision_latents_list=list(vision_latents_list[0])
            vision_latents_list.pop(0) #1 0 0 0
            vision_latents_list.pop(0) #2 2 2 0
            vision_latents_list.pop(0) #-1 -1 0 0
        elif len(scalars)==4:
            vision_latents_list=list(vision_latents_list[0])
            # if self.version.find("sequence")!=-1:scalars=[0.55,0.2,0.25,1.] #[0.15,0.2,0.25,0.4] # scalars=[0.55,0.2,0.25,1.] 
        else:
            raise ValueError(f"not support the '{len(scalars)}' layers mergesion")
        B,C,_,_=vision_latents_list[0].size()
        vision_latents_list=[F.interpolate(latent.to(torch.float32), size=(int(query_size*s),int(query_size*s)), mode="nearest").reshape(B,C,-1).transpose(1,2).to(dtype) for s,latent in zip(scalars,vision_latents_list[::-1])]
        context_feature=torch.stack([latent.mean(1) for latent in vision_latents_list],dim=1)
        for layer in self.layers:
            mul4=True if (self.version.find("sequence")!=-1 and len(scalars)==4) else False
            aggregated_vision_latents = layer(queries, context_feature,vision_latents_list,mul4=mul4)
        if self.version.find("sequence")!=-1:
            if mul4:
                aggregated_vision_latents.insert(0,queries)
                return aggregated_vision_latents
            b,n,c,=queries.size()
            return torch.cat([queries,aggregated_vision_latents.reshape(b,-1,c)],dim=1)
        elif self.version.find("channel")!=-1:
            b,n,c,=queries.size()
            return torch.cat([queries,aggregated_vision_latents.transpose(1,2).reshape(b,n,-1)],dim=-1)
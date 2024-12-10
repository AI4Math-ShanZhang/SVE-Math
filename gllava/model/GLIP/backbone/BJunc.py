import torch.nn as nn
import torch.nn.functional as F
import torch
from  gllava.model.GLIP.backbone.ffc import SpectralTransform
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)
    def forward(self, values, keys, queries, mask=None):
        N,C,H,W = queries.shape
        values, keys, queries=values.reshape(N,C,-1).transpose(1,2), keys.reshape(N,C,-1).transpose(1,2), queries.reshape(N,C,-1).transpose(1,2)
        org=queries
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        
        # Split the embedding into self.num_heads different pieces
        values = self.values(values).view(N, value_len, self.num_heads, self.head_dim)
        keys = self.keys(keys).view(N, key_len, self.num_heads, self.head_dim)
        queries = self.queries(queries).view(N, query_len, self.num_heads, self.head_dim)
        
        values = values.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)
        
        # Scaled dot-product attention
        energy = torch.einsum("nhqd,nhkd->nhqk", [queries, keys]) / (self.head_dim ** (1 / 2))
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy, dim=3)
        
        out = torch.einsum("nhql,nhld->nqhd", [attention.to(values.dtype), values]).reshape(N, query_len, self.head_dim * self.num_heads)
        out = self.fc_out(out)
        return out.transpose(1,2).reshape(N,C,H,W)
class HourglassNet3D(nn.Module):
    def __init__(self, nOutChannels=256):
        super(HourglassNet3D, self).__init__()
        self.nOutChannels = nOutChannels
        self.ffc=SpectralTransform(nOutChannels)
        # self.conv0 = nn.Conv2d(nOutChannels, nOutChannels, kernel_size = 1, stride = 1, padding = 0,bias=False)
        # self.bn0 = nn.BatchNorm2d(nOutChannels)
        # self.conv0_0 = nn.Conv2d(nOutChannels, nOutChannels, kernel_size = 1, stride = 1, padding = 0,bias=False)
        
        self.deconv1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), 
                                     nn.Conv2d(self.nOutChannels, self.nOutChannels, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = nn.BatchNorm2d(self.nOutChannels)

        self.relu = nn.ReLU(inplace = True)
        ###add merge
        self.merge_conv1 =nn.Conv2d(self.nOutChannels, self.nOutChannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.merge_bn1 = nn.BatchNorm2d(self.nOutChannels)
        self.merge_bn2 = nn.BatchNorm2d(self.nOutChannels)
        self.merge_trans=MultiHeadAttention(self.nOutChannels)

    def forward(self, x):
        large=x[1]
        v=self.relu(self.merge_bn2(x[-2]))
        q=self.relu(self.merge_bn2(large))
        boundOut=self.merge_trans(v,v,q)
        boundOut=self.relu(self.merge_bn1(self.merge_conv1(large+boundOut)))
        boundOut=F.interpolate(boundOut.float(),scale_factor=2,mode="nearest").to(large.dtype)
        boundOut = self.relu(self.bn1(self.deconv1[1](boundOut)))
        lagfeat=F.interpolate(boundOut.float(),size=(x[0].shape[2:]),mode="nearest").to(large.dtype)
        ##also work
        # x0=self.relu(self.conv0_0(x[0]))
        # x0=x0+lagfeat
        # boundOut=self.relu(self.bn0(self.conv0(x0)))
        boundOut=self.ffc(x[0]+lagfeat)
        return (boundOut,)+x[2:]
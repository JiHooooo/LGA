import logging
import warnings
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import torch.utils.checkpoint as cp

# from ops.modules import MSDeformAttn
from timm.models.layers import DropPath
from typing import Type

_logger = logging.getLogger(__name__)
import math


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = int(embedding_dim // downsample_rate)
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

class InteractionBlock_global(nn.Module):
    def __init__(self, embed_dim, num_heads=8, downsample_rate=1, cff_ratio=0.5,):
        super().__init__()
        
        self.injector = injector_global(embed_dim, num_heads, downsample_rate, with_cffn=True, 
                                            mlp_dim=int(embed_dim*cff_ratio))
        self.extractor = extractor_global(embed_dim, num_heads, downsample_rate, with_cffn=True, 
                                          mlp_dim=int(embed_dim*cff_ratio))

    def forward(self, vit_feature, adapter_feature):
        
        adapter_feature = self.extractor(vit_feature, adapter_feature)
        vit_feature = self.injector(vit_feature, adapter_feature)
        
        return vit_feature, adapter_feature

class injector_global(nn.Module):
    def __init__(self, embed_dim, num_heads=8, downsample_rate=1, with_cffn=True, mlp_dim=2048, mlp_activation=nn.ReLU,):
        super().__init__()
        
        self.gamma = nn.Parameter(0 * torch.ones((embed_dim)), requires_grad=True)
        self.vit_norm = nn.LayerNorm(embed_dim)
        self.adapter_norm = nn.LayerNorm(embed_dim)
        self.atten = Attention(embedding_dim = embed_dim, num_heads=num_heads, downsample_rate=downsample_rate)
        self.with_cffn = with_cffn
        if self.with_cffn:
            self.mlp = MLPBlock(embed_dim, mlp_dim, mlp_activation)
            self.mlp_norm = nn.LayerNorm(embed_dim)
            self.drop_path = nn.Identity()

    def forward(self, vit_feature, adapter_feature):
        
        vit_feature_norm = self.vit_norm(vit_feature)

        adapter_feature_norm = self.adapter_norm(adapter_feature)
        
        attn = self.atten(q=vit_feature_norm, k=adapter_feature_norm, v=adapter_feature_norm)

        vit_feature = vit_feature + self.gamma * attn

        if self.with_cffn:
            vit_feature = vit_feature +  self.drop_path(self.mlp(self.mlp_norm(vit_feature)))

        return vit_feature

    
class extractor_global(nn.Module):
    def __init__(self, embed_dim, num_heads=8, downsample_rate=1, mlp_dim=2048, mlp_activation=nn.ReLU, with_cffn=True):
        super().__init__()
       
        self.vit_norm = nn.LayerNorm(embed_dim)
        self.adapter_norm = nn.LayerNorm(embed_dim)
        self.atten = Attention(embedding_dim = embed_dim, num_heads=num_heads, downsample_rate=downsample_rate)
        
        self.with_cffn = with_cffn
        if self.with_cffn:
            self.mlp = MLPBlock(embed_dim, mlp_dim, mlp_activation)
            self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, vit_feature, adapter_feature):

        vit_feature_norm = self.vit_norm(vit_feature)

        adapter_feature_norm = self.adapter_norm(adapter_feature)

        attn = self.atten(q=adapter_feature_norm, k=vit_feature_norm, v=vit_feature_norm)

        adapter_feature = adapter_feature + attn

        if self.with_cffn:
            adapter_feature = adapter_feature + self.mlp(self.mlp_norm(adapter_feature))

        return adapter_feature


import torch
import torch.nn as nn
from utilities.config import embed_size, BIAS, dropout
from attention import AttentionHead

class ForwardLayer(nn.Module):
  def __init__(self,embed_size):
    super().__init__()
    self.network = nn.Sequential(
        nn.Linear(embed_size, 6*embed_size, bias=BIAS),
        nn.GELU(),
        nn.Linear(6*embed_size, embed_size, bias=BIAS),
        nn.Dropout(dropout)
    )

  def forward(self,x):
    x = self.network(x)
    return x

class MultiHead(nn.Module):
  def __init__(self, n_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(n_heads)])
    self.combine = nn.Linear(head_size * n_heads, embed_size, bias=BIAS) # 378 -> 384
    self.dropout = nn.Dropout(dropout)

  def forward(self,x):
    x = torch.cat([head(x) for head in self.heads],dim=-1)
    # each head outputs (BS, SL, head_size)
    x = self.combine(x) # (BS, SL, 384)
    x = self.dropout(x)
    return x
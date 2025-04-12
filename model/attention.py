import torch
import torch.nn as nn
from torch.nn import functional as F
from utilities.config import context, embed_size, BIAS, dropout

class AttentionHead(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.queries = nn.Linear(embed_size, head_size, bias=BIAS)
    self.keys = nn.Linear(embed_size, head_size, bias=BIAS)
    self.values = nn.Linear(embed_size, head_size, bias=BIAS)
    # mask out knowledge about future tokens
    self.register_buffer('tril',torch.tril(torch.ones(context,context)))
    self.dropout = nn.Dropout(dropout)

  def forward(self,x):
    BS, SL, VS =  x.shape
    q = self.queries(x) # BS, SL, 54
    k = self.keys(x) # BS, SL, 54
    v = self.values(x) # BS, SL, 54

    attention_w = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # BS, SL, SL
    attention_w = attention_w.masked_fill(self.tril[:SL,:SL]==0, float('-inf'))
    attention_scores = F.softmax(attention_w,dim=-1) # BS, SL, SL

    x = attention_scores @ v # BS, SL, 54
    return x



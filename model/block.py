import torch.nn as nn
from utilities.config import embed_size
from transformer import MultiHead, ForwardLayer

class Block(nn.Module):
  def __init__(self, n_heads):
    super().__init__()
    # dimensionality of each of the heads
    head_size = embed_size // n_heads
    self.ma = MultiHead(n_heads, head_size)
    self.feed_forward = ForwardLayer(embed_size)
    self.ln1 = nn.LayerNorm(embed_size)
    self.ln2 = nn.LayerNorm(embed_size)

  def forward(self,x):
    #ipdb.set_trace()
    # communication
    x = x + self.ma(self.ln1(x))
    # computation
    x = x + self.feed_forward(self.ln2(x))
    return x
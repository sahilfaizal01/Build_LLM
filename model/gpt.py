import torch
import torch.nn as nn
from torch.nn import functional as F
from utilities.config import context, embed_size, n_layers, n_heads, BIAS, device, vocab_size
from block import Block

class GPT(nn.Module):
  def __init__(self):
    super().__init__()
    self.embeddings = nn.Embedding(vocab_size, embed_size) # e.g: 4096 x 384
    self.positions = nn.Embedding(context, embed_size) # e.g: 512 x 384
    # * to unpack elements of a list and pass them as individual args to a function
    self.blocks = nn.Sequential(*[Block(n_heads) for _ in range(n_layers)])
    self.ln = nn.LayerNorm(embed_size)
    # imaging probability of the next token from this vocab representation
    self.final_linear = nn.Linear(embed_size, vocab_size, bias=BIAS) # e.g: 384 x 4096
    self.apply(self._init_weights)

  # parameter initalization - gaussian distribution
  # sampling from a gaussian prob distribution involves generating random values that follow the distribution's characteristic bell curve
  def _init_weights(self,module):
    if isinstance(module, nn.Linear):
      # _ is in place apply and update operator
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module,nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, input, targets=None):
    # BS - Batch Size, SL - Sequence Length
    loss = None
    BS, SL = input.shape # BS x SL
    emb = self.embeddings(input) # BS x SL X 384
    pos = self.positions(torch.arange(SL, device=device)) # SL x 384
    x = emb + pos # BS x SL X 384
    x = self.blocks(x) # BS X SL X 384
    x = self.ln(x) # BS x SL X 384 (Embedding Size)
    logits = self.final_linear(x) # BS x SL X 4096 (vocab size)

    if targets is not None:
      BS, SL, VS = logits.shape
      logits = logits.view(BS*SL, VS)
      targets = targets.view(BS*SL)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self,input, max=500):
    for _ in range(max):
      # taking maximum of last 512 tokens from input
      input = input[:,-context:] # (1,input length until max of SL)
      logits, _ = self(input) # (1, input length, 4096)
      logits = logits[:,-1,:] #selecting last prediction
      probs = F.softmax(logits, dim=-1) # across last dim (1,4096)
      next = torch.multinomial(probs, num_samples=1) #sample from prob distribution
      input = torch.cat((input, next),dim=1)
    return input
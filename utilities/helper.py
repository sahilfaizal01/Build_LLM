import torch
# tokenizer
import sentencepiece as spm
from config import weight_decay, device

# Return a batch of either training or evaluation data
def get_batch(split,device,context,train_data,val_data,batch_size):
    # BS = Batch Size / SL = Sequence Length or context length
    data = train_data if split=="train" else val_data # Select the split
    inds = torch.randint(len(data)-context, (batch_size,)) # (BS)
    x = torch.stack([data[i: i+context] for i in inds]) # (BS,SL)
    y = torch.stack([data[i+1: i+context+1] for i in inds]) # (BS,SL)

    x,y = x.to(device), y.to(device)
    return x,y

# encoding and decoding tokenizer functions
def encode_decode_tokenizer():
    sp = spm.SentencePieceProcessor(model_file='tokenizer/wiki_tokenizer.model')
    vocab_size = sp.get_piece_size()
    # text to nums
    encode = lambda s: sp.Encode(s)
    # nums to text
    decode = lambda l: sp.Decode(l)
    return encode, decode, vocab_size

def get_optimizer_groups(model):
        # Set Weight Decay differently for different kinds of parameters
    p_dict = {p_name: p for p_name, p in model.named_parameters() if p.requires_grad}
    # isolate weight matrices as they benefit specially from weight decay
    weight_decay_p = [p for n, p in p_dict.items() if p.dim() >= 2]  
    # isolate other parameters like bias parameters, that don't benefit from weight decay
    no_weight_decay_p = [p for n, p in p_dict.items() if p.dim() < 2] 
    # store the parameter types in a list of dictionaries
    optimizer_groups = [
        {'params': weight_decay_p, 'weight_decay': weight_decay},
        {'params': no_weight_decay_p, 'weight_decay': 0.0}
    ]
    return optimizer_groups

# Generate a new sample
@torch.no_grad()
def generate_sample(input,model):
    encode, decode, _ = encode_decode_tokenizer()
    t1 = torch.tensor(encode(input), dtype=torch.long, device=device) # Tokenize string -> (tensor of ids)
    t1 = t1[None,:]  # (1 , [size of ids])
    newgen = model.generate(t1,max=64)[0].tolist() # call the generate method, limit output size
    result=decode(newgen) # decode the result with the tokenizer to get back characters
    print(f"{result}")
import torch
import os
from utilities.helper import encode_decode_tokenizer

def load_data():
    # load the wiki dataset
    with open('wiki.txt','r',encoding='utf-8') as f:
        text = f.read()

    encode, _, _ = encode_decode_tokenizer()

    # tokenization of the dataset
    if os.path.exists(f"encoded_data.pt"):
        # load the encoded data if it exists
        data = torch.load("encoded_data.pt") 
    else:
        # do the encoding
        data = torch.Tensor(encode(text),dtype=torch.long) 
        # save the encoded data
        torch.save(data,'encoded_data.pt')
    
    return data

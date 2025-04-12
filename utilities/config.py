import torch
from main import model
from utilities.helper import encode_decode_tokenizer, get_optimizer_groups

# ARCHITECTURE_PARAMS
# vocab size
_, _, vocab_size = encode_decode_tokenizer()
batch_size = 64 # 8 to 128 and beyond
# max no. of tokens that can be processed
context = 512
# dimension of each token
embed_size = 384
# no. of transformer blocks
n_layers = 7
# multi-head attention
n_heads = 7
BIAS = True

# HYPERPARAMETERS
lr = 3e-4
dropout = 0.05
weight_decay = 0.01
grad_clip = 1.0

# TRAINING parameter
train_iters = 100000
# after every 50 itr evaluate the test performance
eval_interval = 50
# average over multiple batches, rather than just calculating the loss over a single batch
eval_iters = 10
# lower amount of memory, faster training

compile = False
checkpoint_dir = 'models/'
checkpoint_fn = 'latest.pt'
checkpoint_load_fn = 'latest.pt'
dtype = torch.bfloat16

# DEVICE
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Available device is:",device)


# optimizer parameters
optimizer_groups = get_optimizer_groups(model)
# Declare optimizer and apply weight decay
optimizer = torch.optim.AdamW(optimizer_groups, lr=lr, betas=(0.9, 0.99))
# Declare scheduler to change learning rate through the training
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_iters, eta_min=lr/10)

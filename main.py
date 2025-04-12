import torch
import os
from data.ingest import load_data
from model.gpt import GPT
from train import trainer
from inference import infer
from utilities.config import device, dtype, optimizer, checkpoint_dir, checkpoint_load_fn

MODE = "TRAIN" # TRAIN or INFERENCE 

data = load_data()
print("Data loaded successfully")
data_size=len(data)
print("Dataset Size:",data_size)

# Split the data into train and test sets
# set the split at 90%-10% (train-test)
spl = int(0.9*data_size)
train_data=data[:spl] 
val_data=data[spl:] 
print(f'Total data: {data_size/1e6:.2f} Million | Training: {len(train_data)/1e6:.2f} Million | Validation: {len(val_data)/1e6:.2f} Million')

# TRAINING STEP 
model = GPT()
model = model.to(dtype)
model = model.to(device)

if compile:
  print("Torch is compiling the model")
  model = torch.compile(model)

print(sum(p.numel() for p in model.parameters()) / 1e6, " Million Parameters")

# Load checkpoint if available
def load_checkpoint(path):
  print("LLM - Loading model")
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  iteration = checkpoint['iteration']
  loss = checkpoint['loss']
  print(f"Loaded iter {iteration} with loss {loss}")
  return iteration, loss

# Mode selection
if MODE == "TRAIN":
  print("LLM - Training mode")
  trainer()
  print("Starting training...")
elif MODE == "INFERENCE":
  print("LLM - Inference mode")
  if os.path.exists(f"{checkpoint_dir}/{checkpoint_load_fn}"):
    start_iteration, loss = load_checkpoint(checkpoint_dir + checkpoint_load_fn)
    best_val_loss = loss
  infer()


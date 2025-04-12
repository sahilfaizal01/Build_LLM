import torch
import sys
import torch.nn as nn
import wandb
from tqdm import tqdm
from main import model
from logger import wandb_log
from utilities.helper import get_batch
from utilities.config import eval_iters, eval_interval, train_iters, optimizer, scheduler, checkpoint_dir, checkpoint_fn, grad_clip

# start training
start_iteration = 0
# Track best loss value
best_val_loss = float('inf')  

# Calculate loss averages
@torch.no_grad()
def calculate_loss(model):
  out = {}
  model.eval()
  for split in ['train','eval']:
    l = torch.zeros(eval_iters)
    for i in range(eval_iters):
      x, y = get_batch(split)
      _, loss = model(x,y)
      l[i] = loss
    out[split] = l.mean().item()
  model.train()
  return out

# Training loop
def trainer():
    try:
        for i in tqdm(range(start_iteration, train_iters)):
            xb, yb = get_batch("train")
            logits, loss = model(xb,yb)

            if(i % eval_interval == 0 or i == train_iters-1):
                l = calculate_loss()
                print(f"\n{i}: train loss: {l['train']} / val loss: {l['eval']}")
                # generate_sample("Once there lived a")

                if l['eval'] < best_val_loss:
                    best_val_loss = l['eval']
                    print('[CHECKPINT]: Saving with loss: ', best_val_loss)
                    torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_val_loss,
                        'iteration': i,
                    }, checkpoint_dir + checkpoint_fn)

                if wandb_log:
                    wandb.log({
                    "loss/train": l['train'],
                    "loss/val": l['eval'],
                    "lr": scheduler.get_last_lr()[0]
                },
                        step=i)

            optimizer.zero_grad(set_to_none = True)
            loss.backward()

            nn.utils.clip_grad_norm(model.parameters(),max_norm=grad_clip)

            optimizer.step()
            scheduler.step()

        if wandb_log:
            wandb.finish()

    except KeyboardInterrupt:
        print("Training interrupted. Cleaning memory..")

    finally:
        # Release GPU memory
        torch.cuda.empty_cache()
        print("GPU memory is released")
        sys.exit(0)

    torch.cuda.empty_cache()
import wandb
from datetime import datetime

wandb_log = True
wandb_project = "LLM_Training"
wandb_run_name = "llm1-"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

if wandb_log:
  wandb.init(project=wandb_project,name=wandb_run_name)
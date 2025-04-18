# tLM - A tiny LLM from Scratch

An implementation of GPT-style LLM Model with 20M parameters from scratch using PyTorch in Python

![image](https://github.com/user-attachments/assets/c7d4604d-d528-44a4-8483-1dd4f4f9829a)

## Insights:
- **No. of parameters:** 19.83 Million (~20 Million)
- **Data Type:** FP16
- **Best Loss:** 2.267 (Initial: 8.375)
- **Total Data Size:** 59.31 Million (Training: 53.29 M and Validation: 5.92M)
- **No. of transformer heads:** 7
- **Context Window:** 512 tokens
- **Embedding Dimension:** 384
- **Tokenizer Vocab Size:** 4096
- **train_iters** = 100000

## Dataset:
Opensource Wikipedia Data

## Train Config:
* **Optimizer:** AdamW (Adam with Weight Decay)
* **Scheduler:** CosineAnnealingLR

## Tools:
- **PyTorch** (Deep Learning Framework)
- **Python** (Programming)
- **Weights and Biases** (Experiment Tracking)

## Next Steps:
- Complete Notes.md
- Distributed training using Deepseed
- Integrate interpretability
- Make it more advanced (latest attention mechanisms)
- Incorporate RL based alignment
- Optimization techniques + On-device deployment

## Training Charts:
### 1. Train Loss Curve
![image](https://github.com/user-attachments/assets/7203a337-0b1a-49cb-b753-cab7b6d01d62)
### 2. Val Loss Curve
![image](https://github.com/user-attachments/assets/a0faf203-80c5-4b5c-ac02-e61aa42e5372)



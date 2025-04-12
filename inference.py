import sys
from main import model
from utilities.helper import generate_sample

# Inference
def infer():
    print("LLM - Inference mode")
    model.eval()
    while True:
        qs = input("Enter text (q to quit): ")
        if qs == "":
            continue
        if qs == "q":
            break
        generate_sample(qs, model)
    sys.exit()
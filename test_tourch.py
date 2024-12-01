import torch

# Check no_grad
print("Torch no_grad:", hasattr(torch, "no_grad"))

# Simple tensor operation
x = torch.tensor([1.0, 2.0, 3.0])
with torch.no_grad():
    y = x * 2
print("Output:", y)

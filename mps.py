import torch

print(torch.backends.mps.is_available())  # True if your Mac GPU works
print(torch.backends.mps.is_built())      # True if PyTorch has MPS compiled in

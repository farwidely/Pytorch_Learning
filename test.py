import torch
import torch.nn as nn
import torchvision

x = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=float)
print(x)
x.requires_grad_(True)
y = x.view(1, 6, 2)
print(type(x))

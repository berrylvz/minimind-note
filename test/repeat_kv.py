import torch
from torch import nn

a = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

print(a)

s1, s2 = a.shape

b = a[:, None, :].expand(s1, 3, s2).reshape(s1 * 3, s2)

print(b)

import torch
from torch import nn


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        # eps is a small constant to avoid division by zero
        self.eps = eps
        # weight is a learnable parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

def test_rmsnorm():
    rmsnorm = RMSNorm(3)
    x = torch.randn(2, 3)
    y = rmsnorm(x)
    assert y.shape == x.shape
    print(x)
    print(y)
    print(nn.Parameter(torch.ones(3)))

test_rmsnorm()

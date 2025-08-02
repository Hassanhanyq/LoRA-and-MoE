import torch.nn as nn

class LoRAExpert(nn.Module):
    def __init__(self, in_features,out_features, rank, alpha=8, dtype=None):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False, dtype=dtype)
        self.B = nn.Linear(rank, out_features, bias=False, dtype=dtype)
        self.scaling = alpha / rank
        self.dtype=dtype
        nn.init.kaiming_uniform_(self.A.weight, a=0)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.B(self.A(x)) * self.scaling

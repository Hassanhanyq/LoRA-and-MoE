import torch.nn as nn

class LoRAExpert(nn.Module):
    def __init__(self, hidden_size, rank, alpha=8):
        super().__init__()
        self.A = nn.Linear(hidden_size, rank, bias=False)
        self.B = nn.Linear(rank, hidden_size, bias=False)
        self.scaling = alpha / rank

    def forward(self, x):
        return self.B(self.A(x)) * self.scaling

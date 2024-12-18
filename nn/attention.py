# torch imports
import torch
import torch.nn as nn
# misc libraries
import math
from dataclasses import dataclass
from pydantic import validate_arguments


@validate_arguments
@dataclass(eq=False, repr=False)
class MHA(nn.Module):

    model_dim: int
    num_heads: int = 1

    def __post_init__(self):
        super().__init__()
        self.W_Q = nn.Linear(self.model_dim, self.model_dim)
        self.W_K = nn.Linear(self.model_dim, self.model_dim)
        self.W_V = nn.Linear(self.model_dim, self.model_dim)
        # Output projection layer
        self.out = nn.Linear(self.model_dim, self.model_dim)
    
    def scaled_dot_product(self, Q, K, V):
        # Q: B, L, D
        # K: B, L, D
        # V: B, L, D

        # First we get an product of the matrices, resulting in B, L, L
        return torch.softmax((Q @ K.transpose(1, 2)) / math.sqrt(self.model_dim), axis=-1) @ V
    
    def forward(self, x):
        # Input x looks like B, L, D
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        # Then we do a scaled dot-product
        z = self.scaled_dot_product(Q, K, V)
        # Do the final projection
        z_out = self.out(z)

        return z_out


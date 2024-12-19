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

    d_model: int
    num_heads: int = 1

    def __post_init__(self):
        super().__init__()
        # Make sure that we can divid the feature dim by the num of heads.
        assert self.d_model % self.num_heads == 0, "Num heads must cleanly divide the feature dim."
        
        self.d_head = self.d_model // self.num_heads
        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)
        # Output projection layer
        self.out = nn.Linear(self.d_model, self.d_model)
    
    def split_heads(self,):
    

    def combine_heads(self,):


    def scaled_dot_product(self, Q, K, V):
        # Q: B, L, D
        # K: B, L, D
        # V: B, L, D

        # First we get an product of the matrices, resulting in B, L, L
        return torch.softmax((Q @ K.transpose(1, 2)) / math.sqrt(self.d_model), axis=-1) @ V
    
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


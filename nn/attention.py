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
    
    def split_heads(self, x):
        # The goal of this function is to take an input tensor 
        # of shape B, L, D and turn it into B, L, H, D_h.
        B, L, _ = x.size()
        # First split the last dimension into num_heads many d_head feature vecs.
        x = x.view(B, L, self.num_heads, self.d_head)
        # Then we transpose the middle two dimensions so that we can do L x L
        x = x.transpose(1, 2)
        return x

    def combine_heads(self, x):
        # The goal of this function is to take an input tensor 
        # of shape B, H, L, D_h and turn it into B, L, D.
        B, H, L, _ = x.size()
        x = x.transpose(1, 2).contiguous() # Always do a contiguous before view.
        x = x.view(B, L, self.d_model)
        return x

    def scaled_dot_product(self, Q, K, V):
        # Q: B, H, L, D_h
        # K: B, H, L, D_h
        # V: B, H, L, D_h
        prod = Q @ K.transpose(-2, -1) # B, H, L, L
        # Then we need to renormalize by the root of D_h because each row the dot scales by D_h.
        scaled_prod = prod / math.sqrt(self.d_head)
        # Then we apply the softmax over the keys.
        weights = torch.softmax(scaled_prod, dim=-1) # B, H, L, L
        # Finally we multiply the weights by V to get the output.
        z = weights @ V
        return z
    
    def forward(self, x):
        # Input x looks like B, L, D
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        # Then we do a scaled dot-product
        att = self.scaled_dot_product(Q, K, V)

        # Finally we combine the heads
        z = self.combine_heads(att)

        # Do the final projection
        z_out = self.out(z)

        return z_out


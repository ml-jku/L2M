"""
Adjusted from the TF implementation provided by: 
- https://github.com/google-research/robotics_transformer/blob/master/tokenizers/token_learner.py

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpBlock(nn.Module):

  def __init__(self, mlp_dim, out_dim, dropout_rate: float = 0.1):
    """
    Initializer for the MLP Block.

    This computes outer_dense(gelu(hidden_dense(input))), with dropout
    applied as necessary.

    Args:
      mlp_dim: The dimension of the inner representation (output of hidden
        layer). Usually larger than the input/output dim.
      out_dim: The output dimension of the block. If None, the model output dim
        is equal to the input dim (usually desired)
      dropout_rate: Dropout rate to be applied after dense ( & activation)
      
    """
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(mlp_dim, mlp_dim),
      nn.GELU(),
      nn.Dropout(dropout_rate),
      nn.Linear(mlp_dim, out_dim),
      nn.Dropout(dropout_rate)
    )
    
  def forward(self, x):
    return self.net(x)


class TokenLearnerModule(nn.Module):
  """TokenLearner module V1.1 (https://arxiv.org/abs/2106.11297)."""

  def __init__(self,
               num_tokens: int = 8,
               bottleneck_dim: int = 64,
               dropout_rate: float = 0.):
    super().__init__()
    self.mlp = MlpBlock(mlp_dim=bottleneck_dim, out_dim=num_tokens, dropout_rate=dropout_rate)
    self.layernorm = nn.LayerNorm(bottleneck_dim)

  def forward(self, x):
    if len(x.shape) == 4:
      batch_size, height, width, channels = x.shape
      x = x.reshape(batch_size, height * width, channels)

    selected = self.layernorm(x)
    # Shape: [bs, h*w, n_token].
    selected = self.mlp(x)  
    # Shape: [bs, n_token, h*w].
    selected = selected.permute(0, 2, 1)  
    selected = F.softmax(selected, dim=-1)

    # Shape: [bs, n_token, c]
    return torch.einsum("...si,...id->...sd", selected, x)

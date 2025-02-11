import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleLinearLayer(nn.Module):
  def __init__(self):
    super(SingleLinearLayer, self).__init__()
    self.flatten = nn.Flatten()  # This layer flattens the input to a 1D tensor
    self.linear = nn.Linear(256 * 256 * 3, 1)  # 256 * 256 * 3 is the number of input features

  def forward(self, x):
    x = self.flatten(x)  # Flatten the input
    x = torch.relu(
      self.linear(x)
    )  # Pass the input through the linear layer
    return x  # The output size is 1, which is the predicted value

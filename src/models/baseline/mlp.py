import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
  def __init__(self):
    super(BaselineMLP, self).__init__()
    self.flatten = nn.Flatten()  # This layer flattens the input to a 1D tensor
    self.fc1 = nn.Linear(256 * 256 * 3, 128)  # 256 * 256 * 3 is the number of input features
    self.fc2 = nn.Linear(128, 64)  # 128 is the number of output features from the previous layer
    self.fc3 = nn.Linear(64, 1)  # 64 is the number of output features from the previous layer

  def forward(self, x):
    x = self.flatten(x)  # Flatten the input
    x = torch.relu(
      self.fc1(x) # Pass the input through the first fully connected layer
    )  # Apply the ReLU activation function to the output of the first fully connected layer
    x = torch.relu(
      self.fc2(x)  # Pass the output of the first fully connected layer through the second fully connected layer
    )  # Apply the ReLU activation function to the output of the second fully connected layer
    x = torch.relu(
      self.fc3(x)
    )  # Pass the output of the second fully connected layer through the third fully connected layer
    return x  # The last layer has an output size of 1, which is the predicted value

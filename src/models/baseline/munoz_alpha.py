import torch
import torch.nn as nn


class AlphaMunoz(nn.Module):
    def __init__(self):
        super(BaselineMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 256 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.ln1 = nn.LayerNorm(128)  # Layer normalization after the first fully connected layer
        self.ln2 = nn.LayerNorm(64)   # Layer normalization after the second fully connected layer

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.ln1(x)  # Apply Layer Normalization
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.ln2(x)  # Apply Layer Normalization
        x = self.dropout(x)
        x = self.fc3(x)
        return x
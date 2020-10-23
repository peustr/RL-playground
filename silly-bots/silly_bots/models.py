import torch.nn as nn
import torch.nn.functional as f


class B8MLP(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, D)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvModel(nn.Module):
    """A convultional model"""
    def __init__(self, n_labels):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 128, 4)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, n_labels)

    def forward(self, input):
        # input is of shape (N, 1025, w)
        y = torch.unsqueeze(input, 1)
        # y is of the shape (N, 1, 1025, w)
        y = F.relu(self.conv1(y))
        # y is of shape (N, 64, 1005, w-20)
        y = F.max_pool2d(y, 8)
        # y is of the shape (N, 64, ~250, ~w/4
        y = F.relu(self.conv2(y))
        # y is of the shape (N, 128, ~230, ~w/4-20)
        y = F.adaptive_max_pool2d(y, 4)
        # y is of the shape (N, 128, 8, 8)
        y = torch.flatten(y, 1)
        # y is of the shape (N, 128 * 8 * 8)
        y = F.relu(self.fc1(y))
        # y is of the shape (N, 512)
        y = self.fc2(y)
        return y

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvModel(nn.Module):
    """A convultional model"""
    def __init__(self, n_labels):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 20)
        self.conv2 = nn.Conv2d(64, 1024, 20)
        self.fc1 = nn.Linear(1024 * 20 * 20, 512)
        self.fc2 = nn.Linear(512, n_labels)

    def forward(self, input):
        # input is of shape (N, 1025, w)
        y = F.relu(self.conv1(input))
        # y is of shape (N, 64, 1005, w-20)
        y = F.max_pool2d(y, 4)
        # y is of the shape (N, 64, ~250, ~w/4
        y = F.relu(self.conv2(y))
        # y is of the shape (N, 1024, ~230, ~w/4-20)
        y = F.adaptive_max_pool2d(y, 20)
        # y is of the shape (N, 1024, 20, 20)
        y = torch.flatten(y, 1)
        # y is of the shape (N, 1024 * 20 * 20)
        y = F.relu(self.fc1(y))
        # y is of the shape (N, 512)
        y = F.softmax(self.fc2(y))
        return y

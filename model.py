import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

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

class HarmonicModel(nn.Module):
    """A model using the root note and harmonics"""
    def __init__(self, n_labels, n_harmonics, kernel_size=10, num_filter_maps=16, dropout=0.5):
        super(HarmonicModel, self).__init__()
        self.conv = nn.Conv1d(n_harmonics+1, num_filter_maps, kernel_size,
                              padding=int(math.floor(kernel_size/2)))
        xavier_uniform_(self.conv.weight)
        self.u = nn.Linear(num_filter_maps, 1)
        xavier_uniform_(self.u.weight)
        self.final = nn.Linear(num_filter_maps, n_labels)
        xavier_uniform_(self.final.weight)

    def forward(self, input):
        conved = torch.tanh(self.conv(input.transpose(1,2)))
        attention = F.softmax(torch.matmul(self.u.weight, conved), dim=2)
        v = torch.matmul(attention, conved.transpose(1,-1))
        y = torch.sigmoid(self.final(v)).squeeze()
        return y

import torch
import torch.nn as nn
import torch.nn.functional as F
from Network.base import BaseNetwork

class LeNet(BaseNetwork):
    def __init__(self, *args, **kwargs):
        super(LeNet, self).__init__(*args, **kwargs)
        self._build()

    def _build(self):
        self.conv1 = nn.Conv2d(in_channels=self.input_channel, out_channels=6, kernel_size=(5, 5),
                               stride=1, padding='valid',)
        self.relu1 = nn.ReLU(inplace=True)
        self.pooling1 = nn.MaxPool2d(kernel_size=3, stride=None, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5),
                               stride=1, padding='valid', )
        self.relu2 = nn.ReLU(inplace=True)
        self.pooling2 = nn.MaxPool2d(kernel_size=3, stride=None, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(300, 120)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(84, self.output_shape)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pooling1(x)
        x = self.relu2(self.conv2(x))
        x = self.pooling2(x)
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x
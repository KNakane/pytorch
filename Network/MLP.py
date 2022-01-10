import torch
import torch.nn as nn
import torch.nn.functional as F
from Network.base import BaseNetwork


class MLP(BaseNetwork):
    def __init__(self, *args, **kwargs):
        super(MLP, self).__init__(*args, **kwargs)
        self._build()

    def _build(self):
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.input_height * self.input_width * self.input_channel, 1024)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(512, self.output_shape)

    def forward(self, x):
        if x.ndim > 2:
            x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

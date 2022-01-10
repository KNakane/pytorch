import torch.nn as nn

class BaseNetwork(nn.Module):
    def __init__(self,
                 input_height,
                 input_width,
                 input_channel,
                 output_shape):
        super(BaseNetwork, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel
        self.output_shape = output_shape

    def _build(self):
        pass

    def forward(self, x):
        raise NotImplementedError()
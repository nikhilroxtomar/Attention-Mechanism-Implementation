import torch
import torch.nn as nn

"""
Implementation of Squeeze and Excitation Network in the PyTorch.
Paper: https://arxiv.org/pdf/1709.01507.pdf
Blog: https://idiotdeveloper.com/squeeze-and-excitation-networks
"""

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel, ratio=8):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.network = nn.Sequential(
            nn.Linear(channel, channel//ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//ratio, channel,  bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        b, c, _, _ = inputs.shape
        x = self.avg_pool(inputs)
        x = x.view(b, c)
        x = self.network(x)
        x = x.view(b, c, 1, 1)
        x = inputs * x
        return x

if __name__ == "__main__":
    inputs = torch.randn((8, 32, 128, 128))
    se = SqueezeAndExcitation(32, ratio=8)
    y = se(inputs)
    print(y.shape)

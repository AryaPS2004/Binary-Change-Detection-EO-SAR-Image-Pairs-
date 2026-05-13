import torch.nn as nn


class SegmentationHead(nn.Module):

    def __init__(self, in_channels=128):

        super().__init__()

        self.head = nn.Conv2d(
            in_channels,
            1,
            kernel_size=1
        )

    def forward(self, x):

        return self.head(x)
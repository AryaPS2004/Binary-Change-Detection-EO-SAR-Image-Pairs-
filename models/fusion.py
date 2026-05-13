import torch
import torch.nn as nn


class FeatureFusion(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1
            ),

            nn.BatchNorm2d(channels),

            nn.ReLU(inplace=True)

        )

    def forward(self, eo_feat, sar_feat):

        fused = torch.cat(
            [eo_feat, sar_feat],
            dim=1
        )

        fused = self.conv(fused)

        return fused
import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):

    def __init__(self, in_channels=3, pretrained=True):

        super().__init__()

        backbone = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
            if pretrained else None
        )

        # -----------------------------------
        # Modify first conv layer
        # -----------------------------------

        if in_channels != 3:

            backbone.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )

        # -----------------------------------
        # Encoder stages
        # -----------------------------------

        self.initial = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )

        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):

        features = []

        x = self.initial(x)
        features.append(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        features.append(x)

        x = self.layer2(x)
        features.append(x)

        x = self.layer3(x)
        features.append(x)

        x = self.layer4(x)
        features.append(x)

        return features
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm


# Double Convolution Block


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)


# Attention Fusion Block


class AttentionFusion(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.attention = nn.Sequential(

            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1
            ),

            nn.Sigmoid()
        )

    def forward(self, eo_feat, sar_feat):

        combined = torch.cat(
            [eo_feat, sar_feat],
            dim=1
        )

        attention = self.attention(combined)

        fused = (
            attention * eo_feat +
            (1 - attention) * sar_feat
        )

        return fused



# Decoder Block


class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):

        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:

            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        x = torch.cat([x, skip], dim=1)

        x = self.conv(x)

        return x



# EO-SAR UNet

class EOSARUNet(nn.Module):

    def __init__(self):

        super().__init__()

    
        # EO Encoder


        self.eo_encoder = timm.create_model(
            "resnet50",
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4)
        )

        # SAR Encoder


        self.sar_encoder = timm.create_model(
            "resnet50",
            pretrained=False,
            in_chans=1,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4)
        )

        # Attention Fusion


        self.fuse1 = AttentionFusion(64)

        self.fuse2 = AttentionFusion(256)

        self.fuse3 = AttentionFusion(512)

        self.fuse4 = AttentionFusion(1024)

        self.fuse5 = AttentionFusion(2048)


        # Decoder


        self.decoder4 = DecoderBlock(
            2048,
            1024,
            1024
        )

        self.decoder3 = DecoderBlock(
            1024,
            512,
            512
        )

        self.decoder2 = DecoderBlock(
            512,
            256,
            256
        )

        self.decoder1 = DecoderBlock(
            256,
            64,
            64
        )

        # Final Segmentation Head
   

        self.segmentation_head = nn.Sequential(

            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                1,
                kernel_size=1
            )
        )

    def forward(self, eo, sar):


        # EO Features


        eo_feats = self.eo_encoder(eo)


        # SAR Features
        

        sar_feats = self.sar_encoder(sar)


        # Attention Fusion


        f1 = self.fuse1(eo_feats[0], sar_feats[0])

        f2 = self.fuse2(eo_feats[1], sar_feats[1])

        f3 = self.fuse3(eo_feats[2], sar_feats[2])

        f4 = self.fuse4(eo_feats[3], sar_feats[3])

        f5 = self.fuse5(eo_feats[4], sar_feats[4])


        # Decoder


        d4 = self.decoder4(f5, f4)

        d3 = self.decoder3(d4, f3)

        d2 = self.decoder2(d3, f2)

        d1 = self.decoder1(d2, f1)

   
        # Final Output


        out = self.segmentation_head(d1)

        out = F.interpolate(
            out,
            size=eo.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return out
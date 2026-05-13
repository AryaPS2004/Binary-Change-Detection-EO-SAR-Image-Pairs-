import torch

from models.network import EOSARUNet


model = EOSARUNet()

eo = torch.randn(2, 3, 256, 256)

sar = torch.randn(2, 1, 256, 256)

output = model(eo, sar)

print(output.shape)
import torch

from losses.focal_dice import FocalDiceLoss


criterion = FocalDiceLoss()

logits = torch.randn(2, 1, 256, 256)

targets = torch.randint(
    0,
    2,
    (2, 1, 256, 256)
).float()

loss = criterion(logits, targets)

print("Loss:", loss.item())
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalDiceLoss(nn.Module):

    def __init__(
        self,
        alpha=0.75,
        gamma=2.0,
        dice_weight=0.5
    ):

        super().__init__()

        self.alpha = alpha

        self.gamma = gamma

        self.dice_weight = dice_weight

    # Focal Loss
  

    def focal_loss(self, logits, targets):

        targets = targets.float()

        probs = torch.sigmoid(logits)

        pos_weight = torch.tensor(
            [4.0],
            device=logits.device
        )

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=pos_weight
        )

        pt = probs * targets + (1 - probs) * (1 - targets)

        alpha_factor = (
            self.alpha * targets +
            (1 - self.alpha) * (1 - targets)
        )

        focal = alpha_factor * ((1 - pt) ** self.gamma) * bce

        return focal.mean()

  
    # Dice Loss


    def dice_loss(self, logits, targets, eps=1e-6):

        targets = targets.float()

        probs = torch.sigmoid(logits)

        probs = probs.view(-1)

        targets = targets.view(-1)

        intersection = (probs * targets).sum()

        dice = (
            (2 * intersection + eps) /
            (probs.sum() + targets.sum() + eps)
        )

        return 1 - dice

    # Combined Loss


    def forward(self, logits, targets):

        focal = self.focal_loss(
            logits,
            targets
        )

        dice = self.dice_loss(
            logits,
            targets
        )

        total_loss = (
            focal +
            self.dice_weight * dice
        )

        return total_loss
import torch

from metrics.segmentation_metrics import compute_metrics


logits = torch.randn(2, 1, 256, 256)

targets = torch.randint(
    0,
    2,
    (2, 1, 256, 256)
)

metrics = compute_metrics(
    logits,
    targets
)

print(metrics)
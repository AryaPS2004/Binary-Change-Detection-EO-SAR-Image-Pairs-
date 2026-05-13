import torch


def compute_metrics(
    logits,
    targets,
    threshold=0.5,
    eps=1e-6
):

    probs = torch.sigmoid(logits)

    preds = (probs >= threshold).float()

    targets = targets.float()

    preds = preds.view(-1)

    targets = targets.view(-1)

  

    TP = ((preds == 1) & (targets == 1)).sum().float()

    FP = ((preds == 1) & (targets == 0)).sum().float()

    FN = ((preds == 0) & (targets == 1)).sum().float()

    # Metrics
 
    precision = TP / (TP + FP + eps)

    recall = TP / (TP + FN + eps)

    f1 = (
        2 * precision * recall /
        (precision + recall + eps)
    )

    iou = TP / (TP + FP + FN + eps)

    return {
        "IoU": iou.item(),
        "Precision": precision.item(),
        "Recall": recall.item(),
        "F1": f1.item()
    }
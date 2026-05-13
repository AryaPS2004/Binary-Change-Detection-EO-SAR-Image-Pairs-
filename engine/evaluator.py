import torch

from tqdm import tqdm

from metrics.segmentation_metrics import compute_metrics


def evaluate_model(
    model,
    dataloader,
    device
):

    model.eval()

    total_metrics = {
        "IoU": 0.0,
        "Precision": 0.0,
        "Recall": 0.0,
        "F1": 0.0
    }

    with torch.no_grad():

        for batch in tqdm(dataloader):

            eo = batch["eo"].to(device)

            sar = batch["sar"].to(device)

            mask = batch["mask"].to(device)

            mask = mask.unsqueeze(1).float()

            logits = model(eo, sar)

            metrics = compute_metrics(
                logits,
                mask
            )

            for key in total_metrics:

                total_metrics[key] += metrics[key]

    for key in total_metrics:

        total_metrics[key] /= len(dataloader)

    return total_metrics
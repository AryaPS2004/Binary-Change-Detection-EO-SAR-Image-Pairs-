import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import EOSARDataset
from data.transforms import get_val_transforms

from models.network import EOSARUNet

from utils.checkpoint import load_checkpoint


def save_prediction(
    eo,
    sar,
    mask,
    pred,
    save_path
):

    fig, ax = plt.subplots(1, 4, figsize=(18, 5))

  
    # EO Image
   

    eo = eo.cpu().numpy().transpose(1, 2, 0)

    eo = (eo - eo.min()) / (
        eo.max() - eo.min() + 1e-6
    )

    ax[0].imshow(eo)

    ax[0].set_title("EO Image")

    ax[0].axis("off")

  
    # SAR Image


    sar = sar.cpu().numpy()[0]

    ax[1].imshow(sar, cmap="gray")

    ax[1].set_title("SAR Image")

    ax[1].axis("off")

  
    # Ground Truth


    ax[2].imshow(mask, cmap="gray")

    ax[2].set_title("Ground Truth")

    ax[2].axis("off")

  
    # Prediction

    ax[3].imshow(pred, cmap="gray")

    ax[3].set_title("Prediction")

    ax[3].axis("off")

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


def run_inference(config, device):



    dataset = EOSARDataset(
        root_dir=config["DATA"]["ROOT"],
        split="test",
        transforms=get_val_transforms(
            config["DATA"]["IMAGE_SIZE"]
        )
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

   
    # Model


    model = EOSARUNet()

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters()
    )

    checkpoint_path = os.path.join(
        config["CHECKPOINT"]["SAVE_DIR"],
        "best_model.pth"
    )

    load_checkpoint(
        model,
        optimizer,
        checkpoint_path,
        device
    )

    model.eval()

    print("Checkpoint loaded successfully!")



    os.makedirs(
        config["OUTPUT"]["VIS_DIR"],
        exist_ok=True
    )

    threshold = config["THRESHOLD"]["DEFAULT"]


    # Inference Loop
   
    with torch.no_grad():

        for idx, batch in enumerate(tqdm(dataloader)):

            eo = batch["eo"].to(device)

            sar = batch["sar"].to(device)

            mask = batch["mask"][0].numpy()

          
            # Forward Pass
        

            logits = model(eo, sar)

            probs = torch.sigmoid(logits)

            pred = (
                probs >= threshold
            ).float()

            pred = pred[0, 0].cpu().numpy()

      
            # Save Visualization


            save_path = os.path.join(
                config["OUTPUT"]["VIS_DIR"],
                f"prediction_{idx}.png"
            )

            save_prediction(
                eo[0],
                sar[0],
                mask,
                pred,
                save_path
            )

    print("Inference completed!")
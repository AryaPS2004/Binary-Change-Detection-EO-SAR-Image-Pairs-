import os

import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm
from engine.evaluator import evaluate_model

from data.dataset import EOSARDataset
from data.transforms import (
    get_train_transforms,
    get_val_transforms
)

from models.network import EOSARUNet

from losses.focal_dice import FocalDiceLoss

from metrics.segmentation_metrics import compute_metrics

from utils.logger import get_logger
from utils.checkpoint import save_checkpoint


def train_model(config, device):


    # Logger
  

    logger = get_logger(
        config["OUTPUT"]["LOG_DIR"]
    )

    logger.info("Starting training...")

  
    # Datasets


    train_dataset = EOSARDataset(
        root_dir=config["DATA"]["ROOT"],
        split="train",
        transforms=get_train_transforms(
            config["DATA"]["IMAGE_SIZE"]
        )
    )

    val_dataset = EOSARDataset(
        root_dir=config["DATA"]["ROOT"],
        split="val",
        transforms=get_val_transforms(
            config["DATA"]["IMAGE_SIZE"]
        )
    )

    # Dataloaders
 
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["TRAIN"]["BATCH_SIZE"],
        shuffle=True,
        num_workers=config["DATA"]["NUM_WORKERS"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["TRAIN"]["BATCH_SIZE"],
        shuffle=False,
        num_workers=config["DATA"]["NUM_WORKERS"]
    )

    logger.info(f"Train samples: {len(train_dataset)}")

    logger.info(f"Val samples: {len(val_dataset)}")

   
    # Model


    model = EOSARUNet()

    model = model.to(device)

  
    # Loss


    criterion = FocalDiceLoss(
        alpha=0.75,
        gamma=2.0,
        dice_weight=0.5
    )

    # Optimizer


    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["TRAIN"]["LR"],
        weight_decay=config["TRAIN"]["WEIGHT_DECAY"]
    )


    # Scheduler


    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["TRAIN"]["EPOCHS"]
    )



    scaler = torch.cuda.amp.GradScaler()


    # Best Validation


    best_f1 = -1.0


    # Training Loop


    for epoch in range(config["TRAIN"]["EPOCHS"]):

        model.train()

        train_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}"
        )

        for batch in progress_bar:

            eo = batch["eo"].to(device)

            sar = batch["sar"].to(device)

            mask = batch["mask"].to(device)

            mask = mask.unsqueeze(1).float()

            optimizer.zero_grad()

   
            # Forward Pass
    
            with torch.cuda.amp.autocast():

                logits = model(eo, sar)

                loss = criterion(
                    logits,
                    mask
                )

       
            # Backpropagation
          

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            train_loss += loss.item()

            progress_bar.set_postfix(
                loss=loss.item()
            )

        scheduler.step()

        avg_train_loss = (
            train_loss / len(train_loader)
        )

        logger.info(
            f"Epoch {epoch+1} "
            f"Train Loss: {avg_train_loss:.4f}"
        )

        

        # Validation
 

        val_metrics = evaluate_model(
            model,
            val_loader,
            device
        )

        print(val_metrics)

        logger.info(
            f"Epoch {epoch+1} "
            f"Val IoU: {val_metrics['IoU']:.4f} "
            f"Val F1: {val_metrics['F1']:.4f}"
        )

        
        


        if val_metrics["F1"] > best_f1:

            best_f1 = val_metrics["F1"]

            checkpoint_path = os.path.join(
                config["CHECKPOINT"]["SAVE_DIR"],
                "best_model.pth"
            )

            save_checkpoint(
                model,
                optimizer,
                epoch,
                checkpoint_path
            )

            logger.info(
                "Best model saved!"
            )

    logger.info("Training completed.")
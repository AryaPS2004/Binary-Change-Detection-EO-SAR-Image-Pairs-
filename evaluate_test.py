import torch

from torch.utils.data import DataLoader

from models.network import EOSARUNet

from data.dataset import EOSARDataset

from engine.evaluator import evaluate_model


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Load Test Dataset


test_dataset = EOSARDataset(
    root_dir="datasets",
    split="test",
    transforms=None
)

test_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False
)


# Load Model


model = EOSARUNet()

checkpoint = torch.load(
    "checkpoints/best_model.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

print("Checkpoint loaded successfully!")


# Evaluate


metrics = evaluate_model(
    model,
    test_loader,
    device
)

print("\n========== TEST RESULTS ==========")

for key, value in metrics.items():

    print(f"{key}: {value:.4f}")
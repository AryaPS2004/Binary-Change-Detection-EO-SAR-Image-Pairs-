import yaml
import torch

from engine.trainer import train_model
from utils.seed import set_seed

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":

    set_seed(42)

    config = load_config("config.yaml")

    device = torch.device(config["TRAIN"]["DEVICE"])

    train_model(config, device)
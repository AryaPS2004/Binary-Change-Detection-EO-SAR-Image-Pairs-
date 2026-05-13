import yaml
import torch

from engine.inferencer import run_inference

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":

    config = load_config("config.yaml")

    device = torch.device(config["TRAIN"]["DEVICE"])

    run_inference(config, device)
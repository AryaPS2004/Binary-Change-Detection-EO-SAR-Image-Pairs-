import yaml
import torch

from engine.evaluator import evaluate_model

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":

    config = load_config("config.yaml")

    device = torch.device(config["TRAIN"]["DEVICE"])

    evaluate_model(config, device)
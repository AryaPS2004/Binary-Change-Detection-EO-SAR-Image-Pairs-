import logging
import os

def get_logger(log_dir):

    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "train.log")

    logger = logging.getLogger("trainer")

    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    fh.setFormatter(formatter)

    logger.addHandler(fh)

    return logger
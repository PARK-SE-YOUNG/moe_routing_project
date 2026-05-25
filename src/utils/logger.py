import logging
import os


def setup_logger(log_dir="logs", log_file="train.log"):

    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("MoE_Project")

    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Log format
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s"
    )

    # File handler
    file_handler = logging.FileHandler(
        os.path.join(log_dir, log_file)
    )

    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
import datetime
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from cnn import BaseCNN
from utils import transform_image_to_tensor


logger = logging.getLogger(__name__)


# choose model by exposed hyperparameters / accuracy on test set
HOT_SWAP_SCALE_FACTOR = 16
HOT_SWAP_EPOCHS = 20
HOT_SWAP_ACC = 99
MODEL_WEIGHTS_PATH = (
    f"weights/mnist_cnn.{HOT_SWAP_SCALE_FACTOR}.{HOT_SWAP_EPOCHS}.{HOT_SWAP_ACC}.pt"
)

MNIST_IMG_DIMENSIONS = (28, 28)
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
DEBUG_MODE = os.getenv("DEBUG_MODE") is not None


def load_model(weights_path: str = MODEL_WEIGHTS_PATH) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaseCNN(factor=HOT_SWAP_SCALE_FACTOR).to(device)
    if Path(weights_path).is_file():
        state = torch.load(weights_path, device)
        model.load_state_dict(state)
    else:
        raise ValueError(f"⚠️ Weight file not found at {MODEL_WEIGHTS_PATH}")
    # ensure model is in evaluation mode before handing off
    model.eval()
    return model


def get_greyscale_image(img: np.ndarray) -> Image.Image:
    """Convert a numpy array representing an RGBA image (from canvas) to a greyscale PIL."""
    # select red channel (all channels should be identical so any will do) and make greyscale
    return Image.fromarray(img[:, :, 0]).convert("L")


def preprocess(img: np.ndarray) -> torch.Tensor:
    """Canvas RGBA → 1-batch 1x28x28 tensor ready for the CNN."""
    return transform_image_to_tensor(
        img=get_greyscale_image(img),
        resize=MNIST_IMG_DIMENSIONS,
    ).unsqueeze(0)  # add singleton batch dimension


def infer(img: np.ndarray) -> tuple[int, float]:
    logger.info("Running inference to get prediction...")
    try:
        model = load_model()
        tensor = preprocess(img)
        # ensure tensor is on same device where model is loaded
        tensor = tensor.to(next(model.parameters()).device)
        with torch.no_grad():
            logits = model(tensor)
            # use softmax to convert logits to a list of probabilities per class (read: digit)
            probs = torch.softmax(logits, dim=1)[0].numpy()
            # get the digit with the highest probability (which we also interpret as 'confidence')
        logger.info(f"Predicted probabilities: {probs}")
        prediction = int(probs.argmax())
        confidence = float(probs[prediction])
        return prediction, confidence
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return -1, 0.0

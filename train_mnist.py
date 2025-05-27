import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2


logger = logging.getLogger(__name__)


# constants
DATA_DIR = "data"
MODEL_PATH = "models/mnist_cnn_{}.pth"
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
RNG_SEED = 42

# hyperparameters (to be fine tuned by trial and error)
VALIDATION_IMAGES = 5000
BATCH_SIZE = 200
LEARNING_RATE = 1e-3
KERNEL_SIZE = 3
EPOCHS = 6


def main() -> None:
    # set up logger (`export LOG_LEVEL_DEBUG=1` in executing shell to see debug logs)
    logging.basicConfig(
        level=logging.DEBUG if os.getenv("LOG_LEVEL_DEBUG") else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # determine if we have GPU, else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Using device: {device}")

    # ---------- 1. Data loading ----------
    # ready transformer to turn PIL images into normalised tensors
    # see https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/5
    transform = v2.Compose(
        (
            # first two transforms replace ToTensor() from transforms v1
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        )
    )

    # MNIST includes 60k training images and 10k test images (all single channel 28x28)
    # we're going to use a small subset of the former for validation purposes
    full_training_ds = datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    training_images_num = len(full_training_ds) - VALIDATION_IMAGES
    training_ds, validation_ds = random_split(
        dataset=full_training_ds,
        lengths=[training_images_num, VALIDATION_IMAGES],
        # fix the generator for deterministic splitting
        generator=torch.Generator().manual_seed(RNG_SEED),
    )
    test_ds = datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=transform
    )

    training_loader = DataLoader(training_ds, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    logger.info(
        f"Loaded {len(training_ds)} training, "
        f"{len(validation_ds)} validation, and {len(test_ds)} test images"
    )

    # ---------- 2. Model definition ----------
    # we build a small convolutional neural network (CNN) appropriate for image classification
    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            # methods for feature extraction
            self.conv1 = nn.Conv2d(1, 32, kernel_size=KERNEL_SIZE, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=KERNEL_SIZE, padding=1)
            self.pool = nn.MaxPool2d(2)
            # methods for classification (fc == fully connected)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            # x is a batch of images with shape (BATCH_SIZE, 1, 28, 28)
            x = self.pool(F.relu(self.conv1(x)))  # 1×28×28 / 32×28×28 / 32x14x14
            x = self.pool(F.relu(self.conv2(x)))  # 32×14×14 / 64×14×14 / 64x7x7
            x = torch.flatten(x, 1)  # flatten to (BATCH_SIZE, 3136)
            x = F.relu(self.fc1(x))  # 64x7x7 = 3136
            return self.fc2(x)  # finally spit out 10 logits, 1 per digit class

    model = SmallCNN().to(device)
    logger.info(f"Model structure:\n{model}")

    # ---------- 3. Define loss function & optimisation ----------
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    logger.info(
        f"Using {optimizer.__class__.__name__} with learning rate {LEARNING_RATE}"
    )

    # ---------- 4. Train and validate ----------
    for epoch in range(EPOCHS):
        logger.debug(f"Running training phase for epoch {epoch + 1}/{EPOCHS}...")
        model.train()
        running_loss, total_loss = 0.0, 0.0
        for i, data in enumerate(training_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients, run forward pass, compute loss, backpropagate, optimise
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_loss += loss.item()
            # emit debug log with running loss every 50 batches
            if i % 50 == 49:
                logger.debug(
                    f"[{epoch + 1}, {(i + 1) * BATCH_SIZE}] loss: {running_loss / 50:.4f}"
                )
                running_loss = 0.0
        training_loss_avg = total_loss / len(training_loader)

        training_log = (
            f"Epoch {epoch + 1}/{EPOCHS} | training loss {training_loss_avg:.4f}"
        )
        if VALIDATION_IMAGES == 0:
            # if we didn't split out a validation dataset, just log training progress
            logger.info(training_log)
            continue

        # else, validate model after each round of training to monitor progress
        logger.debug(f"Running validation phase for epoch {epoch + 1}/{EPOCHS}...")
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        # we disable gradient computation for validation
        with torch.no_grad():
            for i, data in enumerate(validation_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # only do forward pass and compute loss, no backpropagation etc.
                logits = model(inputs)
                loss = criterion(logits, labels)

                running_loss += loss.item()
                predictions = logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        validation_loss_avg = running_loss / len(validation_loader)
        accuracy = correct / total

        logger.info(
            f"{training_log} | val loss {validation_loss_avg:.4f} | val acc {accuracy:.4f}"
        )

    # ---------- 5. Evaluation again test dataset ----------
    logger.debug("Evaluating trained model against test dataset...")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            predictions = model(inputs).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy_pc = 100 * correct / total
    logger.info(f"Test accuracy: {accuracy_pc:.2f}%")

    # and finally we save the model
    torch.save(model.state_dict(), MODEL_PATH.format(int(accuracy_pc)))


if __name__ == "__main__":
    main()

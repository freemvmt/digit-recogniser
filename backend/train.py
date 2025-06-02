import functools
import logging
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from cnn import BaseCNN
from utils import (
    transform_image_to_tensor,
    visualise_data_samples,
)


logger = logging.getLogger(__name__)


# hyperparameters (in practice, we mostly play with number of training epochs and model size)
BATCH_SIZE = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
TRAINING_EPOCHS = 5
MODEL_SCALE_FACTOR = 32

# other constants
DATA_DIR = "data"
RNG_SEED = 42
VALIDATION_IMAGES = 5000
MODEL_WEIGHTS_PATH = "weights/mnist_cnn.{factor}.{epochs}.{acc}.pt"
LOSS_PLOT_PATH = "plots/mnist_cnn_loss.{factor}.{epochs}.{acc}.png"
ACC_PLOT_PATH = "plots/mnist_cnn_acc.{factor}.{epochs}.{acc}.png"
DEBUG_MODE = os.getenv("DEBUG_MODE") is not None


def main() -> None:
    # set up logger (`export DEBUG_MODE=1` in executing shell to see debug logs)
    logging.basicConfig(
        level=logging.DEBUG if DEBUG_MODE else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    # screen out debug logs from matplotlib
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    # determine if we have GPU, else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Using device: {device}")

    # ---------- 1. Data loading ----------
    # MNIST includes 60k training images and 10k test images (all single channel 28x28)
    # we're going to use half of the latter for validation purposes during training
    training_ds = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=functools.partial(transform_image_to_tensor, augment=True),
    )
    full_test_ds = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform_image_to_tensor,
    )
    test_images_num = len(full_test_ds) - VALIDATION_IMAGES
    test_ds, validation_ds = random_split(
        dataset=full_test_ds,
        lengths=[test_images_num, VALIDATION_IMAGES],
        # fix the generator for deterministic splitting
        generator=torch.Generator().manual_seed(RNG_SEED),
    )

    training_loader = DataLoader(training_ds, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    logger.info(
        f"Loaded {len(training_ds)} training, "
        f"{len(validation_ds)} validation, and {len(test_ds)} test images"
    )

    # if in dev/debug mode, visualise data samples for visual sanity check
    if DEBUG_MODE:
        logger.debug("Visualising data samples...")
        visualise_data_samples(training_ds, validation_ds, test_ds)

    # ---------- 2. Define model ----------
    model = BaseCNN(factor=MODEL_SCALE_FACTOR).to(device)
    logger.info(
        f"Initialised model with scale factor {MODEL_SCALE_FACTOR}:\n{summary(model, (1, 28, 28))}"
    )

    # ---------- 3. Define loss function & optimisation ----------
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()
    logger.info(
        f"Using {optimizer.__class__.__name__} with learning rate {LEARNING_RATE} and L2 weight decay {WEIGHT_DECAY}"
    )

    # lists to store values for final plots
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_accuracies = []

    # ---------- 4. Train and validate ----------
    logger.info(
        f"Starting training for {TRAINING_EPOCHS} epochs with batch size {BATCH_SIZE}..."
    )
    for epoch in range(TRAINING_EPOCHS):
        logger.debug(
            f"Running training phase for epoch {epoch + 1}/{TRAINING_EPOCHS}..."
        )
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
        epoch_train_losses.append(training_loss_avg)

        training_log = f"Epoch {epoch + 1}/{TRAINING_EPOCHS} | training loss {training_loss_avg:.4f}"
        if VALIDATION_IMAGES == 0:
            # if we didn't split out a validation dataset, just log training progress
            logger.info(training_log)
            continue

        # else, validate model after each round of training to monitor progress
        logger.debug(
            f"Running validation phase for epoch {epoch + 1}/{TRAINING_EPOCHS}..."
        )
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
        epoch_val_losses.append(validation_loss_avg)
        epoch_val_accuracies.append(accuracy)

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
    torch.save(
        model.state_dict(),
        MODEL_WEIGHTS_PATH.format(
            factor=MODEL_SCALE_FACTOR, epochs=TRAINING_EPOCHS, acc=int(accuracy_pc)
        ),
    )

    # ---------- 6. Plot loss and accuracy for review ----------
    plt.figure(figsize=(10, 5))
    plt.plot(
        range(1, TRAINING_EPOCHS + 1),
        epoch_train_losses,
        label="Training loss",
        color="red",
    )
    plt.plot(
        range(1, TRAINING_EPOCHS + 1),
        epoch_val_losses,
        label="Validation loss",
        color="orange",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and validation loss over epochs")
    plt.legend()
    plt.grid(True)
    plot_path = LOSS_PLOT_PATH.format(
        factor=MODEL_SCALE_FACTOR, epochs=TRAINING_EPOCHS, acc=int(accuracy_pc)
    )
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved loss plot to {plot_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(
        range(1, TRAINING_EPOCHS + 1),
        epoch_val_accuracies,
        label="Validation accuracy",
        color="green",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Validation accuracy over epochs")
    plt.legend()
    plt.grid(True)
    acc_plot_path = ACC_PLOT_PATH.format(
        factor=MODEL_SCALE_FACTOR, epochs=TRAINING_EPOCHS, acc=int(accuracy_pc)
    )
    plt.savefig(acc_plot_path)
    plt.close()
    logger.info(f"Saved accuracy plot to {acc_plot_path}")


if __name__ == "__main__":
    main()

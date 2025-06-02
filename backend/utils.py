from typing import Optional
import logging

import matplotlib.pyplot as plt
from PIL import Image as PILImage
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import v2


logger = logging.getLogger(__name__)


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
VIS_PLOT_PATH = "plots/data_samples_visualisation.png"


def transform_image_to_tensor(
    img: PILImage,
    augment: bool = False,
    resize: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    """Convert a PIL image (from MNIST dataset or frontend canvas) to a normalised tensor."""
    transforms = []
    # while still dealing with a PIL, we may want to introduce some random augmentation
    if augment:
        transforms.append(v2.RandomAffine(degrees=10, translate=(0.1, 0.1)))
    transforms.extend(
        [
            # convert PIL to tensor, ensure values are in uint8
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
        ]
    )
    if resize:
        transforms.append(
            v2.Resize(size=resize, antialias=True),
        )
    transforms.extend(
        [
            # then convert values to float32 and normalise with MNIST mean and std
            v2.ToDtype(torch.float32, scale=True),
            # see https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/5
            v2.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )
    transform = v2.Compose(transforms)
    return transform(img)


def get_image_from_tensor(
    tensor: torch.Tensor,
    denormalise: bool = True,
) -> PILImage:
    """Convert a tensor back to a PIL image, optionally denormalising it."""
    t = tensor.squeeze(0).clone()
    # use tensors based on MNIST std/mean to do inverse of final normalisation step from transform composition
    if denormalise:
        # ensure the mean/std tensors are on the same device as the input tensor
        mnist_mean_tensor = torch.tensor(MNIST_MEAN).to(t.device)
        mnist_std_tensor = torch.tensor(MNIST_STD).to(t.device)
        t = t * mnist_std_tensor + mnist_mean_tensor
    # clip values to [0, 1] range just in case, ensure tensor is in CPU mem, then convert to PIL
    t = torch.clamp(t, 0, 1)
    return v2.ToPILImage()(t.cpu())


def visualise_data_samples(
    training_ds: MNIST,
    val_ds: Optional[MNIST] = None,
    test_ds: Optional[MNIST] = None,
    samples: int = 5,
    plot_path: str = VIS_PLOT_PATH,
) -> None:
    """Developer tooling to visualise transformed tensors from datasets."""
    datasets_to_visualise = {"Training": training_ds}
    if val_ds is not None:
        datasets_to_visualise["Validation"] = val_ds
    if test_ds is not None:
        datasets_to_visualise["Test"] = test_ds

    num_datasets = len(datasets_to_visualise)
    plt.figure(figsize=((samples * 2) + 2, (num_datasets * 3) + 2))
    plot_idx = 1
    for ds_name, ds in datasets_to_visualise.items():
        if len(ds) == 0:
            logger.warning(f"{ds_name} dataset is empty, skipping visualisation.")
            continue
        indices = torch.randperm(len(ds))[:samples]
        for i, sample_idx in enumerate(indices):
            transformed_tensor, label = ds[sample_idx]
            img = get_image_from_tensor(transformed_tensor, denormalise=True)
            plt.subplot(num_datasets, samples, plot_idx)
            plt.imshow(img, cmap="gray")
            plt.title(f"{ds_name}: {label}")
            plt.axis("off")
            plot_idx += 1

    plt.suptitle("Data samples from MNIST datasets", fontsize=16)
    plt.tight_layout(h_pad=2, rect=[0, 0, 1, 0.96])
    plt.savefig(plot_path)
    plt.close()
    logger.debug("Saved visualisation of data samples to /plots")

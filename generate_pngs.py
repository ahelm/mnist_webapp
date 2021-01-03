# -*- coding: utf-8 -*-
import os
from math import ceil
from math import log10
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image
from torchvision import datasets
from tqdm import tqdm


def store_png(dataset: datasets.MNIST, folder_name: str) -> None:
    """Takes dataset and creates PNG from dataset with label in name."""
    data_and_label = tqdm(zip(dataset.data, dataset.targets), total=len(dataset.data))

    for i, (data, label) in enumerate(data_and_label):
        output_dir = output_base_dir / folder_name
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(data.numpy(), "L")
        image.save(
            output_dir / f"{i:0{ceil(log10(dataset.data.shape[0]))}d}_{label:1d}.png"
        )


if __name__ == "__main__":
    load_dotenv()

    DATA_DIR = os.getenv("DATA_DIR", default=".")
    output_base_dir = Path(DATA_DIR).resolve() / "MNIST" / "png"

    train_dataset = datasets.MNIST(
        DATA_DIR,
        train=True,
        download=True,
    )
    test_dataset = datasets.MNIST(
        DATA_DIR,
        train=False,
        download=True,
    )

    store_png(train_dataset, "train")
    store_png(test_dataset, "test")

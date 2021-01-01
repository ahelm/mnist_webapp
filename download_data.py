# -*- coding: utf-8 -*-
from gzip import GzipFile
from pathlib import Path

import numpy as np
import png
import requests

DATA_DIR = (Path(__file__).parent / "data").resolve()

MNIST_BASE_URL = "http://yann.lecun.com/exdb/mnist/"
MNIST_TRAIN_IMAGE = {
    "url": MNIST_BASE_URL + "train-images-idx3-ubyte.gz",
    "magic number": 0x00000803,
    "number of items": 60_000,
    "type": "image",
}
MNIST_TRAIN_LABEL = {
    "url": MNIST_BASE_URL + "train-labels-idx1-ubyte.gz",
    "magic number": 0x00000801,
    "number of items": 60_000,
    "type": "label",
}
MNIST_TEST_IMAGE = {
    "url": MNIST_BASE_URL + "t10k-images-idx3-ubyte.gz",
    "magic number": 0x00000803,
    "number of items": 10_000,
    "type": "image",
}
MNIST_TEST_LABEL = {
    "url": MNIST_BASE_URL + "t10k-labels-idx1-ubyte.gz",
    "magic number": 0x00000801,
    "number of items": 10_000,
    "type": "label",
}


def warning(message: str) -> None:
    """Prints warning with `message` being the message of the warning."""
    print(f"~ {message}")


def status(message: str, add_star: bool = True) -> None:
    """Prints status message where `message` is the message string."""
    print(f"{'*' + message if add_star else message}", end="", flush=True)


def done() -> None:
    """Prints `DONE` -> introduces a new line"""
    print("DONE", flush=True)


def download_archive(url: str) -> Path:
    """Downloads MNIST raw data"""
    fname = url.split("/")[-1]
    archives_dir = DATA_DIR / "archives"
    archives_dir.mkdir(parents=True, exist_ok=True)
    archive_location = archives_dir / fname

    if archive_location.exists():
        warning(f"File {archive_location} exists ... skipping download")
        return archive_location

    status(f"Downloading {archive_location} ... ")
    response = requests.get(url, allow_redirects=True)
    assert response.status_code == 200
    archive_location.write_bytes(response.content)
    done()

    return archive_location


def extract_gz(archive: Path, to: Path) -> None:
    """Extracts gunzip file."""
    status(f"Extracting '{archive}' ... ")

    with GzipFile(archive, "rb") as fp:
        to.write_bytes(fp.read())

    done()
    return None


def get_magic_number(path: Path) -> bytes:
    """Get the magic number of a file object (e.g. image or label)"""
    return path.read_bytes()[:4]


def get_number_of_items(path: Path) -> bytes:
    """Get the number of items stored in the file object."""
    return path.read_bytes()[4:8]


def get_image_data(path: Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes()[16:], dtype=np.uint8).reshape((-1, 28, 28))


def get_labels(path: Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes()[8:], dtype=np.uint8)


def main():
    """Main function which downloads and converts the raw files to PNG."""
    TRAIN_DATASET = (MNIST_TRAIN_IMAGE, MNIST_TRAIN_LABEL, "train")
    TEST_DATASET = (MNIST_TEST_IMAGE, MNIST_TEST_LABEL, "test")

    for img, label, sub_dir in [TRAIN_DATASET, TEST_DATASET]:
        for obj in (img, label):
            archive_path = download_archive(obj["url"])
            data_file = DATA_DIR / archive_path.stem
            extract_gz(archive_path, data_file)
            assert get_magic_number(data_file) != obj["magic number"]
            assert get_number_of_items(data_file) != obj["number of items"]
            obj["raw path"] = data_file

        status("Unpacking data -> ")
        raw_imgs = get_image_data(img["raw path"])

        status("Unpacking labels -> ")
        raw_labels = get_labels(label["raw path"])

        status("Writting image data ... ")
        for i, (arr, l) in enumerate(zip(raw_imgs, raw_labels)):
            output_dir = DATA_DIR / sub_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            png_file = output_dir / f"{i:05}.png"
            if png_file.exists():
                continue
            png.from_array(arr, "L").save(png_file)
        done()

        img["raw path"].unlink()
        label["raw path"].unlink()


if __name__ == "__main__":
    main()

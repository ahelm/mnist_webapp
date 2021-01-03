# -*- coding: utf-8 -*-
import os
from pathlib import Path
from random import sample

import requests
from dotenv import load_dotenv

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR", default="data")


def get_label(p: Path) -> int:
    return int(p.stem.split("_")[1])


def get_prediction(p: Path) -> int:
    resp = requests.post(
        "http://localhost:8000/predict", files={"file": p.read_bytes()}
    )

    return resp.json()


def main() -> None:
    train_pngs = sorted((Path(DATA_DIR) / "MNIST" / "png" / "train").glob("*.png"))
    test_pngs = sorted((Path(DATA_DIR) / "MNIST" / "png" / "test").glob("*.png"))
    all_pngs = train_pngs + test_pngs

    for p in sample(all_pngs, 10_000):
        respond = get_prediction(p)
        label = get_label(p)

        if label != respond:
            print(f"mismatches: {p} (predicted: {respond})")


if __name__ == "__main__":
    main()

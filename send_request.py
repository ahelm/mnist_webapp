# -*- coding: utf-8 -*-
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR", default="data")


test_pngs = sorted((Path(DATA_DIR) / "MNIST" / "png" / "test").glob("*.png"))
test_labels = [s.stem.split("_")[1] for s in test_pngs]

print(f"> using     -> {test_labels[8]}")

resp = requests.post(
    "http://localhost:8000/predict", files={"file": open(test_pngs[8], "rb")}
)
print(f"> predicted -> {resp.json()}")

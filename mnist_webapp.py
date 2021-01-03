# -*- coding: utf-8 -*-
import io
import os
from pathlib import Path

import torch
import torchvision.transforms as transforms
from dotenv import load_dotenv
from flask import Flask
from flask import jsonify
from flask import request
from PIL import Image

from mnist_model import Net

load_dotenv()


app = Flask(__name__)


DATA_DIR = os.getenv("DATA_DIR", default=".")
NORM_MEAN = float(os.getenv("NORM_MEAN", default="0.5"))
NORM_STD = float(os.getenv("NORM_STD", default="0.001"))


model_path = Path(DATA_DIR + "/mnist_model.pt").absolute()
model = Net()
model.load_state_dict(torch.load(model_path))
model.eval()


def transform_image(image_bytes: bytes) -> torch.Tensor:
    transformer = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((NORM_MEAN,), (NORM_STD,)),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes))
    return transformer(image).unsqueeze(0)


def get_prediction(image_bytes: bytes) -> int:
    img_tensor = transform_image(image_bytes)
    outputs = model.forward(img_tensor)
    _, y_hat = outputs.max(1)
    return int(y_hat)


@app.route("/")
def index():
    return "Nothing is here -> check /prediction"


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        file = request.files["file"]
        img_bytes = file.read()
    number = get_prediction(img_bytes)
    return jsonify(number)

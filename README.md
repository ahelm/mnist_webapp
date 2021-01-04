# ML Service for predicting numbers (based on MNIST)

This repo showcases a web service that predicts the hand-written number. It is
built using a simple convolutional neural network (CNN) and is trained on the
MNIST dataset.

## Features

- [ ] Store training data and features in a data warehouse
- [X] Utilizing PyTorch for training and prediction
- [X] Handles multiple requests using WSGI HTTP Server `gunicorn`
- [X] Predict hand-written numbers using PNGs

## Steps to run

Most of the setup for the app is stored inside a docker container. To use the app, run:

```shell
make
```

The `make` command will build the container, train the model, and startup a
webserver. After the server is running, you can send any PNG with a
hand-written number to `http://localhost:8000/predict`. This will reply with
the number predicted. However, to simplify the testing, the PNGs in the MNIST dataset can be generated using

```shell
make pngs
```

This will create 70 000 PNGs (60 000 `train` and 10 000 `test`) in the
directory `data/MNIST/png`. And using the command

```shell
python send_requests.py
```

will randomly use 10 000 PNGs to send it to `http://localhost:8000/predict` and
notify if the numbers of the prediction and the label mismatch. **NOTE: It
requires `requests` to be installed. Consider using `requirements.txt`.**

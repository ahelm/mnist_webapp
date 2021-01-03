# -*- coding: utf-8 -*-
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", default=".")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", default="64"))
TEST_BATCH_SIZE = int(os.getenv("TEST_BATCH_SIZE", default="1024"))
NORM_MEAN = float(os.getenv("NORM_MEAN", default="0.5"))
NORM_STD = float(os.getenv("NORM_STD", default="0.001"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", default="1.0"))
LEARNING_STEP_GAMMA = float(os.getenv("LEARNING_STEP_GAMMA", default="0.7"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", default="1"))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
) -> None:
    model.train()
    with tqdm(total=len(train_loader.dataset), desc="Train") as pbar:
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            pbar.set_postfix_str(f"loss: {loss.item():10.4e}")
            pbar.update(len(data))


def test(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
) -> None:
    model.eval()
    test_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        f" Test: avg.loss {test_loss:8.2e}, "
        f"acc {int(correct):5d}/{len(test_loader.dataset):5d} "
        f"({float(100.0 * correct / len(test_loader.dataset)):5.2f} %)",
        flush=True,
    )


def main():
    dataset_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((NORM_MEAN,), (NORM_STD,)),
        ]
    )

    train_dataset = datasets.MNIST(
        DATA_DIR,
        train=True,
        download=True,
        transform=dataset_transform,
    )
    test_dataset = datasets.MNIST(
        DATA_DIR,
        train=False,
        download=True,
        transform=dataset_transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)

    device = torch.device("cpu")
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=1, gamma=LEARNING_STEP_GAMMA)

    for epoch in range(1, NUM_EPOCHS + 1):
        term_width = os.get_terminal_size().columns
        print(f" EPOCH {epoch} ".center(term_width, "="))
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)
        scheduler.step()

    term_width = os.get_terminal_size().columns
    print(" SAVING MODEL ".center(term_width, "="))
    torch.save(model.state_dict(), Path(f"{DATA_DIR}/mnist_model.pt"))


if __name__ == "__main__":
    main()

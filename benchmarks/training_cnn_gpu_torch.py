"""PyTorch counterpart to ``training_cnn_gpu.mojo``.

Runs the same MNIST-shaped 16/32-channel CNN, batch size, warm-up, timed
steps, Adam settings, and synthetic-device-data protocol as Mantle.
"""
from __future__ import annotations

import time

import torch
from torch import nn


BATCH_SIZE = 64
WARMUP_STEPS = 3
TIMED_STEPS = 20
TRIALS = 5


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize(target: torch.device) -> None:
    if target.type == "mps":
        torch.mps.synchronize()
    elif target.type == "cuda":
        torch.cuda.synchronize()


class Cnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def train_step(
    model: Cnn,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=False)
    loss = loss_fn(model(inputs), targets)
    loss.backward()
    optimizer.step()


def main() -> None:
    target = device()
    inputs = torch.rand(BATCH_SIZE, 1, 28, 28, device=target)
    targets = torch.rand(BATCH_SIZE, 10, device=target)
    loss_fn = nn.MSELoss()

    for trial in range(1, TRIALS + 1):
        model = Cnn().to(target)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(WARMUP_STEPS):
            train_step(model, optimizer, loss_fn, inputs, targets)
        synchronize(target)

        started = time.perf_counter()
        for _ in range(TIMED_STEPS):
            train_step(model, optimizer, loss_fn, inputs, targets)
        synchronize(target)
        print(f"RESULT, cnn-mnist , {trial} , {time.perf_counter() - started:.6f}")


if __name__ == "__main__":
    main()

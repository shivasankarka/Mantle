"""PyTorch MPS reference for training_workloads_gpu.mojo."""

import time

import torch
from torch import nn

WARMUP_STEPS = 5
TIMED_STEPS = 100
TRIALS = 5


# ===----------------------------------------------------------------------===#
# Housing: matches examples/housing.py's bare nn.Linear(13, 1) (14 params).
# ===----------------------------------------------------------------------===#


def run_housing_case(device: torch.device) -> None:
    torch.manual_seed(0)
    batch_size = 64
    loss_fn = nn.MSELoss()

    inputs = torch.rand(batch_size, 13, device=device)
    targets = torch.rand(batch_size, 1, device=device)

    for trial in range(TRIALS):
        model = nn.Linear(13, 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(WARMUP_STEPS):
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(inputs), targets).backward()
            optimizer.step()
        torch.mps.synchronize()

        start = time.perf_counter()
        for _ in range(TIMED_STEPS):
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(inputs), targets).backward()
            optimizer.step()
        torch.mps.synchronize()
        print(f"RESULT,housing,{trial + 1},{time.perf_counter() - start:.9f}")


# ===----------------------------------------------------------------------===#
# Sine: examples/sin_estimate.py's 2-hidden-layer MLP, reused at three
# widths to hit ~1k / ~20k / ~350k total parameters.
#
# sin-large uses 592, not the CPU suite's 590, to match training_workloads_gpu.mojo
# (avoids a MAX GPU matmul performance cliff on non-16-aligned K dimensions).
# ===----------------------------------------------------------------------===#


class SinMLP(nn.Module):
    def __init__(self, n_hidden: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def run_sin_case(label: str, n_hidden: int, device: torch.device) -> None:
    torch.manual_seed(0)
    batch_size = 1024
    inputs = torch.rand(batch_size, 1, device=device)
    targets = torch.rand(batch_size, 1, device=device)
    loss_fn = nn.MSELoss()

    for trial in range(TRIALS):
        model = SinMLP(n_hidden).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(WARMUP_STEPS):
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(inputs), targets).backward()
            optimizer.step()
        torch.mps.synchronize()

        start = time.perf_counter()
        for _ in range(TIMED_STEPS):
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(inputs), targets).backward()
            optimizer.step()
        torch.mps.synchronize()
        print(f"RESULT,{label},{trial + 1},{time.perf_counter() - start:.9f}")


# ===----------------------------------------------------------------------===#
# MNIST: matches examples/mnist.py's CNN (conv 16 -> conv 32 -> linear 10,
# ~29k params).
# ===----------------------------------------------------------------------===#


class MnistCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return self.out(x)


def run_mnist_case(device: torch.device) -> None:
    torch.manual_seed(0)
    batch_size = 64
    inputs = torch.rand(batch_size, 1, 28, 28, device=device)
    targets = torch.zeros(batch_size, dtype=torch.int64, device=device)
    loss_fn = nn.CrossEntropyLoss()

    for trial in range(TRIALS):
        model = MnistCNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(WARMUP_STEPS):
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(inputs), targets).backward()
            optimizer.step()
        torch.mps.synchronize()

        start = time.perf_counter()
        for _ in range(TIMED_STEPS):
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(inputs), targets).backward()
            optimizer.step()
        torch.mps.synchronize()
        print(f"RESULT,mnist,{trial + 1},{time.perf_counter() - start:.9f}")


if __name__ == "__main__":
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable; this benchmark requires Apple Metal.")
    device = torch.device("mps")
    run_housing_case(device)
    run_sin_case("sin-small", 30, device)
    run_sin_case("sin-medium", 139, device)
    run_sin_case("sin-large", 592, device)
    run_mnist_case(device)

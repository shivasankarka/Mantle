"""PyTorch MPS reference for ``training_workloads_gpu.mojo``."""

import time

import torch
from torch import nn


WARMUP_STEPS = 5
TIMED_STEPS = 100
TRIALS = 5
CASES = (
    ("housing-small", 64, 13, 32, 1),
    ("housing-medium", 64, 13, 128, 1),
    ("housing-large", 64, 13, 512, 1),
    ("sin-small", 1024, 1, 32, 1),
    ("sin-medium", 1024, 1, 128, 1),
    ("sin-large", 1024, 1, 512, 1),
    ("mnist-small", 64, 784, 64, 10),
    ("mnist-medium", 64, 784, 256, 10),
    ("mnist-large", 64, 784, 512, 10),
)


class MLP(nn.Module):
    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_outputs),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def step(
    model: MLP,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss_fn(model(inputs), targets).backward()
    optimizer.step()


def run_case(label: str, batch_size: int, n_inputs: int, n_hidden: int, n_outputs: int) -> None:
    torch.manual_seed(0)
    device = torch.device("mps")
    inputs = torch.rand(batch_size, n_inputs, device=device)
    targets = torch.rand(batch_size, n_outputs, device=device)
    loss_fn = nn.MSELoss()

    for trial in range(TRIALS):
        model = MLP(n_inputs, n_hidden, n_outputs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(WARMUP_STEPS):
            step(model, optimizer, loss_fn, inputs, targets)
        torch.mps.synchronize()

        start = time.perf_counter()
        for _ in range(TIMED_STEPS):
            step(model, optimizer, loss_fn, inputs, targets)
        torch.mps.synchronize()
        print(f"RESULT,{label},{trial + 1},{time.perf_counter() - start:.9f}")


if __name__ == "__main__":
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable; this benchmark requires Apple Metal.")
    for case in CASES:
        run_case(*case)

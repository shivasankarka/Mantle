"""PyTorch equivalent of gpu_vs_cpu_benchmark.mojo: same architecture,
sizes, and epoch count, timed on CPU and GPU (MPS) for comparison."""
import time

import torch
import torch.nn as nn

BATCH_SIZE = 256
N_INPUTS = 1024
N_HIDDEN = 2048
N_OUTPUTS = 10
LEARNING_RATE = 0.001
EPOCHS = 20


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_INPUTS, N_HIDDEN)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(N_HIDDEN, N_HIDDEN)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(N_HIDDEN, N_OUTPUTS)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


def run(device: str) -> float:
    torch.manual_seed(0)
    model = MLP().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    x = torch.rand(BATCH_SIZE, N_INPUTS, device=device)
    y = torch.rand(BATCH_SIZE, N_OUTPUTS, device=device)

    print(f"{device}: warming up")
    optim.zero_grad()
    loss_fn(model(x), y).backward()
    optim.step()
    if device == "mps":
        torch.mps.synchronize()

    print(f"{device}: training {EPOCHS} epochs")
    start = time.perf_counter()
    for _ in range(EPOCHS):
        optim.zero_grad()
        loss_fn(model(x), y).backward()
        optim.step()
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    print(f"{device}: {elapsed:.6f} seconds ({elapsed / EPOCHS:.6f} s/epoch)")
    return elapsed


if __name__ == "__main__":
    cpu_seconds = run("cpu")
    gpu_seconds = run("mps") if torch.backends.mps.is_available() else None

    if gpu_seconds:
        print(f"Speedup (CPU/GPU): {cpu_seconds / gpu_seconds}")
    else:
        print("MPS not available, skipping GPU run")

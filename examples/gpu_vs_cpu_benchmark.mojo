"""Trains the same MLP on CPU and GPU with large enough layers that GPU's
matmul throughput should win despite the still-host-fallback POW/MEAN/
broadcast-ADD ops, and prints both wall-clock times."""
from std.random import rand
from std.time import perf_counter_ns as now

import mantle.nn as nn
from mantle import Tensor, TensorShape
from mantle import f32
from mantle import Graph, Symbol, OP
from mantle.core.device import Device


def create_mlp(
    batch_size: Int, n_inputs: Int, n_hidden: Int, n_outputs: Int
) -> Graph:
    var g = Graph()

    var x = g.input(TensorShape(batch_size, n_inputs))
    var y_true = g.input(TensorShape(batch_size, n_outputs))

    var x1 = nn.Linear(g, x, n_outputs=n_hidden)
    var x2 = nn.ReLU(g, x1)
    var x3 = nn.Linear(g, x2, n_outputs=n_hidden)
    var x4 = nn.ReLU(g, x3)
    var y_pred = nn.Linear(g, x4, n_outputs=n_outputs)
    g.out(y_pred)

    var loss = nn.MSELoss(g, y_pred, y_true)
    g.loss(loss)

    return g^


def main() raises:
    comptime batch_size = 256
    comptime n_inputs = 1024
    comptime n_hidden = 2048
    comptime n_outputs = 10
    comptime learning_rate = 0.001
    comptime epochs = 20

    comptime graph = create_mlp(batch_size, n_inputs, n_hidden, n_outputs)

    var x_data = Tensor[f32](batch_size, n_inputs)
    var y_data = Tensor[f32](batch_size, n_outputs)
    rand[f32](x_data.ptr(), x_data.num_elements())
    rand[f32](y_data.ptr(), y_data.num_elements())

    # --- CPU ---
    var model_cpu = nn.Model[graph]()
    var optim_cpu = nn.optim.Adam[graph](model_cpu.parameters, lr=learning_rate)

    print("CPU: warming up")
    _ = model_cpu.forward(x_data.copy(), y_data.copy())
    optim_cpu.zero_grad()
    model_cpu.backward()
    optim_cpu.step()

    print("CPU: training", epochs, "epochs")
    var start_cpu = now()
    for _ in range(epochs):
        _ = model_cpu.forward(x_data.copy(), y_data.copy())
        optim_cpu.zero_grad()
        model_cpu.backward()
        optim_cpu.step()
    var cpu_seconds = Float64(now() - start_cpu) / 1e9
    print("CPU:", cpu_seconds, "seconds (", cpu_seconds / epochs, "s/epoch)")

    # --- GPU ---
    var model_gpu = nn.Model[graph, device = Device.gpu]()
    var optim_gpu = nn.optim.Adam[graph, device = Device.gpu](
        model_gpu.parameters, lr=learning_rate
    )

    print("GPU: warming up")
    _ = model_gpu.forward(x_data.to_gpu(), y_data.to_gpu())
    optim_gpu.zero_grad()
    model_gpu.backward()
    optim_gpu.step()

    print("GPU: training", epochs, "epochs")
    var start_gpu = now()
    for _ in range(epochs):
        _ = model_gpu.forward(x_data.to_gpu(), y_data.to_gpu())
        optim_gpu.zero_grad()
        model_gpu.backward()
        optim_gpu.step()
    var gpu_seconds = Float64(now() - start_gpu) / 1e9
    print("GPU:", gpu_seconds, "seconds (", gpu_seconds / epochs, "s/epoch)")

    print("Speedup (CPU/GPU):", cpu_seconds / gpu_seconds)

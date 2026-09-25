"""Phase profile for the GPU MNIST-medium training workload.

Synchronization after every phase is intentional: this is a diagnostic tool,
not a throughput benchmark. It attributes queued GPU work to gradient clearing,
forward, backward, or Adam respectively.
"""
from std.random import rand
from std.time import perf_counter_ns as now

import mantle.nn as nn
from mantle import Graph, Tensor, TensorShape, f32
from mantle.core.device import Device


comptime WARMUP_STEPS = 5
comptime PROFILE_STEPS = 100


def create_mlp() -> Graph:
    var graph = Graph()
    var inputs = graph.input(TensorShape(64, 784))
    var targets = graph.input(TensorShape(64, 10))
    var hidden1 = nn.Linear(graph, inputs, n_outputs=256)
    var hidden2 = nn.ReLU(graph, hidden1)
    var hidden3 = nn.Linear(graph, hidden2, n_outputs=256)
    var hidden4 = nn.ReLU(graph, hidden3)
    var predictions = nn.Linear(graph, hidden4, n_outputs=10)
    graph.out(predictions)
    graph.loss(nn.MSELoss(graph, predictions, targets))
    return graph^


def main() raises:
    comptime graph = create_mlp()
    var host_inputs = Tensor[f32](64, 784)
    var host_targets = Tensor[f32](64, 10)
    rand[f32](host_inputs.ptr(), host_inputs.num_elements())
    rand[f32](host_targets.ptr(), host_targets.num_elements())
    var inputs = host_inputs.to_gpu()
    var targets = host_targets.to_gpu()
    var model = nn.Model[graph, device=Device.gpu]()
    var optimizer = nn.optim.Adam[graph, device=Device.gpu](
        model.parameters, lr=0.001
    )

    for _ in range(WARMUP_STEPS):
        optimizer.zero_grad()
        _ = model.forward(inputs, targets)
        model.backward()
        optimizer.step()
    inputs.gpu_context().synchronize()

    var zero_grad_ns = 0
    var forward_ns = 0
    var backward_ns = 0
    var adam_ns = 0
    for _ in range(PROFILE_STEPS):
        var start = now()
        optimizer.zero_grad()
        inputs.gpu_context().synchronize()
        zero_grad_ns += now() - start

        start = now()
        _ = model.forward(inputs, targets)
        inputs.gpu_context().synchronize()
        forward_ns += now() - start

        start = now()
        model.backward()
        inputs.gpu_context().synchronize()
        backward_ns += now() - start

        start = now()
        optimizer.step()
        inputs.gpu_context().synchronize()
        adam_ns += now() - start

    print("PROFILE, zero_grad,", Float64(zero_grad_ns) / 1e9)
    print("PROFILE, forward,", Float64(forward_ns) / 1e9)
    print("PROFILE, backward,", Float64(backward_ns) / 1e9)
    print("PROFILE, adam,", Float64(adam_ns) / 1e9)

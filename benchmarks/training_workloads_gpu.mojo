"""GPU training sweep matched to ``training_workloads_gpu_torch.py``.

Each case times 100 MLP optimization steps after five warm-up steps.  Inputs
and targets are uploaded once before warm-up; synchronization happens only
after each group of steps, so the measurement excludes host/device transfers
but includes all queued model work.
"""
from std.random import rand
from std.time import perf_counter_ns as now

import mantle.nn as nn
from mantle import Graph, Tensor, TensorShape, f32
from mantle.core.device import Device


comptime WARMUP_STEPS = 5
comptime TIMED_STEPS = 100
comptime TRIALS = 5


def create_mlp(
    batch_size: Int, n_inputs: Int, n_hidden: Int, n_outputs: Int
) -> Graph:
    var graph = Graph()
    var inputs = graph.input(TensorShape(batch_size, n_inputs))
    var targets = graph.input(TensorShape(batch_size, n_outputs))

    var hidden1 = nn.Linear(graph, inputs, n_outputs=n_hidden)
    var activated1 = nn.ReLU(graph, hidden1)
    var hidden2 = nn.Linear(graph, activated1, n_outputs=n_hidden)
    var activated2 = nn.ReLU(graph, hidden2)
    var predictions = nn.Linear(graph, activated2, n_outputs=n_outputs)
    graph.out(predictions)
    graph.loss(nn.MSELoss(graph, predictions, targets))
    return graph^


def run_case[
    batch_size: Int,
    n_inputs: Int,
    n_hidden: Int,
    n_outputs: Int,
](label: String) raises:
    comptime graph = create_mlp(batch_size, n_inputs, n_hidden, n_outputs)

    var host_inputs = Tensor[f32](batch_size, n_inputs)
    var host_targets = Tensor[f32](batch_size, n_outputs)
    rand[f32](host_inputs.ptr(), host_inputs.num_elements())
    rand[f32](host_targets.ptr(), host_targets.num_elements())
    var inputs = host_inputs.to_gpu()
    var targets = host_targets.to_gpu()

    for trial in range(TRIALS):
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

        var start = now()
        for _ in range(TIMED_STEPS):
            optimizer.zero_grad()
            _ = model.forward(inputs, targets)
            model.backward()
            optimizer.step()
        inputs.gpu_context().synchronize()
        var seconds = Float64(now() - start) / 1e9
        print("RESULT,", label, ",", trial + 1, ",", seconds)


def main() raises:
    run_case[64, 13, 32, 1]("housing-small")
    run_case[64, 13, 128, 1]("housing-medium")
    run_case[64, 13, 512, 1]("housing-large")

    run_case[1024, 1, 32, 1]("sin-small")
    run_case[1024, 1, 128, 1]("sin-medium")
    run_case[1024, 1, 512, 1]("sin-large")

    run_case[64, 784, 64, 10]("mnist-small")
    run_case[64, 784, 256, 10]("mnist-medium")
    run_case[64, 784, 512, 10]("mnist-large")

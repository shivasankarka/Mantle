"""GPU training sweep matched to ``training_workloads_gpu_torch.py``.

Mirrors ``training_workloads.mojo``'s case sizes (housing/mnist fixed at
their example's own parameter counts, sin_estimate's architecture reused at
three widths) so the CPU and GPU charts are directly comparable. Inputs and
targets are uploaded once before warm-up; synchronization happens only after
each group of steps, so the measurement excludes host/device transfers but
includes all queued model work.
"""
from std.random import rand
from std.time import perf_counter_ns as now

import mantle.nn as nn
from mantle import Graph, Tensor, TensorShape, f32
from mantle.core.device import Device


comptime WARMUP_STEPS = 5
comptime TIMED_STEPS = 100
comptime TRIALS = 5


# ===----------------------------------------------------------------------===#
# Housing: matches examples/housing.mojo's bare Linear(13, 1) (14 params).
# ===----------------------------------------------------------------------===#


def create_housing_graph(batch_size: Int) -> Graph:
    var graph = Graph()
    var inputs = graph.input(TensorShape(batch_size, 13))
    var targets = graph.input(TensorShape(batch_size, 1))

    var predictions = nn.Linear(graph, inputs, n_outputs=1)
    graph.out(predictions)
    graph.loss(nn.MSELoss(graph, predictions, targets))
    return graph^


def run_housing_case() raises:
    comptime batch_size = 64
    comptime graph = create_housing_graph(batch_size)

    var host_inputs = Tensor[f32](batch_size, 13)
    var host_targets = Tensor[f32](batch_size, 1)
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
        print("RESULT,", "housing", ",", trial + 1, ",", seconds)


# ===----------------------------------------------------------------------===#
# Sine: examples/sin_estimate.mojo's 2-hidden-layer MLP, reused at three
# widths to hit ~1k / ~20k / ~350k total parameters.
#
# sin-large uses 592, not the CPU suite's 590: MAX's GPU matmul has a large
# (~20-25x) slowdown whenever the K (reduction) dimension isn't a multiple
# of 16 (590 isn't, 592 is) - unrelated to Mantle's own kernels, but it would
# otherwise make this case measure that cliff instead of steady-state perf.
# ===----------------------------------------------------------------------===#


def create_sin_graph(batch_size: Int, n_hidden: Int) -> Graph:
    var graph = Graph()
    var inputs = graph.input(TensorShape(batch_size, 1))
    var targets = graph.input(TensorShape(batch_size, 1))

    var hidden1 = nn.Linear(graph, inputs, n_outputs=n_hidden)
    var activated1 = nn.ReLU(graph, hidden1)
    var hidden2 = nn.Linear(graph, activated1, n_outputs=n_hidden)
    var activated2 = nn.ReLU(graph, hidden2)
    var predictions = nn.Linear(graph, activated2, n_outputs=1)
    graph.out(predictions)
    graph.loss(nn.MSELoss(graph, predictions, targets))
    return graph^


def run_sin_case[n_hidden: Int](label: String) raises:
    comptime batch_size = 1024
    comptime graph = create_sin_graph(batch_size, n_hidden)

    var host_inputs = Tensor[f32](batch_size, 1)
    var host_targets = Tensor[f32](batch_size, 1)
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


# ===----------------------------------------------------------------------===#
# MNIST: matches examples/mnist.mojo's CNN (conv 16 -> conv 32 -> linear 10,
# ~29k params). Mixed Conv2d/Linear graphs use the explicit-parameters
# zero_grad/step overloads — the stable GPU path for large graphs (see
# training_cnn_gpu.mojo).
# ===----------------------------------------------------------------------===#


@fieldwise_init
struct MNISTCNN(Copyable, Movable):
    var conv1: nn.Conv2dLayer
    var relu1: nn.ReLULayer
    var pool1: nn.MaxPool2dLayer
    var conv2: nn.Conv2dLayer
    var relu2: nn.ReLULayer
    var pool2: nn.MaxPool2dLayer
    var flatten: nn.FlattenLayer
    var head: nn.LinearLayer


def create_mnist_graph(batch_size: Int) -> Graph:
    var network = MNISTCNN(
        nn.Conv2d(16, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(10),
    )
    return nn.classification_graph(network, TensorShape(batch_size, 1, 28, 28))


def run_mnist_case() raises:
    comptime batch_size = 64
    comptime graph = create_mnist_graph(batch_size)

    var host_inputs = Tensor[f32](batch_size, 1, 28, 28)
    var host_targets = Tensor[f32](batch_size, 10)
    rand[f32](host_inputs.ptr(), host_inputs.num_elements())
    for row in range(batch_size):
        host_targets[row * 10] = 1.0
    var inputs = host_inputs.to_gpu()
    var targets = host_targets.to_gpu()

    for trial in range(TRIALS):
        var model = nn.Model[graph, device=Device.gpu]()
        var optimizer = nn.optim.Adam[graph, device=Device.gpu](
            model.parameters, lr=0.001
        )

        for _ in range(WARMUP_STEPS):
            optimizer.zero_grad(model.parameters)
            _ = model.forward(inputs, targets)
            model.backward()
            optimizer.step(model.parameters)
        inputs.gpu_context().synchronize()

        var start = now()
        for _ in range(TIMED_STEPS):
            optimizer.zero_grad(model.parameters)
            _ = model.forward(inputs, targets)
            model.backward()
            optimizer.step(model.parameters)
        inputs.gpu_context().synchronize()
        var seconds = Float64(now() - start) / 1e9
        print("RESULT,", "mnist", ",", trial + 1, ",", seconds)


def main() raises:
    run_housing_case()

    run_sin_case[30]("sin-small")
    run_sin_case[139]("sin-medium")
    run_sin_case[592]("sin-large")

    run_mnist_case()

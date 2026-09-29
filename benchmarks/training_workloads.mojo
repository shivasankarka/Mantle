"""Reproducible CPU training sweep for the Mantle / PyTorch comparison chart.

Each case times 100 MLP/CNN optimization steps after five warm-up steps.
Inputs are allocated once and reused, so the numbers cover forward, loss
backward, and Adam rather than dataset I/O.

Case sizes mirror the actual examples rather than an arbitrary sweep:
housing and mnist each match their example's fixed architecture (14 and
~29k parameters respectively), while sin_estimate's architecture is reused
at three widths to cover ~1k / ~20k / ~350k parameters.
"""
from std.random import rand
from std.time import perf_counter_ns as now

import mantle.nn as nn
from mantle import Graph, Tensor, TensorShape, f32


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

    var inputs = Tensor[f32](batch_size, 13)
    var targets = Tensor[f32](batch_size, 1)
    rand[f32](inputs.ptr(), inputs.num_elements())
    rand[f32](targets.ptr(), targets.num_elements())

    for trial in range(TRIALS):
        var model = nn.Model[graph]()
        var optimizer = nn.optim.Adam[graph](model.parameters, lr=0.001)

        for _ in range(WARMUP_STEPS):
            optimizer.zero_grad()
            _ = model.forward(inputs, targets)
            model.backward()
            optimizer.step()

        var start = now()
        for _ in range(TIMED_STEPS):
            optimizer.zero_grad()
            _ = model.forward(inputs, targets)
            model.backward()
            optimizer.step()
        var seconds = Float64(now() - start) / 1e9
        print("RESULT,", "housing", ",", trial + 1, ",", seconds)


# ===----------------------------------------------------------------------===#
# Sine: examples/sin_estimate.mojo's 2-hidden-layer MLP, reused at three
# widths to hit ~1k / ~20k / ~350k total parameters.
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

    var inputs = Tensor[f32](batch_size, 1)
    var targets = Tensor[f32](batch_size, 1)
    rand[f32](inputs.ptr(), inputs.num_elements())
    rand[f32](targets.ptr(), targets.num_elements())

    for trial in range(TRIALS):
        var model = nn.Model[graph]()
        var optimizer = nn.optim.Adam[graph](model.parameters, lr=0.001)

        for _ in range(WARMUP_STEPS):
            optimizer.zero_grad()
            _ = model.forward(inputs, targets)
            model.backward()
            optimizer.step()

        var start = now()
        for _ in range(TIMED_STEPS):
            optimizer.zero_grad()
            _ = model.forward(inputs, targets)
            model.backward()
            optimizer.step()
        var seconds = Float64(now() - start) / 1e9
        print("RESULT,", label, ",", trial + 1, ",", seconds)


# ===----------------------------------------------------------------------===#
# MNIST: matches examples/mnist.mojo's CNN (conv 16 -> conv 32 -> linear 10,
# ~29k params).
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

    var inputs = Tensor[f32](batch_size, 1, 28, 28)
    var targets = Tensor[f32](batch_size, 10)
    rand[f32](inputs.ptr(), inputs.num_elements())
    for row in range(batch_size):
        targets[row * 10] = 1.0

    for trial in range(TRIALS):
        var model = nn.Model[graph]()
        var optimizer = nn.optim.Adam[graph](model.parameters, lr=0.001)

        for _ in range(WARMUP_STEPS):
            optimizer.zero_grad()
            _ = model.forward(inputs, targets)
            model.backward()
            optimizer.step()

        var start = now()
        for _ in range(TIMED_STEPS):
            optimizer.zero_grad()
            _ = model.forward(inputs, targets)
            model.backward()
            optimizer.step()
        var seconds = Float64(now() - start) / 1e9
        print("RESULT,", "mnist", ",", trial + 1, ",", seconds)


def main() raises:
    run_housing_case()

    run_sin_case[30]("sin-small")
    run_sin_case[139]("sin-medium")
    run_sin_case[590]("sin-large")

    run_mnist_case()

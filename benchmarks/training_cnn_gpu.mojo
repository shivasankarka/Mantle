"""Native GPU CNN training benchmark using MNIST-shaped synthetic batches.

The workload is intentionally self-contained: it measures convolution,
pooling, dense layers, autograd, and Adam without disk I/O or host/device
transfers in the timed region.
"""
from std.random import rand
from std.time import perf_counter_ns as now
from std.utils.index import IndexList

import mantle.nn as nn
from mantle import Graph, Tensor, TensorShape, OP, f32
from mantle.autograd.attributes import Attribute, AttributeVector
from mantle.core.device import Device


comptime BATCH_SIZE = 64
comptime WARMUP_STEPS = 3
comptime TIMED_STEPS = 20
comptime TRIALS = 5


def create_cnn() -> Graph:
    var graph = Graph()
    var inputs = graph.input(TensorShape(BATCH_SIZE, 1, 28, 28))
    var targets = graph.input(TensorShape(BATCH_SIZE, 10))

    var x = nn.Conv2d(
        graph,
        inputs,
        out_channels=16,
        kernel_size=IndexList[2](5, 5),
        padding=IndexList[2](2, 2),
    )
    x = nn.ReLU(graph, x)
    x = nn.MaxPool2d(graph, x, kernel_size=IndexList[2](2, 2))
    x = nn.Conv2d(
        graph,
        x,
        out_channels=32,
        kernel_size=IndexList[2](5, 5),
        padding=IndexList[2](2, 2),
    )
    x = nn.ReLU(graph, x)
    x = nn.MaxPool2d(graph, x, kernel_size=IndexList[2](2, 2))
    x = graph.op(
        OP.RESHAPE,
        x,
        attributes=AttributeVector(
            Attribute("shape", TensorShape(BATCH_SIZE, 32 * 7 * 7))
        ),
    )
    var predictions = nn.Linear(graph, x, n_outputs=10)
    graph.out(predictions)
    graph.loss(nn.MSELoss(graph, predictions, targets))
    return graph^


def main() raises:
    comptime graph = create_cnn()
    var host_inputs = Tensor[f32](TensorShape(BATCH_SIZE, 1, 28, 28))
    var host_targets = Tensor[f32](TensorShape(BATCH_SIZE, 10))
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
        print("RESULT, cnn-mnist ,", trial + 1, ",", seconds)

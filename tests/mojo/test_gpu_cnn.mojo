"""Native GPU Conv2d/MaxPool2d smoke test."""
from std.random import rand
from std.testing import assert_true
from std.utils.index import IndexList

import mantle.nn as nn
from mantle import Graph, Tensor, TensorShape, OP, f32
from mantle.autograd.attributes import Attribute, AttributeVector
from mantle.core.device import Device
from mantle.autograd.ops.gpu_conv import (
    gpu_conv2d_forward_max,
    gpu_conv2d_forward_native,
)


def test_max_conv_matches_native() raises:
    comptime batch = 2
    comptime channels = 2
    comptime height = 8
    comptime width = 8
    comptime out_channels = 3
    comptime kernel_size = 3
    var host_inputs = Tensor[f32](TensorShape(batch, channels, height, width))
    var host_kernel = Tensor[f32](
        TensorShape(out_channels, channels, kernel_size, kernel_size)
    )
    var host_bias = Tensor[f32](TensorShape(out_channels))
    rand[f32](host_inputs.ptr(), host_inputs.num_elements())
    rand[f32](host_kernel.ptr(), host_kernel.num_elements())
    rand[f32](host_bias.ptr(), host_bias.num_elements())
    var inputs = host_inputs.to_gpu()
    var kernel = host_kernel.to_gpu()
    var bias = host_bias.to_gpu()
    var native_output = Tensor[f32, Device.gpu](
        TensorShape(batch, out_channels, height, width), uninitialized=True
    )
    var max_output = Tensor[f32, Device.gpu](
        TensorShape(batch, out_channels, height, width), uninitialized=True
    )
    gpu_conv2d_forward_native[
        batch,
        channels,
        height,
        width,
        out_channels,
        kernel_size,
        kernel_size,
        height,
        width,
        1,
        1,
        1,
        1,
        1,
        1,
    ](native_output, inputs, kernel, bias)
    gpu_conv2d_forward_max[
        batch,
        channels,
        height,
        width,
        out_channels,
        kernel_size,
        kernel_size,
        height,
        width,
        1,
        1,
        1,
        1,
        1,
        1,
    ](max_output, inputs, kernel, bias)
    inputs.gpu_context().synchronize()
    var native_host = native_output.to_host()
    var max_host = max_output.to_host()
    for i in range(native_host.num_elements()):
        var difference = abs(Float32(native_host[i]) - Float32(max_host[i]))
        assert_true(difference < 1e-4, "MAX Conv2d must match native Conv2d")


def make_graph() -> Graph:
    var graph = Graph()
    var inputs = graph.input(TensorShape(2, 1, 8, 8))
    var targets = graph.input(TensorShape(2, 4))
    var x = nn.Conv2d(
        graph,
        inputs,
        out_channels=3,
        kernel_size=IndexList[2](3, 3),
        padding=IndexList[2](1, 1),
    )
    x = nn.ReLU(graph, x)
    x = nn.MaxPool2d(graph, x, kernel_size=IndexList[2](2, 2))
    x = graph.op(
        OP.RESHAPE,
        x,
        attributes=AttributeVector(
            Attribute("shape", TensorShape(2, 3 * 4 * 4))
        ),
    )
    var predictions = nn.Linear(graph, x, n_outputs=4)
    graph.out(predictions)
    graph.loss(nn.MSELoss(graph, predictions, targets))
    return graph^


def main() raises:
    test_max_conv_matches_native()
    comptime graph = make_graph()
    var host_inputs = Tensor[f32](TensorShape(2, 1, 8, 8))
    var host_targets = Tensor[f32](TensorShape(2, 4))
    rand[f32](host_inputs.ptr(), host_inputs.num_elements())
    rand[f32](host_targets.ptr(), host_targets.num_elements())
    var inputs = host_inputs.to_gpu()
    var targets = host_targets.to_gpu()
    var model = nn.Model[graph, device=Device.gpu]()
    var optimizer = nn.optim.Adam[graph, device=Device.gpu](
        model.parameters, lr=0.001
    )

    optimizer.zero_grad(model.parameters)
    var loss = model.forward(inputs, targets)
    inputs.gpu_context().synchronize()
    model.backward()
    inputs.gpu_context().synchronize()
    optimizer.step(model.parameters)
    inputs.gpu_context().synchronize()
    optimizer.zero_grad(model.parameters)
    loss = model.forward(inputs, targets)
    model.backward()
    optimizer.step(model.parameters)
    inputs.gpu_context().synchronize()
    var host_loss = loss.to_host()
    assert_true(host_loss[0] == host_loss[0], "GPU CNN loss should be finite")
    print("test_gpu_cnn: PASSED")

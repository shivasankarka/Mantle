"""GPU ConvTranspose2d: forward/backward parity against the CPU kernels, and
a training smoke test on GPU tensors."""
from std.random import rand
from std.testing import assert_true
from std.utils.index import IndexList

import mantle.nn as nn
from mantle import Graph, Tensor, TensorShape, OP, f32
from mantle.autograd.attributes import Attribute, AttributeVector
from mantle.autograd.ops.ops import forward_op, backward_op
from mantle.autograd.ops.conv_transpose import CONVTRANSPOSE2D
from mantle.core.device import Device
from mantle.nn.model import Model
import mantle.nn.optim as optim


def test_gpu_conv_transpose2d_matches_cpu() raises:
    comptime input_shape = TensorShape(2, 2, 4, 4)
    comptime kernel_shape = TensorShape(2, 3, 3, 3)
    comptime bias_shape = TensorShape(3)
    comptime attrs = AttributeVector(
        Attribute("padding", IndexList[2](1, 1)),
        Attribute("stride", IndexList[2](2, 2)),
        Attribute("dilation", IndexList[2](1, 1)),
        Attribute("output_padding", IndexList[2](1, 1)),
    )
    comptime output_shape = CONVTRANSPOSE2D.result_shape(
        input_shape, kernel_shape, bias_shape, attrs
    )

    var host_inputs = Tensor[f32](input_shape)
    var host_kernel = Tensor[f32](kernel_shape)
    var host_bias = Tensor[f32](bias_shape)
    var host_ug = Tensor[f32](output_shape)
    rand[f32](host_inputs.ptr(), host_inputs.num_elements())
    rand[f32](host_kernel.ptr(), host_kernel.num_elements())
    rand[f32](host_bias.ptr(), host_bias.num_elements())
    rand[f32](host_ug.ptr(), host_ug.num_elements())

    var cpu_out = Tensor[f32](output_shape, uninitialized=True)
    CONVTRANSPOSE2D.forward[input_shape, kernel_shape, bias_shape, attrs](
        cpu_out, host_inputs, host_kernel, host_bias
    )

    var inputs = host_inputs.to_gpu()
    var kernel = host_kernel.to_gpu()
    var bias = host_bias.to_gpu()
    var ug = host_ug.to_gpu()
    var gpu_out = Tensor[f32, Device.gpu](output_shape, uninitialized=True)
    forward_op[
        OP.CONVTRANSPOSE2D, input_shape, kernel_shape, bias_shape, attrs,
        Device.gpu,
    ](gpu_out, inputs, kernel, bias)
    inputs.gpu_context().synchronize()
    var gpu_out_host = gpu_out.to_host()
    for i in range(cpu_out.num_elements()):
        var difference = abs(Float32(cpu_out[i]) - Float32(gpu_out_host[i]))
        assert_true(
            difference < 1e-3, "GPU ConvTranspose2d forward must match CPU"
        )

    comptime for tensor_id in range(3):
        comptime grad_shape = input_shape if tensor_id == 0 else (
            kernel_shape if tensor_id == 1 else bias_shape
        )
        var cpu_grad = CONVTRANSPOSE2D.backward[
            tensor_id, output_shape, input_shape, kernel_shape, bias_shape,
            attrs,
        ](host_ug, host_inputs, host_kernel, host_bias)

        var gpu_grad = Tensor[f32, Device.gpu](grad_shape)
        backward_op[
            tensor_id, OP.CONVTRANSPOSE2D, output_shape, input_shape,
            kernel_shape, bias_shape, attrs, Device.gpu,
        ](ug, inputs, kernel, bias, gpu_grad)
        inputs.gpu_context().synchronize()
        var gpu_grad_host = gpu_grad.to_host()
        for i in range(cpu_grad.num_elements()):
            var difference = abs(
                Float32(cpu_grad[i]) - Float32(gpu_grad_host[i])
            )
            assert_true(
                difference < 1e-2,
                "GPU ConvTranspose2d backward[tensor_id="
                + String(tensor_id)
                + "] must match CPU",
            )

    print("test_gpu_conv_transpose2d_matches_cpu: PASSED")


def make_graph() -> Graph:
    var graph = Graph()
    var inputs = graph.input(TensorShape(2, 2, 4, 4))
    var targets = graph.input(TensorShape(2, 3, 8, 8))
    var out = nn.ConvTranspose2d(
        graph,
        inputs,
        out_channels=3,
        kernel_size=IndexList[2](4, 4),
        padding=IndexList[2](1, 1),
        stride=IndexList[2](2, 2),
    )
    graph.out(out)
    graph.loss(nn.MSELoss(graph, out, targets))
    return graph^


def test_gpu_conv_transpose2d_trains() raises:
    comptime graph = make_graph()
    var host_inputs = Tensor[f32](TensorShape(2, 2, 4, 4))
    var host_targets = Tensor[f32](TensorShape(2, 3, 8, 8))
    rand[f32](host_inputs.ptr(), host_inputs.num_elements())
    rand[f32](host_targets.ptr(), host_targets.num_elements())
    var inputs = host_inputs.to_gpu()
    var targets = host_targets.to_gpu()
    var model = nn.Model[graph, device=Device.gpu]()
    var adam = optim.Adam[graph, device=Device.gpu](model.parameters, lr=0.01)

    var initial_loss = model.forward(inputs, targets).to_host()
    for _ in range(10):
        adam.zero_grad(model.parameters)
        _ = model.forward(inputs, targets)
        model.backward()
        adam.step(model.parameters)
    inputs.gpu_context().synchronize()
    var final_loss = model.forward(inputs, targets).to_host()

    assert_true(
        final_loss[0] < initial_loss[0],
        "GPU ConvTranspose2d training should decrease loss (got "
        + String(initial_loss[0])
        + " -> "
        + String(final_loss[0])
        + ")",
    )
    print(
        "test_gpu_conv_transpose2d_trains: PASSED (loss "
        + String(initial_loss[0])
        + " -> "
        + String(final_loss[0])
        + ")"
    )


def main() raises:
    test_gpu_conv_transpose2d_matches_cpu()
    test_gpu_conv_transpose2d_trains()

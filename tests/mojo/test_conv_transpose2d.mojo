"""Test ConvTranspose2d: shape correctness, training convergence, and a
finite-difference gradient check against the analytic backward pass."""
from std.testing import assert_true

from mantle import f32
from mantle.autograd.graph import Graph
from mantle.autograd.ops import OP
from mantle.autograd.ops.ops import backward_op
from mantle.autograd.ops.conv_transpose import CONVTRANSPOSE2D
from mantle.autograd.attributes import AttributeVector, Attribute
from mantle.core.tensor import Tensor, TensorShape
from mantle.core.tensorutils import fill
from std.utils.index import IndexList
from std.random import rand
import mantle.nn as nn
from mantle.nn.model import Model
import mantle.nn.optim as optim


def make_graph(
    batch: Int, in_channels: Int, size: Int, out_channels: Int, k: Int
) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch, in_channels, size, size))
    var out = nn.ConvTranspose2d(
        g, x, out_channels, kernel_size=IndexList[2](k, k), stride=IndexList[2](2, 2), padding=IndexList[2](1, 1)
    )
    g.out(out)
    return g^


def test_conv_transpose2d_output_shape() raises:
    comptime batch = 2
    comptime in_channels = 3
    comptime size = 4
    comptime out_channels = 5
    comptime k = 4
    comptime g = make_graph(batch, in_channels, size, out_channels, k)
    var model = Model[g](inference_only=True)

    var x = Tensor[f32](TensorShape(batch, in_channels, size, size))
    fill(x, 0.1)

    var out = model.inference(x)[0].copy()
    # stride=2, padding=1, k=4 -> out = (size-1)*2 - 2*1 + (4-1) + 1 = 2*size.
    assert_true(
        out.shape() == TensorShape(batch, out_channels, 2 * size, 2 * size),
        "output shape matches (batch, out_channels, 2*size, 2*size), got "
        + String(out.shape()),
    )
    print("test_conv_transpose2d_output_shape: PASSED")


def make_train_graph(
    batch: Int, in_channels: Int, size: Int, out_channels: Int, k: Int
) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch, in_channels, size, size))
    var y = g.input(TensorShape(batch, out_channels, 2 * size, 2 * size))
    var out = nn.ConvTranspose2d(
        g, x, out_channels, kernel_size=IndexList[2](k, k), stride=IndexList[2](2, 2), padding=IndexList[2](1, 1)
    )
    var loss = nn.MSELoss(g, out, y)
    g.loss(loss)
    return g^


def test_conv_transpose2d_trains() raises:
    comptime batch = 2
    comptime in_channels = 3
    comptime size = 4
    comptime out_channels = 5
    comptime k = 4
    comptime g = make_train_graph(
        batch, in_channels, size, out_channels, k
    )
    var model = Model[g]()
    var sgd = optim.SGD[g](model.parameters, lr=0.05)

    var x = Tensor[f32](TensorShape(batch, in_channels, size, size))
    var y = Tensor[f32](
        TensorShape(batch, out_channels, 2 * size, 2 * size)
    )
    fill(x, 0.5)
    fill(y, 0.1)

    var initial_loss: Float32 = model.forward(x, y)[0]
    for _ in range(15):
        _ = model.forward(x, y)
        model.backward()
        sgd.step()
        sgd.zero_grad()

    var final_loss: Float32 = model.forward(x, y)[0]
    assert_true(
        final_loss < initial_loss,
        "ConvTranspose2d training should decrease loss (got "
        + String(initial_loss)
        + " -> "
        + String(final_loss)
        + ")",
    )
    print(
        "test_conv_transpose2d_trains: PASSED (loss "
        + String(initial_loss)
        + " -> "
        + String(final_loss)
        + ")"
    )


# ===----------------------------------------------------------------------===#
# Finite-difference gradient check
# ===----------------------------------------------------------------------===#


def loss(
    inputs: Tensor[f32], kernel: Tensor[f32], bias: Tensor[f32], ug: Tensor[f32]
) raises -> Scalar[f32]:
    """`sum(ug * ConvTranspose2d(inputs, kernel, bias))`, whose gradient wrt
    each argument is exactly what `backward_op` should compute given `ug`."""
    comptime input_shape = TensorShape(1, 2, 3, 3)
    comptime kernel_shape = TensorShape(2, 2, 3, 3)
    comptime bias_shape = TensorShape(2)
    comptime attrs = AttributeVector(
        Attribute("padding", IndexList[2](1, 1)),
        Attribute("stride", IndexList[2](2, 2)),
        Attribute("dilation", IndexList[2](1, 1)),
        Attribute("output_padding", IndexList[2](1, 1)),
    )
    comptime output_shape = CONVTRANSPOSE2D.result_shape(
        input_shape, kernel_shape, bias_shape, attrs
    )
    var out = Tensor[f32](output_shape, uninitialized=True)
    CONVTRANSPOSE2D.forward[input_shape, kernel_shape, bias_shape, attrs](
        out, inputs, kernel, bias
    )
    var total: Scalar[f32] = 0
    for i in range(out.num_elements()):
        total += out[i] * ug[i]
    return total


def test_conv_transpose2d_gradcheck() raises:
    comptime input_shape = TensorShape(1, 2, 3, 3)
    comptime kernel_shape = TensorShape(2, 2, 3, 3)
    comptime bias_shape = TensorShape(2)
    comptime attrs = AttributeVector(
        Attribute("padding", IndexList[2](1, 1)),
        Attribute("stride", IndexList[2](2, 2)),
        Attribute("dilation", IndexList[2](1, 1)),
        Attribute("output_padding", IndexList[2](1, 1)),
    )
    comptime output_shape = CONVTRANSPOSE2D.result_shape(
        input_shape, kernel_shape, bias_shape, attrs
    )

    var inputs = Tensor[f32](input_shape)
    var kernel = Tensor[f32](kernel_shape)
    var bias = Tensor[f32](bias_shape)
    var ug = Tensor[f32](output_shape)
    rand[f32](inputs.ptr(), inputs.num_elements())
    rand[f32](kernel.ptr(), kernel.num_elements())
    rand[f32](bias.ptr(), bias.num_elements())
    rand[f32](ug.ptr(), ug.num_elements())

    comptime eps: Scalar[f32] = 1e-2

    # --- grad wrt input ---
    var analytic_input_grad = Tensor[f32](input_shape)
    backward_op[
        0, OP.CONVTRANSPOSE2D, output_shape, input_shape, kernel_shape,
        bias_shape, attrs,
    ](ug, inputs, kernel, bias, analytic_input_grad)
    for i in range(inputs.num_elements()):
        var orig = inputs[i]
        inputs[i] = orig + eps
        var plus = loss(inputs, kernel, bias, ug)
        inputs[i] = orig - eps
        var minus = loss(inputs, kernel, bias, ug)
        inputs[i] = orig
        var numeric = (plus - minus) / (2 * eps)
        assert_true(
            abs(numeric - analytic_input_grad[i]) < 1e-2,
            "input grad mismatch at "
            + String(i)
            + ": numeric="
            + String(numeric)
            + " analytic="
            + String(analytic_input_grad[i]),
        )

    # --- grad wrt kernel ---
    var analytic_kernel_grad = Tensor[f32](kernel_shape)
    backward_op[
        1, OP.CONVTRANSPOSE2D, output_shape, input_shape, kernel_shape,
        bias_shape, attrs,
    ](ug, inputs, kernel, bias, analytic_kernel_grad)
    for i in range(kernel.num_elements()):
        var orig = kernel[i]
        kernel[i] = orig + eps
        var plus = loss(inputs, kernel, bias, ug)
        kernel[i] = orig - eps
        var minus = loss(inputs, kernel, bias, ug)
        kernel[i] = orig
        var numeric = (plus - minus) / (2 * eps)
        assert_true(
            abs(numeric - analytic_kernel_grad[i]) < 1e-2,
            "kernel grad mismatch at "
            + String(i)
            + ": numeric="
            + String(numeric)
            + " analytic="
            + String(analytic_kernel_grad[i]),
        )

    # --- grad wrt bias ---
    var analytic_bias_grad = Tensor[f32](bias_shape)
    backward_op[
        2, OP.CONVTRANSPOSE2D, output_shape, input_shape, kernel_shape,
        bias_shape, attrs,
    ](ug, inputs, kernel, bias, analytic_bias_grad)
    for i in range(bias.num_elements()):
        var orig = bias[i]
        bias[i] = orig + eps
        var plus = loss(inputs, kernel, bias, ug)
        bias[i] = orig - eps
        var minus = loss(inputs, kernel, bias, ug)
        bias[i] = orig
        var numeric = (plus - minus) / (2 * eps)
        assert_true(
            abs(numeric - analytic_bias_grad[i]) < 1e-2,
            "bias grad mismatch at "
            + String(i)
            + ": numeric="
            + String(numeric)
            + " analytic="
            + String(analytic_bias_grad[i]),
        )

    print("test_conv_transpose2d_gradcheck: PASSED")


def main() raises:
    test_conv_transpose2d_output_shape()
    test_conv_transpose2d_trains()
    test_conv_transpose2d_gradcheck()

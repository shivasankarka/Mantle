"""Test Conv1d: shape correctness and training convergence."""
from std.testing import assert_true

from mantle import f32
from mantle.autograd.graph import Graph
from mantle.core.tensor import Tensor, TensorShape
from mantle.core.tensorutils import fill
import mantle.nn as nn
from mantle.nn.model import Model
import mantle.nn.optim as optim


def make_graph(
    batch: Int, in_channels: Int, length: Int, out_channels: Int, k: Int
) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch, in_channels, length))
    var out = nn.Conv1d(g, x, out_channels, kernel_size=k, padding=1)
    g.out(out)
    return g^


def test_conv1d_output_shape() raises:
    comptime batch = 2
    comptime in_channels = 3
    comptime length = 10
    comptime out_channels = 4
    comptime k = 3
    comptime g = make_graph(batch, in_channels, length, out_channels, k)
    var model = Model[g](inference_only=True)

    var x = Tensor[f32](TensorShape(batch, in_channels, length))
    fill(x, 0.1)

    var out = model.inference(x)[0].copy()
    # padding=1, stride=1, k=3 -> same length as input.
    assert_true(
        out.shape() == TensorShape(batch, out_channels, length),
        "output shape matches (batch, out_channels, length)",
    )
    print("test_conv1d_output_shape: PASSED")


def make_train_graph(
    batch: Int, in_channels: Int, length: Int, out_channels: Int, k: Int
) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch, in_channels, length))
    var y = g.input(TensorShape(batch, out_channels, length))
    var out = nn.Conv1d(g, x, out_channels, kernel_size=k, padding=1)
    var loss = nn.MSELoss(g, out, y)
    g.loss(loss)
    return g^


def test_conv1d_trains() raises:
    comptime batch = 2
    comptime in_channels = 3
    comptime length = 10
    comptime out_channels = 4
    comptime k = 3
    comptime g = make_train_graph(batch, in_channels, length, out_channels, k)
    var model = Model[g]()
    var sgd = optim.SGD[g](model.parameters, lr=0.05)

    var x = Tensor[f32](TensorShape(batch, in_channels, length))
    var y = Tensor[f32](TensorShape(batch, out_channels, length))
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
        "Conv1d training should decrease loss (got "
        + String(initial_loss)
        + " -> "
        + String(final_loss)
        + ")",
    )
    print(
        "test_conv1d_trains: PASSED (loss "
        + String(initial_loss)
        + " -> "
        + String(final_loss)
        + ")"
    )


def main() raises:
    test_conv1d_output_shape()
    test_conv1d_trains()

"""Test FeedForward block: shape correctness and training convergence."""
from std.testing import assert_true

from mantle import f32
from mantle.autograd.graph import Graph
from mantle.core.tensor import Tensor, TensorShape
from mantle.core.tensorutils import fill
import mantle.nn as nn
from mantle.nn.model import Model
import mantle.nn.optim as optim


def make_graph(b: Int, t: Int, d: Int, d_ff: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(b, t, d))
    var out = nn.FeedForward(g, x, d_ff)
    g.out(out)
    return g^


def test_feedforward_output_shape() raises:
    comptime b = 2
    comptime t = 3
    comptime d = 8
    comptime d_ff = 32
    comptime g = make_graph(b, t, d, d_ff)
    var model = Model[g](inference_only=True)

    var x = Tensor[f32](TensorShape(b, t, d))
    fill(x, 0.1)

    var out = model.inference(x)[0].copy()
    assert_true(out.shape() == TensorShape(b, t, d), "output shape matches input")
    print("test_feedforward_output_shape: PASSED")


def make_train_graph(b: Int, t: Int, d: Int, d_ff: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(b, t, d))
    var y = g.input(TensorShape(b, t, d))
    var out = nn.FeedForward(g, x, d_ff)
    var loss = nn.MSELoss(g, out, y)
    g.loss(loss)
    return g^


def test_feedforward_trains() raises:
    comptime b = 2
    comptime t = 3
    comptime d = 8
    comptime d_ff = 32
    comptime g = make_train_graph(b, t, d, d_ff)
    var model = Model[g]()
    var sgd = optim.SGD[g](model.parameters, lr=0.05)

    var x = Tensor[f32](TensorShape(b, t, d))
    var y = Tensor[f32](TensorShape(b, t, d))
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
        "FeedForward training should decrease loss (got "
        + String(initial_loss)
        + " -> "
        + String(final_loss)
        + ")",
    )
    print(
        "test_feedforward_trains: PASSED (loss "
        + String(initial_loss)
        + " -> "
        + String(final_loss)
        + ")"
    )


def main() raises:
    test_feedforward_output_shape()
    test_feedforward_trains()

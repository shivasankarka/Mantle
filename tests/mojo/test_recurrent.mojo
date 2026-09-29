"""Test LSTM/GRU: shape correctness and training convergence."""
from std.testing import assert_true

from mantle import f32
from mantle.autograd.graph import Graph
from mantle.core.tensor import Tensor, TensorShape
from mantle.core.tensorutils import fill
import mantle.nn as nn
from mantle.nn.model import Model
import mantle.nn.optim as optim


# ===----------------------------------------------------------------------===#
# LSTM
# ===----------------------------------------------------------------------===#


def make_lstm_graph(b: Int, t: Int, d: Int, h: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(b, t, d))
    var out = nn.LSTM(g, x, h)
    g.out(out)
    return g^


def test_lstm_output_shape() raises:
    comptime b = 2
    comptime t = 4
    comptime d = 3
    comptime h = 5
    comptime g = make_lstm_graph(b, t, d, h)
    var model = Model[g](inference_only=True)

    var x = Tensor[f32](TensorShape(b, t, d))
    fill(x, 0.1)

    var out = model.inference(x)[0].copy()
    assert_true(
        out.shape() == TensorShape(b, t, h), "output shape matches (b, t, h)"
    )
    print("test_lstm_output_shape: PASSED")


def make_lstm_train_graph(b: Int, t: Int, d: Int, h: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(b, t, d))
    var y = g.input(TensorShape(b, t, h))
    var out = nn.LSTM(g, x, h)
    var loss = nn.MSELoss(g, out, y)
    g.loss(loss)
    return g^


def test_lstm_trains() raises:
    comptime b = 2
    comptime t = 4
    comptime d = 3
    comptime h = 5
    comptime g = make_lstm_train_graph(b, t, d, h)
    var model = Model[g]()
    var sgd = optim.SGD[g](model.parameters, lr=0.1)

    var x = Tensor[f32](TensorShape(b, t, d))
    var y = Tensor[f32](TensorShape(b, t, h))
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
        "LSTM training should decrease loss (got "
        + String(initial_loss)
        + " -> "
        + String(final_loss)
        + ")",
    )
    print(
        "test_lstm_trains: PASSED (loss "
        + String(initial_loss)
        + " -> "
        + String(final_loss)
        + ")"
    )


# ===----------------------------------------------------------------------===#
# GRU
# ===----------------------------------------------------------------------===#


def make_gru_graph(b: Int, t: Int, d: Int, h: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(b, t, d))
    var out = nn.GRU(g, x, h)
    g.out(out)
    return g^


def test_gru_output_shape() raises:
    comptime b = 2
    comptime t = 4
    comptime d = 3
    comptime h = 5
    comptime g = make_gru_graph(b, t, d, h)
    var model = Model[g](inference_only=True)

    var x = Tensor[f32](TensorShape(b, t, d))
    fill(x, 0.1)

    var out = model.inference(x)[0].copy()
    assert_true(
        out.shape() == TensorShape(b, t, h), "output shape matches (b, t, h)"
    )
    print("test_gru_output_shape: PASSED")


def make_gru_train_graph(b: Int, t: Int, d: Int, h: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(b, t, d))
    var y = g.input(TensorShape(b, t, h))
    var out = nn.GRU(g, x, h)
    var loss = nn.MSELoss(g, out, y)
    g.loss(loss)
    return g^


def test_gru_trains() raises:
    comptime b = 2
    comptime t = 4
    comptime d = 3
    comptime h = 5
    comptime g = make_gru_train_graph(b, t, d, h)
    var model = Model[g]()
    var sgd = optim.SGD[g](model.parameters, lr=0.1)

    var x = Tensor[f32](TensorShape(b, t, d))
    var y = Tensor[f32](TensorShape(b, t, h))
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
        "GRU training should decrease loss (got "
        + String(initial_loss)
        + " -> "
        + String(final_loss)
        + ")",
    )
    print(
        "test_gru_trains: PASSED (loss "
        + String(initial_loss)
        + " -> "
        + String(final_loss)
        + ")"
    )


def main() raises:
    test_lstm_output_shape()
    test_lstm_trains()
    test_gru_output_shape()
    test_gru_trains()

"""Test PositionalEmbedding: shape correctness and training convergence."""
from std.testing import assert_true

from mantle import f32
from mantle.autograd.graph import Graph
from mantle.core.tensor import Tensor, TensorShape
from mantle.core.tensorutils import fill
import mantle.nn as nn
from mantle.nn.model import Model
import mantle.nn.optim as optim


def make_graph(b: Int, t: Int, d: Int, max_seq_len: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(b, t, d))
    var out = nn.PositionalEmbedding(g, x, max_seq_len)
    g.out(out)
    return g^


def test_positional_embedding_output_shape() raises:
    comptime b = 2
    comptime t = 3
    comptime d = 4
    comptime max_seq_len = 8
    comptime g = make_graph(b, t, d, max_seq_len)
    var model = Model[g](inference_only=True)

    var x = Tensor[f32](TensorShape(b, t, d))
    fill(x, 0.0)

    var out = model.inference(x)[0].copy()
    assert_true(out.shape() == TensorShape(b, t, d), "output shape matches input")
    print("test_positional_embedding_output_shape: PASSED")


def test_positional_embedding_full_seq_len() raises:
    # seq_len == max_seq_len exercises the no-slice branch
    comptime b = 1
    comptime t = 4
    comptime d = 4
    comptime g = make_graph(b, t, d, t)
    var model = Model[g](inference_only=True)

    var x = Tensor[f32](TensorShape(b, t, d))
    fill(x, 0.0)

    var out = model.inference(x)[0].copy()
    assert_true(out.shape() == TensorShape(b, t, d), "full-seq-len output shape matches input")
    print("test_positional_embedding_full_seq_len: PASSED")


def make_train_graph(b: Int, t: Int, d: Int, max_seq_len: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(b, t, d))
    var y = g.input(TensorShape(b, t, d))
    var out = nn.PositionalEmbedding(g, x, max_seq_len)
    var loss = nn.MSELoss(g, out, y)
    g.loss(loss)
    return g^


def test_positional_embedding_trains() raises:
    comptime b = 2
    comptime t = 3
    comptime d = 4
    comptime max_seq_len = 8
    comptime g = make_train_graph(b, t, d, max_seq_len)
    var model = Model[g]()
    var sgd = optim.SGD[g](model.parameters, lr=0.1)

    var x = Tensor[f32](TensorShape(b, t, d))
    var y = Tensor[f32](TensorShape(b, t, d))
    fill(x, 0.0)
    fill(y, 1.0)

    var initial_loss: Float32 = model.forward(x, y)[0]
    for _ in range(15):
        _ = model.forward(x, y)
        model.backward()
        sgd.step()
        sgd.zero_grad()

    var final_loss: Float32 = model.forward(x, y)[0]
    assert_true(
        final_loss < initial_loss,
        "PositionalEmbedding training should decrease loss (got "
        + String(initial_loss)
        + " -> "
        + String(final_loss)
        + ")",
    )
    print(
        "test_positional_embedding_trains: PASSED (loss "
        + String(initial_loss)
        + " -> "
        + String(final_loss)
        + ")"
    )


def main() raises:
    test_positional_embedding_output_shape()
    test_positional_embedding_full_seq_len()
    test_positional_embedding_trains()

"""Test MultiHeadAttention: shape correctness, causal masking, and training
convergence."""
from std.testing import assert_true
from std.math import abs

from mantle import f32
from mantle.autograd.graph import Graph
from mantle.core.tensor import Tensor, TensorShape
from mantle.core.tensorutils import fill
import mantle.nn as nn
from mantle.nn.model import Model
import mantle.nn.optim as optim


# ===----------------------------------------------------------------------===#
# Shape / inference correctness
# ===----------------------------------------------------------------------===#


def make_mha_graph(
    b: Int, t: Int, d: Int, heads: Int, causal: Bool
) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(b, t, d))
    var out = nn.MultiHeadAttention(g, x, heads, causal=causal)
    g.out(out)
    return g^


def test_mha_output_shape() raises:
    comptime b = 2
    comptime t = 4
    comptime d = 8
    comptime heads = 2
    comptime g = make_mha_graph(b, t, d, heads, False)
    var model = Model[g](inference_only=True)

    var x = Tensor[f32](TensorShape(b, t, d))
    fill(x, 0.1)

    var out = model.inference(x)[0].copy()
    assert_true(out.shape() == TensorShape(b, t, d), "output shape matches input")
    print("test_mha_output_shape: PASSED")


def test_mha_causal_first_token_ignores_future() raises:
    # With a causal mask, the output at position 0 should be identical
    # regardless of what values occupy positions > 0 (since it can only
    # attend to itself).
    comptime b = 1
    comptime t = 3
    comptime d = 4
    comptime heads = 1
    comptime g = make_mha_graph(b, t, d, heads, True)
    var model = Model[g](inference_only=True)

    var x1 = Tensor[f32](TensorShape(b, t, d))
    for i in range(d):
        x1[i] = Float32(i) * 0.1  # position 0
    for i in range(d, 2 * d):
        x1[i] = 5.0  # position 1
    for i in range(2 * d, 3 * d):
        x1[i] = -5.0  # position 2

    var x2 = Tensor[f32](TensorShape(b, t, d))
    for i in range(d):
        x2[i] = Float32(i) * 0.1  # same position 0
    for i in range(d, 2 * d):
        x2[i] = 100.0  # different position 1
    for i in range(2 * d, 3 * d):
        x2[i] = -100.0  # different position 2

    var out1 = model.inference(x1)[0].copy()
    var out2 = model.inference(x2)[0].copy()

    for i in range(d):
        assert_true(
            abs(Float32(out1[i]) - Float32(out2[i])) < 1e-4,
            "position 0 output should be unaffected by future positions",
        )

    print("test_mha_causal_first_token_ignores_future: PASSED")


# ===----------------------------------------------------------------------===#
# Training convergence
# ===----------------------------------------------------------------------===#


def make_train_graph(b: Int, t: Int, d: Int, heads: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(b, t, d))
    var y = g.input(TensorShape(b, t, d))
    var out = nn.MultiHeadAttention(g, x, heads, causal=True)
    var loss = nn.MSELoss(g, out, y)
    g.loss(loss)
    return g^


def test_mha_trains() raises:
    comptime b = 2
    comptime t = 4
    comptime d = 8
    comptime heads = 2
    comptime g = make_train_graph(b, t, d, heads)
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
        "MHA training should decrease loss (got "
        + String(initial_loss)
        + " -> "
        + String(final_loss)
        + ")",
    )
    print(
        "test_mha_trains: PASSED (loss "
        + String(initial_loss)
        + " -> "
        + String(final_loss)
        + ")"
    )


def main() raises:
    test_mha_output_shape()
    test_mha_causal_first_token_ignores_future()
    test_mha_trains()

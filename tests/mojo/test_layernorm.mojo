"""Test LayerNorm composite: normalizes to ~0 mean/~1 std, and trains."""
from std.testing import assert_true
from std.math import abs, sqrt

from mantle import f32
from mantle.autograd.graph import Graph
from mantle.core.tensor import Tensor, TensorShape
from mantle.core.tensorutils import fill
import mantle.nn as nn
from mantle.nn.model import Model
import mantle.nn.optim as optim


# ===----------------------------------------------------------------------===#
# Normalization correctness (gamma=1, beta=0 at init)
# ===----------------------------------------------------------------------===#


def make_layernorm_graph(batch: Int, d: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch, d))
    var normed = nn.LayerNorm(g, x, d)
    g.out(normed)
    return g^


def test_layernorm_normalizes() raises:
    comptime batch = 2
    comptime d = 4
    comptime g = make_layernorm_graph(batch, d)
    var model = Model[g](inference_only=True)

    var x = Tensor[f32](TensorShape(batch, d))
    x[0] = 1.0; x[1] = 2.0; x[2] = 3.0; x[3] = 4.0
    x[4] = 10.0; x[5] = 0.0; x[6] = -10.0; x[7] = 20.0

    var out = model.inference(x)[0].copy()

    for row in range(batch):
        var mean: Float32 = 0.0
        for col in range(d):
            mean += Float32(out[row * d + col])
        mean /= Float32(d)
        assert_true(abs(mean) < 1e-4, "row " + String(row) + " mean ~0")

        var var_: Float32 = 0.0
        for col in range(d):
            var diff = Float32(out[row * d + col]) - mean
            var_ += diff * diff
        var_ /= Float32(d)
        assert_true(
            abs(sqrt(var_) - 1.0) < 1e-3, "row " + String(row) + " std ~1"
        )

    print("test_layernorm_normalizes: PASSED")


# ===----------------------------------------------------------------------===#
# Training convergence (Linear + LayerNorm stack)
# ===----------------------------------------------------------------------===#


def make_train_graph(batch: Int, d: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch, d))
    var y = g.input(TensorShape(batch, d))
    var lin = nn.Linear(g, x, d)
    var normed = nn.LayerNorm(g, lin, d)
    var loss = nn.MSELoss(g, normed, y)
    g.loss(loss)
    return g^


def test_layernorm_trains() raises:
    comptime batch = 4
    comptime d = 3
    comptime g = make_train_graph(batch, d)
    var model = Model[g]()
    var sgd = optim.SGD[g](model.parameters, lr=0.05)

    var x = Tensor[f32](TensorShape(batch, d))
    var y = Tensor[f32](TensorShape(batch, d))
    fill(x, 1.0)
    fill(y, 0.5)

    var initial_loss: Float32 = model.forward(x, y)[0]
    for _ in range(20):
        _ = model.forward(x, y)
        model.backward()
        sgd.step()
        sgd.zero_grad()

    var final_loss: Float32 = model.forward(x, y)[0]
    assert_true(
        final_loss < initial_loss,
        "LayerNorm stack training should decrease loss (got "
        + String(initial_loss)
        + " -> "
        + String(final_loss)
        + ")",
    )
    print(
        "test_layernorm_trains: PASSED (loss "
        + String(initial_loss)
        + " -> "
        + String(final_loss)
        + ")"
    )


def main() raises:
    test_layernorm_normalizes()
    test_layernorm_trains()

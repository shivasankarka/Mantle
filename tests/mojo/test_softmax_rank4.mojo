"""Verify Softmax composite works on rank-4 tensors with axis=-1 (last axis),
the shape attention scores need: (B, H, T, T)."""
from std.testing import assert_true
from std.math import abs

from mantle import f32
from mantle.autograd.graph import Graph
from mantle.autograd.ops import OP
from mantle.autograd.params import Param
from mantle.core.tensor import Tensor, TensorShape
from mantle.core.tensorutils import fill
import mantle.nn as nn
from mantle.nn.model import Model
import mantle.nn.optim as optim


def make_graph(b: Int, h: Int, t: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(b, h, t, t))
    var out = nn.Softmax(g, x, axis=3)
    g.out(out)
    return g^


def test_softmax_rank4_sums_to_one() raises:
    comptime b = 2
    comptime h = 2
    comptime t = 3
    comptime g = make_graph(b, h, t)
    var model = Model[g](inference_only=True)

    var x = Tensor[f32](TensorShape(b, h, t, t))
    for i in range(x.num_elements()):
        x[i] = Float32(i % 7) - 3.0

    var out = model.inference(x)[0].copy()

    for bi in range(b):
        for hi in range(h):
            for ti in range(t):
                var s: Float32 = 0.0
                for tj in range(t):
                    var idx = ((bi * h + hi) * t + ti) * t + tj
                    var v = Float32(out[idx])
                    assert_true(v > 0.0 and v < 1.0, "value in (0,1)")
                    s += v
                assert_true(abs(s - 1.0) < 1e-4, "row sums to 1")

    print("test_softmax_rank4_sums_to_one: PASSED")


# ===----------------------------------------------------------------------===#
# Training convergence through rank-4 Softmax (backward correctness)
# ===----------------------------------------------------------------------===#


def make_train_graph(b: Int, h: Int, t: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(b, h, t, t))
    var y = g.input(TensorShape(b, h, t, t))
    var bias = g.param(TensorShape(t, t), init=Param("constant", Scalar[f32](0.0), Scalar[f32](0.0)))
    var scaled = g.op(OP.ADD, x, bias)
    var probs = nn.Softmax(g, scaled, axis=3)
    var loss = nn.MSELoss(g, probs, y)
    g.loss(loss)
    return g^


def test_softmax_rank4_backward_trains() raises:
    comptime b = 2
    comptime h = 2
    comptime t = 3
    comptime g = make_train_graph(b, h, t)
    var model = Model[g]()
    var sgd = optim.SGD[g](model.parameters, lr=0.1)

    var x = Tensor[f32](TensorShape(b, h, t, t))
    var y = Tensor[f32](TensorShape(b, h, t, t))
    for i in range(x.num_elements()):
        x[i] = Float32(i % 5) - 2.0
    fill(y, 1.0 / Float32(t))

    var initial_loss: Float32 = model.forward(x, y)[0]
    for _ in range(15):
        _ = model.forward(x, y)
        model.backward()
        sgd.step()
        sgd.zero_grad()

    var final_loss: Float32 = model.forward(x, y)[0]
    assert_true(
        final_loss < initial_loss,
        "rank-4 softmax training should decrease loss (got "
        + String(initial_loss)
        + " -> "
        + String(final_loss)
        + ")",
    )
    print(
        "test_softmax_rank4_backward_trains: PASSED (loss "
        + String(initial_loss)
        + " -> "
        + String(final_loss)
        + ")"
    )


def main() raises:
    test_softmax_rank4_sums_to_one()
    test_softmax_rank4_backward_trains()

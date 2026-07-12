"""Test NEG, ABS, SQRT ops and L1Loss."""
from std.testing import assert_true
from std.math import abs, sqrt

from mantle import f32
from mantle.autograd.graph import Graph
from mantle.autograd.ops import OP
from mantle.core.tensor import Tensor, TensorShape
from mantle.core.tensorutils import fill
import mantle.nn as nn
from mantle.nn.model import Model
import mantle.nn.optim as optim


# ===----------------------------------------------------------------------===#
# Graph helpers
# ===----------------------------------------------------------------------===#


def make_unary_graph(op: OP, n: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(n))
    var out = g.op(op, x)
    g.out(out)
    return g^


def make_l1_graph(batch: Int, n: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch, n))
    var y = g.input(TensorShape(batch, n))
    var linear_out = nn.Linear(g, x, n)
    var loss = nn.L1Loss(g, linear_out, y)
    g.loss(loss)
    return g^


# ===----------------------------------------------------------------------===#
# NEG
# ===----------------------------------------------------------------------===#


def test_neg() raises:
    comptime g = make_unary_graph(OP.NEG, 4)
    var model = Model[g](inference_only=True)

    var x = Tensor[f32](TensorShape(4))
    x[0] = 1.0; x[1] = -2.0; x[2] = 0.0; x[3] = 3.5

    var out = model.inference(x)[0].copy()

    assert_true(abs(Float32(out[0]) - (-1.0)) < 1e-6, "NEG[0]")
    assert_true(abs(Float32(out[1]) - (2.0)) < 1e-6, "NEG[1]")
    assert_true(abs(Float32(out[2]) - (0.0)) < 1e-6, "NEG[2]")
    assert_true(abs(Float32(out[3]) - (-3.5)) < 1e-6, "NEG[3]")
    print("test_neg: PASSED")


# ===----------------------------------------------------------------------===#
# ABS
# ===----------------------------------------------------------------------===#


def test_abs() raises:
    comptime g = make_unary_graph(OP.ABS, 4)
    var model = Model[g](inference_only=True)

    var x = Tensor[f32](TensorShape(4))
    x[0] = 1.0; x[1] = -2.0; x[2] = 0.0; x[3] = -3.5

    var out = model.inference(x)[0].copy()

    assert_true(abs(Float32(out[0]) - 1.0) < 1e-6, "ABS[0]")
    assert_true(abs(Float32(out[1]) - 2.0) < 1e-6, "ABS[1]")
    assert_true(abs(Float32(out[2]) - 0.0) < 1e-6, "ABS[2]")
    assert_true(abs(Float32(out[3]) - 3.5) < 1e-6, "ABS[3]")
    print("test_abs: PASSED")


# ===----------------------------------------------------------------------===#
# SQRT
# ===----------------------------------------------------------------------===#


def test_sqrt() raises:
    comptime g = make_unary_graph(OP.SQRT, 4)
    var model = Model[g](inference_only=True)

    var x = Tensor[f32](TensorShape(4))
    x[0] = 0.0; x[1] = 1.0; x[2] = 4.0; x[3] = 9.0

    var out = model.inference(x)[0].copy()

    assert_true(abs(Float32(out[0]) - 0.0) < 1e-6, "SQRT[0]")
    assert_true(abs(Float32(out[1]) - 1.0) < 1e-6, "SQRT[1]")
    assert_true(abs(Float32(out[2]) - 2.0) < 1e-5, "SQRT[2]")
    assert_true(abs(Float32(out[3]) - 3.0) < 1e-5, "SQRT[3]")
    print("test_sqrt: PASSED")


# ===----------------------------------------------------------------------===#
# L1Loss
# ===----------------------------------------------------------------------===#


def test_l1loss() raises:
    comptime g = make_l1_graph(8, 4)
    var model = Model[g]()
    var sgd = optim.SGD[g](model.parameters, lr=0.01)

    var x = Tensor[f32](TensorShape(8, 4))
    var y = Tensor[f32](TensorShape(8, 4))
    fill(x, 1.0)
    fill(y, 2.0)

    var initial_loss: Float32 = model.forward(x, y)[0]
    model.backward()
    sgd.step()
    sgd.zero_grad()

    for _ in range(9):
        _ = model.forward(x, y)
        model.backward()
        sgd.step()
        sgd.zero_grad()

    var final_loss: Float32 = model.forward(x, y)[0]
    assert_true(
        final_loss < initial_loss,
        "L1Loss should decrease (got "
        + String(initial_loss)
        + " -> "
        + String(final_loss)
        + ")",
    )
    print(
        "test_l1loss: PASSED (loss "
        + String(initial_loss)
        + " -> "
        + String(final_loss)
        + ")"
    )


def main() raises:
    test_neg()
    test_abs()
    test_sqrt()
    test_l1loss()

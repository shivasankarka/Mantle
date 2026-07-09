"""Test BatchNorm2d forward/backward correctness."""
from std.testing import assert_true

from mantle import f32
from mantle.autograd import OP
from mantle.autograd.graph import Graph
from mantle.core.tensor import Tensor, TensorShape
import mantle.nn as nn
from mantle.nn.model import Model


def make_bn_2d(N: Int, C: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(N, C))
    var out = nn.BatchNorm2d(g, x)
    g.out(out)
    return g^


def make_bn_4d(N: Int, C: Int, H: Int, W: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(N, C, H, W))
    var out = nn.BatchNorm2d(g, x)
    g.out(out)
    return g^


def make_bn_loss(N: Int, C: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(N, C))
    var y_true = g.input(TensorShape(N, C))
    var bn = nn.BatchNorm2d(g, x)
    var loss = nn.MSELoss(g, bn, y_true)
    g.loss(loss)
    return g^


def main() raises:
    # --- test 1: 2D zero mean ---
    comptime g2d = make_bn_2d(4, 3)
    var model2d = Model[g2d](inference_only=True)


    var inp2d = Tensor[f32](TensorShape(4, 3))
    for n in range(4):
        for c in range(3):
            inp2d[n * 3 + c] = Float32(n + c * 10)

    var out2d = model2d.inference(inp2d)
    var result2d = out2d[0].copy()

    for c in range(3):
        var mean: Float32 = 0.0
        for n in range(4):
            mean += result2d[n * 3 + c]
        mean /= 4.0
        assert_true(mean * mean < 1e-5, "mean should be ~0 for channel " + String(c))

    print("test_batchnorm_2d_zero_mean: PASSED")

    # --- test 2: 4D shape ---
    comptime g4d = make_bn_4d(2, 3, 4, 4)
    var model4d = Model[g4d](inference_only=True)

    var inp4d = Tensor[f32](TensorShape(2, 3, 4, 4))
    for i in range(inp4d.num_elements()):
        inp4d[i] = Float32(i)

    var out4d = model4d.inference(inp4d)
    assert_true(
        out4d[0].copy().shape() == TensorShape(2, 3, 4, 4),
        "output shape mismatch",
    )
    print("test_batchnorm_4d_shape: PASSED")

    # --- test 3: backward produces scalar loss ---
    comptime gloss = make_bn_loss(4, 2)
    var model_loss = Model[gloss]()

    var inp_loss = Tensor[f32](TensorShape(4, 2))
    var target = Tensor[f32](TensorShape(4, 2))
    for i in range(inp_loss.num_elements()):
        inp_loss[i] = Float32(i + 1)

    var loss = model_loss.forward(inp_loss, target)
    model_loss.backward()
    assert_true(loss.num_elements() == 1, "loss should be scalar")
    print("test_batchnorm_backward: PASSED")

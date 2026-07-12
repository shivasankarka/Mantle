"""Test Softmax activation: outputs sum to 1, values in (0, 1), loss decreases."""
from std.testing import assert_true
from std.math import abs

from mantle import f32
from mantle.autograd.graph import Graph
from mantle.core.tensor import Tensor, TensorShape
from mantle.core.tensorutils import fill
import mantle.nn as nn
from mantle.nn.model import Model
import mantle.nn.optim as optim


def make_softmax_inference_graph(batch: Int, n_in: Int, n_cls: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch, n_in))
    var logits = nn.Linear(g, x, n_cls)
    var probs = nn.Softmax(g, logits, axis=1)
    g.out(probs)
    return g^


def make_softmax_train_graph(batch: Int, n_in: Int, n_cls: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch, n_in))
    var y_true = g.input(TensorShape(batch, n_cls))
    var logits = nn.Linear(g, x, n_cls)
    var loss = nn.CrossEntropyLoss(g, logits, y_true)
    g.loss(loss)
    return g^


def main() raises:
    # --- test 1: softmax outputs sum to 1, all values in (0, 1) ---
    comptime gi = make_softmax_inference_graph(4, 8, 3)
    var model_i = Model[gi](inference_only=True)

    var x = Tensor[f32](TensorShape(4, 8))
    fill(x, 1.0)

    var inf_out = model_i.inference(x)
    var probs = inf_out[0].copy()

    for i in range(4):
        var row_sum: Float32 = 0.0
        for j in range(3):
            row_sum += probs[i * 3 + j]
        assert_true(
            abs(row_sum - 1.0) < 1e-5,
            "Softmax row " + String(i) + " should sum to 1, got " + String(row_sum),
        )

    for i in range(4 * 3):
        assert_true(
            probs[i] > 0.0 and probs[i] < 1.0,
            "Softmax output should be in (0, 1)",
        )

    print("test_softmax_output: PASSED (rows sum to 1, values in (0,1))")

    # --- test 2: CrossEntropy loss decreases with SGD ---
    comptime gt = make_softmax_train_graph(4, 8, 3)
    var model_t = Model[gt]()
    var sgd = optim.SGD[gt](model_t.parameters, lr=0.05)

    var xt = Tensor[f32](TensorShape(4, 8))
    var yt = Tensor[f32](TensorShape(4, 3))
    fill(xt, 1.0)
    for i in range(4):
        yt[i * 3 + 0] = 1.0  # one-hot: class 0

    var initial_loss: Float32 = model_t.forward(xt, yt)[0]
    model_t.backward()
    sgd.step()
    sgd.zero_grad()

    for _ in range(9):
        _ = model_t.forward(xt, yt)
        model_t.backward()
        sgd.step()
        sgd.zero_grad()

    var final_loss: Float32 = model_t.forward(xt, yt)[0]
    assert_true(
        final_loss < initial_loss,
        "Softmax+CrossEntropy: loss should decrease (got "
        + String(initial_loss)
        + " -> "
        + String(final_loss)
        + ")",
    )
    print(
        "test_softmax_backward: PASSED (loss "
        + String(initial_loss)
        + " -> "
        + String(final_loss)
        + ")"
    )

"""Test SGD and SGD+Momentum optimizers."""
from std.testing import assert_true

from mantle import f32
from mantle.autograd.graph import Graph
from mantle.core.tensor import Tensor, TensorShape
from mantle.core.tensorutils import fill
import mantle.nn as nn
from mantle.nn.model import Model
import mantle.nn.optim as optim


def make_linear_graph(batch_size: Int, n_in: Int, n_out: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch_size, n_in))
    var y_true = g.input(TensorShape(batch_size, n_out))
    var y_pred = nn.Linear(g, x, n_out)
    g.out(y_pred)
    var loss = nn.MSELoss(g, y_pred, y_true)
    g.loss(loss)
    return g^


def main() raises:
    # --- SGD (no momentum): loss should decrease ---
    comptime g = make_linear_graph(8, 4, 1)

    var model = Model[g]()
    var sgd = optim.SGD[g](model.parameters, lr=0.01)

    var x = Tensor[f32](TensorShape(8, 4))
    var y = Tensor[f32](TensorShape(8, 1))
    fill(x, 1.0)
    fill(y, 2.0)

    var initial_loss: Float32 = model.forward(x, y)[0]
    model.backward()
    sgd.step()
    sgd.zero_grad()

    for _ in range(4):
        _ = model.forward(x, y)
        model.backward()
        sgd.step()
        sgd.zero_grad()

    var final_loss: Float32 = model.forward(x, y)[0]
    assert_true(
        final_loss < initial_loss,
        "SGD: loss should decrease after training steps",
    )
    print("test_sgd_no_momentum: PASSED (loss " + String(initial_loss) + " -> " + String(final_loss) + ")")

    # --- SGD + Momentum ---
    comptime gm = make_linear_graph(8, 4, 1)

    var model_m = Model[gm]()
    var sgd_m = optim.SGD[gm](model_m.parameters, lr=0.01, momentum=0.9)

    var xm = Tensor[f32](TensorShape(8, 4))
    var ym = Tensor[f32](TensorShape(8, 1))
    fill(xm, 1.0)
    fill(ym, 2.0)

    var initial_lm: Float32 = model_m.forward(xm, ym)[0]
    model_m.backward()
    sgd_m.step()
    sgd_m.zero_grad()

    for _ in range(4):
        _ = model_m.forward(xm, ym)
        model_m.backward()
        sgd_m.step()
        sgd_m.zero_grad()

    var final_lm: Float32 = model_m.forward(xm, ym)[0]
    assert_true(
        final_lm < initial_lm,
        "SGD+Momentum: loss should decrease after training steps",
    )
    print("test_sgd_momentum: PASSED (loss " + String(initial_lm) + " -> " + String(final_lm) + ")")

"""Test clip_grad_norm gradient clipping utility."""
from std.testing import assert_true
from std.math import sqrt

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
    # --- clip_grad_norm: norm should be <= max_norm after clipping ---
    comptime g = make_linear_graph(8, 4, 1)

    var model = Model[g]()

    var x = Tensor[f32](TensorShape(8, 4))
    var y = Tensor[f32](TensorShape(8, 1))
    fill(x, 1.0)
    fill(y, 100.0)  # large target => large gradients

    _ = model.forward(x, y)
    model.backward()

    # Compute the norm before clipping
    var norm_before = optim.clip_grad_norm[g](model.parameters, max_norm=1.0)

    # Verify the norm was reduced to at most max_norm (with a small tolerance
    # for the epsilon in the denominator)
    var norm_after = optim.clip_grad_norm[g](model.parameters, max_norm=1.0)
    assert_true(
        Float32(norm_after) <= 1.01,
        "clip_grad_norm: post-clip norm should be <= max_norm",
    )
    print(
        "test_clip_grad_norm: PASSED (norm "
        + String(norm_before)
        + " -> "
        + String(norm_after)
        + ")"
    )

    # --- When norm < max_norm, gradients are untouched (norm unchanged) ---
    comptime g2 = make_linear_graph(2, 2, 1)

    var model2 = Model[g2]()

    var x2 = Tensor[f32](TensorShape(2, 2))
    var y2 = Tensor[f32](TensorShape(2, 1))
    fill(x2, 0.001)
    fill(y2, 0.001)

    _ = model2.forward(x2, y2)
    model2.backward()

    var small_norm = optim.clip_grad_norm[g2](model2.parameters, max_norm=1000.0)
    # With max_norm much larger than actual norm, clipping is a no-op.
    # A second call should return the same (already-small) norm.
    var small_norm2 = optim.clip_grad_norm[g2](model2.parameters, max_norm=1000.0)
    assert_true(
        abs(Float32(small_norm) - Float32(small_norm2)) < 1e-4,
        "clip_grad_norm: below-threshold norm should be unchanged",
    )
    print(
        "test_clip_grad_no_clip: PASSED (norm stable at "
        + String(small_norm)
        + ")"
    )

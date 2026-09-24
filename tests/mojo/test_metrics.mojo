"""Tests for host-side classification metrics."""
from std.math import abs
from std.testing import assert_true

from mantle import f32
from mantle.core.tensor import Tensor, TensorShape
from mantle.nn.metrics import accuracy


def test_accuracy_one_hot() raises:
    var logits = Tensor[f32](TensorShape(3, 3))
    var targets = Tensor[f32](TensorShape(3, 3))

    # Predictions: 1, 0, 2. Targets: 1, 2, 2 => 2 / 3 correct.
    logits[1] = 1.0
    logits[3] = 1.0
    logits[8] = 1.0
    targets[1] = 1.0
    targets[5] = 1.0
    targets[8] = 1.0

    assert_true(
        abs(accuracy(logits, targets) - (2.0 / 3.0)) < 1e-6,
        "accuracy should compare per-row argmax values",
    )
    print("test_accuracy_one_hot: PASSED")


def main() raises:
    test_accuracy_one_hot()

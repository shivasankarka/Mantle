"""Regression tests for DataLoader batch boundaries."""
from std.testing import assert_true

from mantle import Tensor, TensorShape, f32
from mantle.data import DataLoader


def test_drop_last_default() raises:
    var data = Tensor[f32](TensorShape(5, 1))
    var labels = Tensor[f32](TensorShape(5, 1))
    var loader = DataLoader(data, labels, batch_size=2)
    assert_true(
        loader.__len__() == 2,
        "drop_last should be the fixed-shape default",
    )
    print("test_drop_last_default: PASSED")


def test_keeps_final_partial_batch() raises:
    var data = Tensor[f32](TensorShape(5, 1))
    var labels = Tensor[f32](TensorShape(5, 1))
    var loader = DataLoader(data, labels, batch_size=2, drop_last=False)
    var rows = 0
    for batch in loader:
        rows += batch.data.dim(0)
    assert_true(rows == 5, "drop_last=False should yield every sample")
    print("test_keeps_final_partial_batch: PASSED")


def main() raises:
    test_drop_last_default()
    test_keeps_final_partial_batch()

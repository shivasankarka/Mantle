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


def test_shuffle_keeps_data_and_labels_aligned() raises:
    var data = Tensor[f32](TensorShape(8, 1))
    var labels = Tensor[f32](TensorShape(8, 1))
    for i in range(8):
        data[i] = Float32(i)
        labels[i] = Float32(i)
    var loader = DataLoader(data, labels, batch_size=2, shuffle=True)
    var seen = 0
    for batch in loader:
        for i in range(batch.data.num_elements()):
            assert_true(
                batch.data[i] == batch.labels[i],
                "shuffling must keep data and labels aligned",
            )
            seen += 1
    assert_true(seen == 8, "shuffled loader should yield every sample")
    print("test_shuffle_keeps_data_and_labels_aligned: PASSED")


def test_epoch_iterator_shares_dataset_storage() raises:
    var data = Tensor[f32](TensorShape(2, 1))
    var labels = Tensor[f32](TensorShape(2, 1))
    var loader = DataLoader(data, labels, batch_size=2)
    var iterator = loader.__iter__()
    loader.data[0] = 7.0
    var batch = iterator.__next__()
    assert_true(
        batch.data[0] == 7.0,
        "epoch iterator should share the loader's dataset storage",
    )
    print("test_epoch_iterator_shares_dataset_storage: PASSED")


def main() raises:
    test_drop_last_default()
    test_keeps_final_partial_batch()
    test_shuffle_keeps_data_and_labels_aligned()
    test_epoch_iterator_shares_dataset_storage()

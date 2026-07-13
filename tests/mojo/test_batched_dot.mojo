"""Test batched (rank >= 3) OP.DOT forward/backward."""
from std.testing import assert_true

from mantle import f32
from mantle.autograd import OP
from mantle.core.tensorutils import fill
from mantle.nn import Tensor, TensorShape

from tests import test_binary_op, test_binary_op_backward


# ===----------------------------------------------------------------------===#
# (B, T, D) @ (D, K) -- Linear-on-batched-input case, t2 shared across batch
# ===----------------------------------------------------------------------===#


def test_batched_dot_shared_t2() raises:
    comptime t1_shape = TensorShape(2, 3, 4)  # (B=2, T=3, D=4)
    comptime t2_shape = TensorShape(4, 5)  # (D=4, K=5)
    var t1 = Tensor[f32](t1_shape)
    var t2 = Tensor[f32](t2_shape)
    fill(t1, 1.0)
    fill(t2, 2.0)

    # Each output element = sum over D=4 of 1*2 = 8
    var expected = Tensor[f32](2, 3, 5)
    fill(expected, 8.0)

    test_binary_op[OP.DOT, t1_shape, t2_shape](t1, t2, expected)
    print("test_batched_dot_shared_t2: PASSED")


def test_batched_dot_shared_t2_backward() raises:
    comptime t1_shape = TensorShape(2, 3, 4)
    comptime t2_shape = TensorShape(4, 5)
    comptime ug_shape = TensorShape(2, 3, 5)

    var t1 = Tensor[f32](t1_shape)
    var t2 = Tensor[f32](t2_shape)
    var ug = Tensor[f32](ug_shape)
    fill(t1, 1.0)
    fill(t2, 2.0)
    fill(ug, 1.0)

    # d/dt1 = ug @ t2^T -> each element = sum over K=5 of 1*2 = 10
    var grad_1_expected = Tensor[f32](2, 3, 4)
    fill(grad_1_expected, 10.0)

    # d/dt2 = sum over batch,T of t1^T @ ug -> each element = (2*3) * 1*1 = 6
    var grad_2_expected = Tensor[f32](4, 5)
    fill(grad_2_expected, 6.0)

    test_binary_op_backward[OP.DOT, t1_shape, t2_shape, ug_shape](
        t1, t2, ug, grad_1_expected, grad_2_expected
    )
    print("test_batched_dot_shared_t2_backward: PASSED")


# ===----------------------------------------------------------------------===#
# (B, H, T, d) @ (B, H, d, T) -- matching batch dims, attention-score case
# ===----------------------------------------------------------------------===#


def test_batched_dot_matching_batch() raises:
    comptime t1_shape = TensorShape(2, 2, 3, 4)  # (B=2, H=2, T=3, d=4)
    comptime t2_shape = TensorShape(2, 2, 4, 3)  # (B=2, H=2, d=4, T=3)
    var t1 = Tensor[f32](t1_shape)
    var t2 = Tensor[f32](t2_shape)
    fill(t1, 1.0)
    fill(t2, 2.0)

    var expected = Tensor[f32](2, 2, 3, 3)
    fill(expected, 8.0)  # sum over d=4 of 1*2

    test_binary_op[OP.DOT, t1_shape, t2_shape](t1, t2, expected)
    print("test_batched_dot_matching_batch: PASSED")


def test_batched_dot_matching_batch_backward() raises:
    comptime t1_shape = TensorShape(2, 2, 3, 4)
    comptime t2_shape = TensorShape(2, 2, 4, 3)
    comptime ug_shape = TensorShape(2, 2, 3, 3)

    var t1 = Tensor[f32](t1_shape)
    var t2 = Tensor[f32](t2_shape)
    var ug = Tensor[f32](ug_shape)
    fill(t1, 1.0)
    fill(t2, 2.0)
    fill(ug, 1.0)

    # d/dt1 = ug @ t2^T per batch -> sum over T=3 of 1*2 = 6
    var grad_1_expected = Tensor[f32](2, 2, 3, 4)
    fill(grad_1_expected, 6.0)

    # d/dt2 = t1^T @ ug per batch -> sum over T=3 of 1*1 = 3
    var grad_2_expected = Tensor[f32](2, 2, 4, 3)
    fill(grad_2_expected, 3.0)

    test_binary_op_backward[OP.DOT, t1_shape, t2_shape, ug_shape](
        t1, t2, ug, grad_1_expected, grad_2_expected
    )
    print("test_batched_dot_matching_batch_backward: PASSED")


def main() raises:
    test_batched_dot_shared_t2()
    test_batched_dot_shared_t2_backward()
    test_batched_dot_matching_batch()
    test_batched_dot_matching_batch_backward()

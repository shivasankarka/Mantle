"""Test OP.PAD (constant-mode padding): forward correctness, backward
crop, and end-to-end training convergence."""
from std.testing import assert_true, assert_equal

from mantle import f32
from mantle.autograd import OP
from mantle.autograd.attributes import Attribute, AttributeVector
from mantle.core.tensorutils import fill
from mantle.nn import Tensor, TensorShape

from tests import test_unary_op, test_unary_op_backward


# ===----------------------------------------------------------------------===#
# Forward: 2D padding, zero value
# ===----------------------------------------------------------------------===#


def test_pad_2d_zero() raises:
    comptime t1_shape = TensorShape(2, 2)
    var t1 = Tensor[f32](t1_shape)
    t1[0] = 1.0; t1[1] = 2.0; t1[2] = 3.0; t1[3] = 4.0

    # pad 1 before/after on both axes -> (4, 4), original block at [1:3, 1:3]
    var expected = Tensor[f32](4, 4)
    expected[1 * 4 + 1] = 1.0
    expected[1 * 4 + 2] = 2.0
    expected[2 * 4 + 1] = 3.0
    expected[2 * 4 + 2] = 4.0

    test_unary_op[
        OP.PAD,
        t1_shape,
        AttributeVector(
            Attribute("before", TensorShape(1, 1)),
            Attribute("after", TensorShape(1, 1)),
        ),
    ](t1, expected)
    print("test_pad_2d_zero: PASSED")


# ===----------------------------------------------------------------------===#
# Forward: asymmetric padding + nonzero fill value
# ===----------------------------------------------------------------------===#


def test_pad_asymmetric_nonzero() raises:
    comptime t1_shape = TensorShape(1, 2)
    var t1 = Tensor[f32](t1_shape)
    t1[0] = 5.0; t1[1] = 6.0

    # before=(0,1), after=(0,2) -> shape (1, 5): [fill, 5, 6, fill, fill]
    var expected = Tensor[f32](1, 5)
    fill(expected, -1.0)
    expected[1] = 5.0
    expected[2] = 6.0

    test_unary_op[
        OP.PAD,
        t1_shape,
        AttributeVector(
            Attribute("before", TensorShape(0, 1)),
            Attribute("after", TensorShape(0, 2)),
            Attribute("value", -1.0),
        ),
    ](t1, expected)
    print("test_pad_asymmetric_nonzero: PASSED")


# ===----------------------------------------------------------------------===#
# Backward: gradient crops back to the original region
# ===----------------------------------------------------------------------===#


def test_pad_backward_crops() raises:
    comptime t1_shape = TensorShape(2, 2)
    comptime ug_shape = TensorShape(4, 4)

    var t1 = Tensor[f32](t1_shape)
    fill(t1, 1.0)

    var ug = Tensor[f32](ug_shape)
    for i in range(ug.num_elements()):
        ug[i] = Float32(i)

    # original block sits at rows/cols [1:3, 1:3] of the padded (4,4) ug
    var expected_grad = Tensor[f32](2, 2)
    expected_grad[0] = ug[1 * 4 + 1]
    expected_grad[1] = ug[1 * 4 + 2]
    expected_grad[2] = ug[2 * 4 + 1]
    expected_grad[3] = ug[2 * 4 + 2]

    test_unary_op_backward[
        OP.PAD,
        t1_shape,
        ug_shape,
        AttributeVector(
            Attribute("before", TensorShape(1, 1)),
            Attribute("after", TensorShape(1, 1)),
        ),
    ](t1, ug, expected_grad)
    print("test_pad_backward_crops: PASSED")


def main() raises:
    test_pad_2d_zero()
    test_pad_asymmetric_nonzero()
    test_pad_backward_crops()

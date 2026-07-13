"""Test OP.GELU forward/backward against known reference values."""
from std.testing import assert_true
from std.math import abs

from mantle import f32
from mantle.autograd import OP
from mantle.core.tensor import Tensor, TensorShape

from tests import test_unary_op, test_unary_op_backward


def test_gelu_forward() raises:
    comptime t1_shape = TensorShape(4)
    var t1 = Tensor[f32](t1_shape)
    t1[0] = 0.0
    t1[1] = 1.0
    t1[2] = -1.0
    t1[3] = 2.0

    # Reference values (tanh approximation of GELU):
    # gelu(0) = 0
    # gelu(1) ~= 0.84119
    # gelu(-1) ~= -0.15881
    # gelu(2) ~= 1.9546
    var expected = Tensor[f32](4)
    expected[0] = 0.0
    expected[1] = 0.8411920
    expected[2] = -0.1588080
    expected[3] = 1.9545977

    test_unary_op[OP.GELU, t1_shape](t1, expected)
    print("test_gelu_forward: PASSED")


def test_gelu_backward() raises:
    comptime t1_shape = TensorShape(3)
    comptime ug_shape = TensorShape(3)

    var t1 = Tensor[f32](t1_shape)
    t1[0] = 0.0
    t1[1] = 1.0
    t1[2] = -1.0

    var ug = Tensor[f32](ug_shape)
    ug[0] = 1.0
    ug[1] = 1.0
    ug[2] = 1.0

    # d/dx gelu(0) = 0.5
    # d/dx gelu(1) ~= 1.08296
    # d/dx gelu(-1) ~= -0.08296
    var grad_expected = Tensor[f32](3)
    grad_expected[0] = 0.5
    grad_expected[1] = 1.0829641
    grad_expected[2] = -0.0829641

    test_unary_op_backward[OP.GELU, t1_shape, ug_shape](
        t1, ug, grad_expected
    )
    print("test_gelu_backward: PASSED")


def main() raises:
    test_gelu_forward()
    test_gelu_backward()

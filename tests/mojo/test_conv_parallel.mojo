"""Regression test for parallel Conv2D input-gradient tiling."""
from std.testing import assert_true
from std.utils.index import IndexList

from mantle import f32
from mantle.autograd.attributes import Attribute, AttributeVector
from mantle.autograd.ops.conv import CONV2D
from mantle.core.tensor import Tensor, TensorShape
from mantle.core.tensorutils import fill


def test_input_gradient_parallel_tiles() raises:
    # Two images and three input channels ensure the kernel creates more than
    # one task in both dimensions. With 1x1 all-one filters and upper
    # gradients, every input value receives one contribution per output
    # channel, i.e. exactly 2.0.
    comptime input_shape = TensorShape(2, 3, 3, 3)
    comptime kernel_shape = TensorShape(2, 3, 1, 1)
    comptime bias_shape = TensorShape(2)
    comptime upper_grad_shape = TensorShape(2, 2, 3, 3)
    comptime attributes = AttributeVector(
        Attribute("padding", IndexList[2](0, 0)),
        Attribute("stride", IndexList[2](1, 1)),
        Attribute("dilation", IndexList[2](1, 1)),
    )

    var inputs = Tensor[f32](input_shape)
    var kernel = Tensor[f32](kernel_shape)
    var bias = Tensor[f32](bias_shape)
    var upper_grad = Tensor[f32](upper_grad_shape)
    fill(kernel, 1.0)
    fill(upper_grad, 1.0)

    var output = Tensor[f32](upper_grad_shape)
    fill(inputs, 1.0)
    fill(bias, 1.0)
    CONV2D.forward[input_shape, kernel_shape, bias_shape, attributes](
        output, inputs, kernel, bias
    )
    for i in range(output.num_elements()):
        assert_true(
            output[i] == 4.0,
            "forward result should be the three-channel sum plus bias",
        )

    var input_grad = CONV2D.backward[
        0,
        upper_grad_shape,
        input_shape,
        kernel_shape,
        bias_shape,
        attributes,
    ](upper_grad, inputs, kernel, bias)

    for i in range(input_grad.num_elements()):
        assert_true(
            input_grad[i] == 2.0,
            "input gradient should include both output channels",
        )

    var kernel_grad = CONV2D.backward[
        1,
        upper_grad_shape,
        input_shape,
        kernel_shape,
        bias_shape,
        attributes,
    ](upper_grad, inputs, kernel, bias)
    for i in range(kernel_grad.num_elements()):
        assert_true(
            kernel_grad[i] == 18.0,
            "filter gradient should sum every batch and spatial position",
        )
    print("test_input_gradient_parallel_tiles: PASSED")


def main() raises:
    test_input_gradient_parallel_tiles()

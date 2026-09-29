# ===----------------------------------------------------------------------=== #
# Mantle: Transposed Convolution Ops
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""ConvTranspose Ops (mantle.autograd.ops.conv_transpose)
------------------------------------------------
2D transposed convolution (CPU only).
"""
from mantle import f32
from mantle.core.tensor import Tensor, TensorShape
from mantle.autograd.attributes import AttributeVector
from mantle.autograd.ops.conv import CONV2D

from max.algorithm import parallelize
from std.utils.index import IndexList


@always_inline
def get_transpose_result_shape(
    input_shape: TensorShape,
    kernel_shape: TensorShape,
    padding: IndexList[2],
    stride: IndexList[2],
    dilation: IndexList[2],
    output_padding: IndexList[2],
) -> IndexList[2]:
    """Spatial size a transposed convolution produces from `input_shape`."""
    var result_x_dim = (
        (input_shape[-2] - 1) * stride[0]
        - 2 * padding[0]
        + dilation[0] * (kernel_shape[-2] - 1)
        + output_padding[0]
        + 1
    )
    var result_y_dim = (
        (input_shape[-1] - 1) * stride[1]
        - 2 * padding[1]
        + dilation[1] * (kernel_shape[-1] - 1)
        + output_padding[1]
        + 1
    )
    return IndexList[2](result_x_dim, result_y_dim)


struct CONVTRANSPOSE2D:
    """2D transposed convolution.

    Notes:
        A transposed convolution's forward pass is the same scatter
        (col2im) computation as `CONV2D`'s input gradient, so this delegates
        to `CONV2D.forward`/`CONV2D.backward` with input/output roles
        swapped instead of duplicating the im2col/col2im kernels.
        `output_padding` must satisfy `0 <= output_padding < stride` on each
        axis (same constraint as PyTorch's `ConvTranspose2d`), or the
        result shape this computes won't round-trip through `CONV2D`'s own
        forward formula.
    """

    @staticmethod
    def result_shape(
        input_shape: TensorShape,
        kernel_shape: TensorShape,
        bias_shape: TensorShape,
        attributes: AttributeVector,
    ) -> TensorShape:
        # inputs.shape [batch, in_channels, iX, iY]
        # kernel.shape [in_channels, out_channels, kX, kY]
        # output.shape [batch, out_channels, oX, oY]
        var padding = attributes["padding"].value().to_static[2]()
        var stride = attributes["stride"].value().to_static[2]()
        var dilation = attributes["dilation"].value().to_static[2]()
        var output_padding = attributes["output_padding"].value().to_static[
            2
        ]()
        var res = get_transpose_result_shape(
            input_shape,
            kernel_shape,
            padding,
            stride,
            dilation,
            output_padding,
        )
        return TensorShape(input_shape[0], kernel_shape[1], res[0], res[1])

    @staticmethod
    def forward[
        input_shape: TensorShape,
        kernel_shape: TensorShape,
        bias_shape: TensorShape,
        attributes: AttributeVector,
    ](
        mut outputs: Tensor[f32],
        inputs: Tensor[f32],
        kernel: Tensor[f32],
        bias: Tensor[f32],
    ) raises:
        comptime output_shape = Self.result_shape(
            input_shape, kernel_shape, bias_shape, attributes
        )
        var unbiased = CONV2D.backward[
            0, input_shape, output_shape, kernel_shape, bias_shape, attributes
        ](inputs, bias, kernel, bias)

        comptime out_channels = kernel_shape[1]
        comptime spatial = output_shape[2] * output_shape[3]
        comptime out_strides = output_shape.strides()

        def add_bias(
            work_item: Int,
        ) {mut outputs, imm unbiased, imm bias}:
            var batch = work_item // out_channels
            var out_ch = work_item % out_channels
            var channel_offset = (
                batch * out_strides[0] + out_ch * out_strides[1]
            )
            for position in range(spatial):
                outputs[channel_offset + position] = (
                    unbiased[channel_offset + position] + bias[out_ch]
                )

        parallelize(add_bias, output_shape[0] * out_channels)

    @staticmethod
    def backward[
        tensor_id: Int,
        ug_shape: TensorShape,
        input_shape: TensorShape,
        kernel_shape: TensorShape,
        bias_shape: TensorShape,
        attributes: AttributeVector,
    ](
        ug: Tensor[f32],
        inputs: Tensor[f32],
        kernel: Tensor[f32],
        bias: Tensor[f32],
    ) raises -> Tensor[f32]:
        comptime if tensor_id == 0:
            var res = Tensor[f32](input_shape, uninitialized=True)
            comptime zero_bias_shape = TensorShape(kernel_shape[0])
            var zero_bias = Tensor[f32](zero_bias_shape)
            CONV2D.forward[
                ug_shape, kernel_shape, zero_bias_shape, attributes
            ](res, ug, kernel, zero_bias)
            return res^
        elif tensor_id == 1:
            return CONV2D.backward[
                1, input_shape, ug_shape, kernel_shape, bias_shape, attributes
            ](inputs, ug, kernel, bias)
        else:
            comptime assert tensor_id == 2
            return CONV2D.backward[
                2, ug_shape, input_shape, kernel_shape, bias_shape, attributes
            ](ug, inputs, kernel, bias)

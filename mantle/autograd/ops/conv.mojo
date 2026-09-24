# ===----------------------------------------------------------------------=== #
# Mantle: Convolution Ops
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Conv Ops (mantle.autograd.ops.conv)
------------------------------------------------
im2col-based 2D convolution with forward and backward passes.
"""
from mantle import f32, nelts
from mantle.core.tensor import Tensor, TensorShape
from mantle.autograd.attributes import AttributeVector
from mantle.autograd.ops.matmul import dot, dot_transpose_t2

from std.algorithm import vectorize
from max.algorithm import parallelize
from std.utils.index import IndexList
from std.memory import unsafe_memset_zero, Pointer
from std.memory.alloc import unsafe_alloc


# ===----------------------------------------------------------------------===#
# Shape Helpers
# ===----------------------------------------------------------------------===#


@always_inline
def get_result_shape(
    input_shape: TensorShape,
    kernel_shape: TensorShape,
    padding: IndexList[2],
    stride: IndexList[2],
    dilation: IndexList[2],
) -> IndexList[2]:
    """
    Calculates the X and Y dimensions of the resulting convolution.
    Dimensions X, Y are on the end of the shape (..., X, Y)
        dimension X on index -2.
        dimension Y on index -1.
    """

    var result_x_dim = (
        (
            input_shape[-2]
            + (2 * padding[0])
            - dilation[0] * (kernel_shape[-2] - 1)
            - 1
        )
        // stride[0]
    ) + 1
    var result_y_dim = (
        (
            input_shape[-1]
            + (2 * padding[1])
            - dilation[1] * (kernel_shape[-1] - 1)
            - 1
        )
        // stride[1]
    ) + 1

    return IndexList[2](result_x_dim, result_y_dim)


struct CONV2D:
    @staticmethod
    def result_shape(
        input_shape: TensorShape,
        kernel_shape: TensorShape,
        bias_shape: TensorShape,
        attributes: AttributeVector,
    ) -> TensorShape:
        # Output shape = [batch, out_channels, oX, oY]

        var padding = attributes["padding"].value().to_static[2]()
        var stride = attributes["stride"].value().to_static[2]()
        var dilation = attributes["dilation"].value().to_static[2]()
        var res = get_result_shape(
            input_shape, kernel_shape, padding, stride, dilation
        )

        return TensorShape(input_shape[0], kernel_shape[0], res[0], res[1])

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
        """
        Performs a 2D convolution on the input tensor using the kernel and bias.
            inputs.shape     [batch, in_channels, iX, iY]
            kernel.shape     [out_channels, in_channels, kX, kY] (or weights)
            bias.shape       [out_channels].
            output.shape     [batch, out_channels, oX, oY].
        """
        comptime padding = attributes["padding"].value().to_static[2]()
        comptime stride = attributes["stride"].value().to_static[2]()
        comptime dilation = attributes["dilation"].value().to_static[2]()

        comptime padding_x = padding[0]
        comptime padding_y = padding[1]
        comptime stride_x = stride[0]
        comptime stride_y = stride[1]
        comptime dilation_x = dilation[0]
        comptime dilation_y = dilation[1]

        comptime batch_size = input_shape[0]
        comptime in_channels = input_shape[1]
        comptime in_x = input_shape[2]
        comptime in_y = input_shape[3]
        comptime out_channels = kernel_shape[0]
        comptime k_x = kernel_shape[2]
        comptime k_y = kernel_shape[3]
        comptime output_shape = Self.result_shape(
            input_shape, kernel_shape, bias_shape, attributes
        )
        comptime out_x = output_shape[2]
        comptime out_y = output_shape[3]
        comptime col_x = out_x
        comptime col_y = out_y
        comptime col_shape = TensorShape(
            batch_size, col_x * col_y, in_channels * k_x * k_y
        )  # [batch, colX * colY, in_channels * kX * kY]
        comptime col_shape_stripped = TensorShape(
            in_channels * k_x * k_y, col_x, col_y
        )

        comptime inputs_strides = input_shape.strides()
        comptime kernel_strides = kernel_shape.strides()
        comptime outputs_strides = output_shape.strides()
        comptime col_strides = col_shape.strides()

        var col_ptr = unsafe_alloc[Scalar[f32]](col_shape.num_elements())
        unsafe_memset_zero(col_ptr, col_shape.num_elements())

        def im2col(batch: Int) {imm col_ptr, imm inputs}:
            for ux in range(out_x):
                for uy in range(out_y):
                    for in_ch in range(in_channels):
                        for kx in range(k_x):
                            for ky in range(k_y):
                                var ix = (
                                    ux * stride_x - padding_x + kx * dilation_x
                                )
                                var iy = (
                                    uy * stride_y - padding_y + ky * dilation_y
                                )

                                if ix < 0 or iy < 0 or ix >= in_x or iy >= in_y:
                                    continue

                                var col_index = (
                                    batch * col_strides[0]
                                    + (ux * col_y + uy) * col_strides[1]
                                    + (in_ch * k_x * k_y + kx * k_y + ky)
                                )

                                var input_index = (
                                    batch * inputs_strides[0]
                                    + in_ch * inputs_strides[1]
                                    + ix * inputs_strides[2]
                                    + iy
                                )

                                col_ptr[unsafe_offset=col_index] = inputs[
                                    input_index
                                ]

        parallelize(im2col, batch_size)

        # im2col is `(spatial_positions, kernel_values)` while each filter is
        # `(kernel_values)`. Route the product through the same
        # Accelerate-backed SGEMM path as Linear, then transpose the small
        # spatial-major result into Mantle's public NCHW layout.
        comptime gemm_shape = TensorShape(col_x * col_y, out_channels)
        var gemm_ptr = unsafe_alloc[Scalar[f32]](
            batch_size * gemm_shape.num_elements()
        )
        for batch in range(batch_size):
            dot_transpose_t2[
                TensorShape(col_x * col_y, in_channels * k_x * k_y),
                TensorShape(out_channels, in_channels * k_x * k_y),
            ](
                gemm_ptr.unsafe_offset(batch * gemm_shape.num_elements()),
                col_ptr.unsafe_offset(batch * col_strides[0]),
                kernel.ptr(),
            )

        def store_output(
            work_item: Int,
        ) {mut outputs, imm gemm_ptr, imm bias}:
            var batch = work_item // out_channels
            var out_ch = work_item % out_channels
            for position in range(col_x * col_y):
                var output_index = (
                    batch * outputs_strides[0]
                    + out_ch * outputs_strides[1]
                    + position
                )
                var gemm_index = (
                    batch * gemm_shape.num_elements()
                    + position * out_channels
                    + out_ch
                )
                outputs[output_index] = (
                    gemm_ptr[unsafe_offset=gemm_index] + bias[out_ch]
                )

        parallelize(store_output, batch_size * out_channels)

        gemm_ptr.unsafe_free()
        col_ptr.unsafe_free()

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
        """
        Backward operation of 2D convolution.

        Upper gradient of shape: [batch, out_channels, uX, uY].
        """

        comptime padding = attributes["padding"].value().to_static[2]()
        comptime stride = attributes["stride"].value().to_static[2]()
        comptime dilation = attributes["dilation"].value().to_static[2]()
        comptime padding_0 = padding[0]
        comptime padding_1 = padding[1]
        comptime stride_0 = stride[0]
        comptime stride_1 = stride[1]
        comptime dilation_0 = dilation[0]
        comptime dilation_1 = dilation[1]

        comptime inputs_strides = input_shape.strides()
        comptime kernel_strides = kernel_shape.strides()
        comptime ug_strides = ug_shape.strides()
        comptime inputs_strides_0 = inputs_strides[0]
        comptime inputs_strides_1 = inputs_strides[1]
        comptime inputs_strides_2 = inputs_strides[2]
        comptime kernel_strides_0 = kernel_strides[0]
        comptime kernel_strides_1 = kernel_strides[1]
        comptime kernel_strides_2 = kernel_strides[2]
        comptime ug_strides_0 = ug_strides[0]
        comptime ug_strides_1 = ug_strides[1]
        comptime ug_strides_2 = ug_strides[2]

        comptime input_shape_0 = input_shape[0]
        comptime input_shape_1 = input_shape[1]
        comptime input_shape_2 = input_shape[2]
        comptime input_shape_3 = input_shape[3]
        comptime kernel_shape_2 = kernel_shape[2]
        comptime kernel_shape_3 = kernel_shape[3]
        comptime ug_shape_0 = ug_shape[0]
        comptime ug_shape_1 = ug_shape[1]
        comptime ug_shape_2 = ug_shape[2]
        comptime ug_shape_3 = ug_shape[3]

        var res: Tensor[f32]

        comptime if tensor_id == 0:
            # Input gradient is `col_grad = dY^T @ W`, followed by col2im.
            # Materialize the small NCHW -> spatial-major transpose per batch
            # so the product uses the Accelerate SGEMM path.
            res = Tensor[f32](input_shape)
            comptime positions = ug_shape_2 * ug_shape_3
            comptime kernel_values = (
                input_shape_1 * kernel_shape_2 * kernel_shape_3
            )
            comptime col_grad_shape = TensorShape(positions, kernel_values)
            var col_grad_ptr = unsafe_alloc[Scalar[f32]](
                input_shape_0 * col_grad_shape.num_elements()
            )

            for batch in range(input_shape_0):
                var upper_grad_transposed = Tensor[f32](
                    TensorShape(positions, ug_shape_1), uninitialized=True
                )
                for out_ch in range(ug_shape_1):
                    for position in range(positions):
                        upper_grad_transposed[
                            position * ug_shape_1 + out_ch
                        ] = ug[
                            batch * ug_strides_0
                            + out_ch * ug_strides_1
                            + position
                        ]
                dot[
                    TensorShape(positions, ug_shape_1),
                    TensorShape(ug_shape_1, kernel_values),
                ](
                    col_grad_ptr.unsafe_offset(
                        batch * col_grad_shape.num_elements()
                    ),
                    upper_grad_transposed.ptr(),
                    kernel.ptr(),
                )

            # A separate (batch, input-channel) tile owns every element it
            # writes. This makes the col2im accumulation race-free while
            # exposing all input channels to the CPU scheduler.
            def col2im(work_item: Int) {mut res, imm col_grad_ptr}:
                var batch = work_item // input_shape_1
                var in_ch = work_item % input_shape_1
                for ux in range(ug_shape_2):
                    for uy in range(ug_shape_3):
                        for kx in range(kernel_shape_2):
                            for ky in range(kernel_shape_3):
                                var ix = (
                                    ux * stride_0 - padding_0 + kx * dilation_0
                                )
                                var iy = (
                                    uy * stride_1 - padding_1 + ky * dilation_1
                                )
                                if (
                                    ix < 0
                                    or iy < 0
                                    or ix >= input_shape_2
                                    or iy >= input_shape_3
                                ):
                                    continue
                                var input_index = (
                                    batch * inputs_strides_0
                                    + in_ch * inputs_strides_1
                                    + ix * inputs_strides_2
                                    + iy
                                )
                                var col_index = (
                                    batch * col_grad_shape.num_elements()
                                    + (ux * ug_shape_3 + uy) * kernel_values
                                    + in_ch * kernel_shape_2 * kernel_shape_3
                                    + kx * kernel_shape_3
                                    + ky
                                )
                                res[input_index] += col_grad_ptr[
                                    unsafe_offset=col_index
                                ]

            parallelize(col2im, input_shape_0 * input_shape_1)
            col_grad_ptr.unsafe_free()

        elif tensor_id == 1:
            # Filter gradient is the batched matrix product
            #   dW = sum_b dY[b] @ im2col(X[b]).
            # The old scalar-loop implementation performed this product one
            # filter weight at a time; use Accelerate SGEMM as Linear does.
            res = Tensor[f32](kernel_shape)
            comptime positions = ug_shape_2 * ug_shape_3
            comptime kernel_values = (
                input_shape_1 * kernel_shape_2 * kernel_shape_3
            )
            comptime col_shape = TensorShape(positions, kernel_values)
            var col_ptr = unsafe_alloc[Scalar[f32]](
                input_shape_0 * col_shape.num_elements()
            )
            unsafe_memset_zero(
                col_ptr, input_shape_0 * col_shape.num_elements()
            )

            def im2col(batch: Int) {imm col_ptr, imm inputs}:
                for ux in range(ug_shape_2):
                    for uy in range(ug_shape_3):
                        for in_ch in range(input_shape_1):
                            for kx in range(kernel_shape_2):
                                for ky in range(kernel_shape_3):
                                    var ix = (
                                        ux * stride_0
                                        - padding_0
                                        + kx * dilation_0
                                    )
                                    var iy = (
                                        uy * stride_1
                                        - padding_1
                                        + ky * dilation_1
                                    )
                                    if (
                                        ix < 0
                                        or iy < 0
                                        or ix >= input_shape_2
                                        or iy >= input_shape_3
                                    ):
                                        continue
                                    var col_index = (
                                        batch * col_shape.num_elements()
                                        + (ux * ug_shape_3 + uy) * kernel_values
                                        + in_ch
                                        * kernel_shape_2
                                        * kernel_shape_3
                                        + kx * kernel_shape_3
                                        + ky
                                    )
                                    var input_index = (
                                        batch * inputs_strides_0
                                        + in_ch * inputs_strides_1
                                        + ix * inputs_strides_2
                                        + iy
                                    )
                                    col_ptr[unsafe_offset=col_index] = inputs[
                                        input_index
                                    ]

            parallelize(im2col, input_shape_0)
            for batch in range(input_shape_0):
                var batch_kernel_grad = Tensor[f32](
                    kernel_shape, uninitialized=True
                )
                dot[
                    TensorShape(ug_shape_1, positions),
                    TensorShape(positions, kernel_values),
                ](
                    batch_kernel_grad.ptr(),
                    ug.ptr().unsafe_offset(batch * ug_strides_0),
                    col_ptr.unsafe_offset(batch * col_shape.num_elements()),
                )
                for i in range(kernel_shape.num_elements()):
                    res[i] += batch_kernel_grad[i]

            col_ptr.unsafe_free()

        else:
            # Bias
            # Sum of upper gradient over batch and X, Y dimensions
            # out_channels == ug_shape[1] == bias_shape[0]
            res = Tensor[f32](bias_shape)

            # Psuedocode
            # For every channel in the bias tensor,
            # Iterate over the upper gradient across the batch
            # For each batch, sum the upper gradient across X, Y dimensions
            # Add the sum to the bias tensor

            def bias_grad(out_ch: Int) {mut res, imm ug}:
                var channel_offset = out_ch * ug_strides_1
                var sum: Scalar[f32] = 0
                for batch in range(ug_shape_0):
                    var batch_offset = batch * ug_strides_0 + channel_offset

                    def vec_sum[
                        Nelts: Int
                    ](ux_uy: Int) {mut sum, imm ug, imm batch_offset}:
                        sum += ug.load[Nelts](batch_offset + ux_uy).reduce_add()

                    vectorize[nelts](ug_shape_2 * ug_shape_3, vec_sum)

                res[out_ch] = sum

            parallelize(bias_grad, ug_shape_1)

        return res^

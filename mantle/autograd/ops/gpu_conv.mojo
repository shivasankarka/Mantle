# ===----------------------------------------------------------------------=== #
# Mantle: GPU convolution
# ===----------------------------------------------------------------------=== #
"""GPU Conv2d with MAX dispatch and an NCHW-native fallback."""
from std.math import ceildiv
from max.gpu import thread_idx, block_idx, block_dim
from std.ffi import _Global
from std.os import abort
from std.utils import IndexList
from layout import TileTensor
from layout.tile_layout import row_major
from nn.conv.conv import conv_gpu

from mantle import f32
from mantle.core.tensor import Tensor, TensorShape, _shared_device_context
from mantle.core.device import Device
from .gpu_matmul import (
    gpu_matmul,
    gpu_matmul_bt,
    gpu_matmul_at,
    gpu_transpose_4d,
)

comptime _BLOCK = 256


def gpu_conv2d_forward_max[
    batch: Int,
    channels: Int,
    in_h: Int,
    in_w: Int,
    out_channels: Int,
    kh: Int,
    kw: Int,
    out_h: Int,
    out_w: Int,
    pad_h: Int,
    pad_w: Int,
    stride_h: Int,
    stride_w: Int,
    dilation_h: Int,
    dilation_w: Int,
](
    mut output: Tensor[f32, Device.gpu],
    inputs: Tensor[f32, Device.gpu],
    kernel: Tensor[f32, Device.gpu],
    bias: Tensor[f32, Device.gpu],
) raises:
    """MAX conv_gpu adapter: NCHW/OIHW Mantle buffers to NHWC/RSCF."""
    var input_nhwc = Tensor[f32, Device.gpu](
        TensorShape(batch, in_h, in_w, channels), uninitialized=True
    )
    var filter_rscf = Tensor[f32, Device.gpu](
        TensorShape(kh, kw, channels, out_channels), uninitialized=True
    )
    var output_nhwc = Tensor[f32, Device.gpu](
        TensorShape(batch, out_h, out_w, out_channels), uninitialized=True
    )
    gpu_transpose_4d(input_nhwc, inputs, TensorShape(0, 2, 3, 1))
    gpu_transpose_4d(filter_rscf, kernel, TensorShape(2, 3, 1, 0))
    var input_tt = TileTensor(
        ptr=input_nhwc.gpu_ptr().unsafe_origin_cast[MutAnyOrigin](),
        layout=row_major[batch, in_h, in_w, channels](),
    )
    var filter_tt = TileTensor(
        ptr=filter_rscf.gpu_ptr().unsafe_origin_cast[MutAnyOrigin](),
        layout=row_major[kh, kw, channels, out_channels](),
    )
    var output_tt = TileTensor(
        ptr=output_nhwc.gpu_ptr().unsafe_origin_cast[MutAnyOrigin](),
        layout=row_major[batch, out_h, out_w, out_channels](),
    )
    var stride = IndexList[2](stride_h, stride_w)
    var dilation = IndexList[2](dilation_h, dilation_w)
    var padding = IndexList[4](pad_h, pad_h, pad_w, pad_w)
    conv_gpu[
        input_type=DType.float32,
        filter_type=DType.float32,
        output_type=DType.float32,
    ](
        input_tt,
        filter_tt,
        output_tt,
        stride,
        dilation,
        padding,
        1,
        output.gpu_context(),
    )
    # NHWC is spatial-major, which is the same flattened layout consumed by
    # this conversion kernel. Fold the bias add into the NHWC -> NCHW pass.
    _cached_nchw_from_spatial_kernel()._call_with_pack_checked(
        output.gpu_context(),
        output.gpu_ptr(),
        output_nhwc.gpu_ptr(),
        bias.gpu_ptr(),
        Int64(output.num_elements()),
        Int64(out_channels),
        Int64(out_h),
        Int64(out_w),
        grid_dim=ceildiv(output.num_elements(), _BLOCK),
        block_dim=min(output.num_elements(), _BLOCK),
    )


def _make_kernel_fn[
    declared_arg_types: TypeList[Trait=AnyType, ...],
    //,
    func: def(* args: * declared_arg_types) thin -> None,
]() -> type_of(_shared_device_context().compile_function[func]()):
    try:
        return _shared_device_context().compile_function[func]()
    except e:
        abort("Mantle: GPU convolution kernel compile failed: " + String(e))


def _im2col_kernel(
    col: Pointer[Scalar[f32], MutAnyOrigin],
    inputs: Pointer[Scalar[f32], MutAnyOrigin],
    rows: Int64,
    channels: Int64,
    in_h: Int64,
    in_w: Int64,
    out_h: Int64,
    out_w: Int64,
    kh: Int64,
    kw: Int64,
    pad_h: Int64,
    pad_w: Int64,
    stride_h: Int64,
    stride_w: Int64,
    dilation_h: Int64,
    dilation_w: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    var kernel_values = channels * kh * kw
    if i < rows * kernel_values:
        var row = i // kernel_values
        var feature = i % kernel_values
        var batch = row // (out_h * out_w)
        var position = row % (out_h * out_w)
        var oy = position % out_w
        var ox = position // out_w
        var channel = feature // (kh * kw)
        var kernel_pos = feature % (kh * kw)
        var ky = kernel_pos % kw
        var kx = kernel_pos // kw
        var iy = ox * stride_h - pad_h + kx * dilation_h
        var ix = oy * stride_w - pad_w + ky * dilation_w
        if iy >= 0 and ix >= 0 and iy < in_h and ix < in_w:
            col.unsafe_store(
                Int(i),
                inputs.unsafe_load(
                    Int(((batch * channels + channel) * in_h + iy) * in_w + ix)
                ),
            )
        else:
            col.unsafe_store(Int(i), 0.0)


comptime _im2col_kernel_global = _Global[
    "mantle_gpu_kernel_im2col", _make_kernel_fn[_im2col_kernel]
]


def _cached_im2col_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_im2col_kernel]())
):
    return _im2col_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def _nchw_from_spatial_kernel(
    output: Pointer[Scalar[f32], MutAnyOrigin],
    spatial: Pointer[Scalar[f32], MutAnyOrigin],
    bias: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
    channels: Int64,
    out_h: Int64,
    out_w: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var position = i % (out_h * out_w)
        var channel = (i // (out_h * out_w)) % channels
        var batch = i // (channels * out_h * out_w)
        var spatial_index = (
            batch * out_h * out_w + position
        ) * channels + channel
        output.unsafe_store(
            Int(i),
            spatial.unsafe_load(Int(spatial_index))
            + bias.unsafe_load(Int(channel)),
        )


comptime _nchw_from_spatial_kernel_global = _Global[
    "mantle_gpu_kernel_nchw_from_spatial",
    _make_kernel_fn[_nchw_from_spatial_kernel],
]


def _cached_nchw_from_spatial_kernel() raises -> (
    type_of(
        _shared_device_context().compile_function[_nchw_from_spatial_kernel]()
    )
):
    return _nchw_from_spatial_kernel_global.get_or_create_ptr()[
        unsafe_offset=0
    ].copy()


def gpu_conv2d_forward_native[
    batch: Int,
    channels: Int,
    in_h: Int,
    in_w: Int,
    out_channels: Int,
    kh: Int,
    kw: Int,
    out_h: Int,
    out_w: Int,
    pad_h: Int,
    pad_w: Int,
    stride_h: Int,
    stride_w: Int,
    dilation_h: Int,
    dilation_w: Int,
](
    mut output: Tensor[f32, Device.gpu],
    inputs: Tensor[f32, Device.gpu],
    kernel: Tensor[f32, Device.gpu],
    bias: Tensor[f32, Device.gpu],
) raises:
    comptime rows = batch * out_h * out_w
    comptime kernel_values = channels * kh * kw
    var col = Tensor[f32, Device.gpu](
        TensorShape(rows, kernel_values), uninitialized=True
    )
    var spatial = Tensor[f32, Device.gpu](
        TensorShape(rows, out_channels), uninitialized=True
    )
    var ctx = output.gpu_context()
    _cached_im2col_kernel()._call_with_pack_checked(
        ctx,
        col.gpu_ptr(),
        inputs.gpu_ptr(),
        Int64(rows),
        Int64(channels),
        Int64(in_h),
        Int64(in_w),
        Int64(out_h),
        Int64(out_w),
        Int64(kh),
        Int64(kw),
        Int64(pad_h),
        Int64(pad_w),
        Int64(stride_h),
        Int64(stride_w),
        Int64(dilation_h),
        Int64(dilation_w),
        grid_dim=ceildiv(col.num_elements(), _BLOCK),
        block_dim=min(col.num_elements(), _BLOCK),
    )
    gpu_matmul_bt[rows, kernel_values, out_channels](spatial, col, kernel)
    _cached_nchw_from_spatial_kernel()._call_with_pack_checked(
        ctx,
        output.gpu_ptr(),
        spatial.gpu_ptr(),
        bias.gpu_ptr(),
        Int64(output.num_elements()),
        Int64(out_channels),
        Int64(out_h),
        Int64(out_w),
        grid_dim=ceildiv(output.num_elements(), _BLOCK),
        block_dim=min(output.num_elements(), _BLOCK),
    )


def gpu_conv2d_forward[
    batch: Int,
    channels: Int,
    in_h: Int,
    in_w: Int,
    out_channels: Int,
    kh: Int,
    kw: Int,
    out_h: Int,
    out_w: Int,
    pad_h: Int,
    pad_w: Int,
    stride_h: Int,
    stride_w: Int,
    dilation_h: Int,
    dilation_w: Int,
](
    mut output: Tensor[f32, Device.gpu],
    inputs: Tensor[f32, Device.gpu],
    kernel: Tensor[f32, Device.gpu],
    bias: Tensor[f32, Device.gpu],
) raises:
    """Dispatch Conv2d through MAX's target-tuned GPU implementation."""
    gpu_conv2d_forward_max[
        batch,
        channels,
        in_h,
        in_w,
        out_channels,
        kh,
        kw,
        out_h,
        out_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
    ](output, inputs, kernel, bias)


def _spatial_from_nchw_kernel(
    spatial: Pointer[Scalar[f32], MutAnyOrigin],
    src: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
    channels: Int64,
    height: Int64,
    width: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var position = i % (height * width)
        var channel = (i // (height * width)) % channels
        var batch = i // (channels * height * width)
        spatial.unsafe_store(
            Int((batch * height * width + position) * channels + channel),
            src.unsafe_load(Int(i)),
        )


comptime _spatial_from_nchw_kernel_global = _Global[
    "mantle_gpu_kernel_spatial_from_nchw",
    _make_kernel_fn[_spatial_from_nchw_kernel],
]


def _cached_spatial_from_nchw_kernel() raises -> (
    type_of(
        _shared_device_context().compile_function[_spatial_from_nchw_kernel]()
    )
):
    return _spatial_from_nchw_kernel_global.get_or_create_ptr()[
        unsafe_offset=0
    ].copy()


def _col2im_kernel(
    dst: Pointer[Scalar[f32], MutAnyOrigin],
    col: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
    channels: Int64,
    in_h: Int64,
    in_w: Int64,
    out_h: Int64,
    out_w: Int64,
    kh: Int64,
    kw: Int64,
    pad_h: Int64,
    pad_w: Int64,
    stride_h: Int64,
    stride_w: Int64,
    dilation_h: Int64,
    dilation_w: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var x = i % in_w
        var y = (i // in_w) % in_h
        var channel = (i // (in_h * in_w)) % channels
        var batch = i // (channels * in_h * in_w)
        var total: Scalar[f32] = 0.0
        # Invert the convolution coordinate equation from this input element
        # to its contributing output positions.  The former implementation
        # scanned every output location (O(OH*OW*KH*KW) per input); each
        # input can only participate through its KH*KW kernel offsets.
        for kx in range(Int(kh)):
            var output_y_numerator = y + pad_h - Int64(kx) * dilation_h
            if output_y_numerator < 0 or output_y_numerator % stride_h != 0:
                continue
            var ox = output_y_numerator // stride_h
            if ox >= out_h:
                continue
            for ky in range(Int(kw)):
                var output_x_numerator = x + pad_w - Int64(ky) * dilation_w
                if output_x_numerator < 0 or output_x_numerator % stride_w != 0:
                    continue
                var oy = output_x_numerator // stride_w
                if oy < out_w:
                    var row = batch * out_h * out_w + ox * out_w + oy
                    var feature = (channel * kh + Int64(kx)) * kw + Int64(ky)
                    total += col.unsafe_load(
                        Int(row * channels * kh * kw + feature)
                    )
        dst.unsafe_store(Int(i), total)


comptime _col2im_kernel_global = _Global[
    "mantle_gpu_kernel_col2im", _make_kernel_fn[_col2im_kernel]
]


def _cached_col2im_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_col2im_kernel]())
):
    return _col2im_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def _transpose_kernel_gradient_kernel(
    output: Pointer[Scalar[f32], MutAnyOrigin],
    source: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
    kernel_values: Int64,
    out_channels: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var out_channel = i // kernel_values
        var feature = i % kernel_values
        output.unsafe_store(
            Int(i),
            source.unsafe_load(Int(feature * out_channels + out_channel)),
        )


comptime _transpose_kernel_gradient_kernel_global = _Global[
    "mantle_gpu_kernel_transpose_conv_gradient",
    _make_kernel_fn[_transpose_kernel_gradient_kernel],
]


def _cached_transpose_kernel_gradient_kernel() raises -> (
    type_of(
        _shared_device_context().compile_function[
            _transpose_kernel_gradient_kernel
        ]()
    )
):
    return _transpose_kernel_gradient_kernel_global.get_or_create_ptr()[
        unsafe_offset=0
    ].copy()


def _bias_gradient_kernel(
    output: Pointer[Scalar[f32], MutAnyOrigin],
    upper_grad: Pointer[Scalar[f32], MutAnyOrigin],
    out_channels: Int64,
    rows: Int64,
):
    var channel = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if channel < out_channels:
        var total: Scalar[f32] = 0.0
        for row in range(Int(rows)):
            total += upper_grad.unsafe_load(
                Int((Int64(row) * out_channels) + channel)
            )
        output.unsafe_store(Int(channel), total)


comptime _bias_gradient_kernel_global = _Global[
    "mantle_gpu_kernel_conv_bias_gradient",
    _make_kernel_fn[_bias_gradient_kernel],
]


def _cached_bias_gradient_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_bias_gradient_kernel]())
):
    return _bias_gradient_kernel_global.get_or_create_ptr()[
        unsafe_offset=0
    ].copy()


def gpu_conv2d_input_backward[
    batch: Int,
    channels: Int,
    in_h: Int,
    in_w: Int,
    out_channels: Int,
    kh: Int,
    kw: Int,
    out_h: Int,
    out_w: Int,
    pad_h: Int,
    pad_w: Int,
    stride_h: Int,
    stride_w: Int,
    dilation_h: Int,
    dilation_w: Int,
](
    mut output: Tensor[f32, Device.gpu],
    upper_grad: Tensor[f32, Device.gpu],
    kernel: Tensor[f32, Device.gpu],
) raises:
    comptime rows = batch * out_h * out_w
    comptime kernel_values = channels * kh * kw
    var spatial = Tensor[f32, Device.gpu](
        TensorShape(rows, out_channels), uninitialized=True
    )
    var col_grad = Tensor[f32, Device.gpu](
        TensorShape(rows, kernel_values), uninitialized=True
    )
    var ctx = output.gpu_context()
    _cached_spatial_from_nchw_kernel()._call_with_pack_checked(
        ctx,
        spatial.gpu_ptr(),
        upper_grad.gpu_ptr(),
        Int64(upper_grad.num_elements()),
        Int64(out_channels),
        Int64(out_h),
        Int64(out_w),
        grid_dim=ceildiv(upper_grad.num_elements(), _BLOCK),
        block_dim=min(upper_grad.num_elements(), _BLOCK),
    )
    gpu_matmul[rows, out_channels, kernel_values](col_grad, spatial, kernel)
    _cached_col2im_kernel()._call_with_pack_checked(
        ctx,
        output.gpu_ptr(),
        col_grad.gpu_ptr(),
        Int64(output.num_elements()),
        Int64(channels),
        Int64(in_h),
        Int64(in_w),
        Int64(out_h),
        Int64(out_w),
        Int64(kh),
        Int64(kw),
        Int64(pad_h),
        Int64(pad_w),
        Int64(stride_h),
        Int64(stride_w),
        Int64(dilation_h),
        Int64(dilation_w),
        grid_dim=ceildiv(output.num_elements(), _BLOCK),
        block_dim=min(output.num_elements(), _BLOCK),
    )


def gpu_conv2d_kernel_backward[
    batch: Int,
    channels: Int,
    in_h: Int,
    in_w: Int,
    out_channels: Int,
    kh: Int,
    kw: Int,
    out_h: Int,
    out_w: Int,
    pad_h: Int,
    pad_w: Int,
    stride_h: Int,
    stride_w: Int,
    dilation_h: Int,
    dilation_w: Int,
](
    mut output: Tensor[f32, Device.gpu],
    inputs: Tensor[f32, Device.gpu],
    upper_grad: Tensor[f32, Device.gpu],
) raises:
    comptime rows = batch * out_h * out_w
    comptime kernel_values = channels * kh * kw
    var col = Tensor[f32, Device.gpu](
        TensorShape(rows, kernel_values), uninitialized=True
    )
    var spatial = Tensor[f32, Device.gpu](
        TensorShape(rows, out_channels), uninitialized=True
    )
    var transposed = Tensor[f32, Device.gpu](
        TensorShape(kernel_values, out_channels), uninitialized=True
    )
    var ctx = output.gpu_context()
    _cached_im2col_kernel()._call_with_pack_checked(
        ctx,
        col.gpu_ptr(),
        inputs.gpu_ptr(),
        Int64(rows),
        Int64(channels),
        Int64(in_h),
        Int64(in_w),
        Int64(out_h),
        Int64(out_w),
        Int64(kh),
        Int64(kw),
        Int64(pad_h),
        Int64(pad_w),
        Int64(stride_h),
        Int64(stride_w),
        Int64(dilation_h),
        Int64(dilation_w),
        grid_dim=ceildiv(col.num_elements(), _BLOCK),
        block_dim=min(col.num_elements(), _BLOCK),
    )
    _cached_spatial_from_nchw_kernel()._call_with_pack_checked(
        ctx,
        spatial.gpu_ptr(),
        upper_grad.gpu_ptr(),
        Int64(upper_grad.num_elements()),
        Int64(out_channels),
        Int64(out_h),
        Int64(out_w),
        grid_dim=ceildiv(upper_grad.num_elements(), _BLOCK),
        block_dim=min(upper_grad.num_elements(), _BLOCK),
    )
    gpu_matmul_at[rows, kernel_values, out_channels](transposed, col, spatial)
    _cached_transpose_kernel_gradient_kernel()._call_with_pack_checked(
        ctx,
        output.gpu_ptr(),
        transposed.gpu_ptr(),
        Int64(output.num_elements()),
        Int64(kernel_values),
        Int64(out_channels),
        grid_dim=ceildiv(output.num_elements(), _BLOCK),
        block_dim=min(output.num_elements(), _BLOCK),
    )


def gpu_conv2d_bias_backward[
    batch: Int,
    out_channels: Int,
    out_h: Int,
    out_w: Int,
](
    mut output: Tensor[f32, Device.gpu],
    upper_grad: Tensor[f32, Device.gpu],
) raises:
    comptime rows = batch * out_h * out_w
    var ctx = output.gpu_context()
    # The reduction expects spatial-major rows. Use a short-lived layout conversion.
    var spatial = Tensor[f32, Device.gpu](
        TensorShape(rows, out_channels), uninitialized=True
    )
    _cached_spatial_from_nchw_kernel()._call_with_pack_checked(
        ctx,
        spatial.gpu_ptr(),
        upper_grad.gpu_ptr(),
        Int64(upper_grad.num_elements()),
        Int64(out_channels),
        Int64(out_h),
        Int64(out_w),
        grid_dim=ceildiv(upper_grad.num_elements(), _BLOCK),
        block_dim=min(upper_grad.num_elements(), _BLOCK),
    )
    _cached_bias_gradient_kernel()._call_with_pack_checked(
        ctx,
        output.gpu_ptr(),
        spatial.gpu_ptr(),
        Int64(out_channels),
        Int64(rows),
        grid_dim=ceildiv(out_channels, _BLOCK),
        block_dim=min(out_channels, _BLOCK),
    )

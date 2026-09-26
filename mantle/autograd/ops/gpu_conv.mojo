# ===----------------------------------------------------------------------=== #
# Mantle: GPU convolution
# ===----------------------------------------------------------------------=== #
"""GPU Conv2d with MAX dispatch and an NCHW-native fallback."""
from std.math import ceildiv
from max.gpu import thread_idx, block_idx, block_dim
from std.ffi import _Global
from std.os import abort
from std.sys import CompilationTarget
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
    comptime if channels != 1:
        _cached_spatial_from_nchw_kernel[
            channels, in_h, in_w
        ]()._call_with_pack_checked(
            output.gpu_context(),
            input_nhwc.gpu_ptr(),
            inputs.gpu_ptr(),
            Int64(inputs.num_elements()),
            grid_dim=ceildiv(inputs.num_elements(), _BLOCK),
            block_dim=min(inputs.num_elements(), _BLOCK),
        )
    _cached_filter_oihw_to_rscf_kernel[
        channels, out_channels, kh, kw
    ]()._call_with_pack_checked(
        output.gpu_context(),
        filter_rscf.gpu_ptr(),
        kernel.gpu_ptr(),
        Int64(kernel.num_elements()),
        grid_dim=ceildiv(kernel.num_elements(), _BLOCK),
        block_dim=min(kernel.num_elements(), _BLOCK),
    )
    var input_ptr = input_nhwc.gpu_ptr().unsafe_origin_cast[MutAnyOrigin]()
    comptime if channels == 1:
        # NCHW and NHWC have identical storage when C == 1.
        input_ptr = (
            inputs.gpu_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )
    var input_tt = TileTensor(
        ptr=input_ptr,
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
    _cached_nchw_from_spatial_kernel[
        out_channels, out_h, out_w
    ]()._call_with_pack_checked(
        output.gpu_context(),
        output.gpu_ptr(),
        output_nhwc.gpu_ptr(),
        bias.gpu_ptr(),
        Int64(output.num_elements()),
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


def _conv2d_forward_direct_kernel[
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
    output: Pointer[Scalar[f32], MutAnyOrigin],
    inputs: Pointer[Scalar[f32], MutAnyOrigin],
    kernel: Pointer[Scalar[f32], MutAnyOrigin],
    bias: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    """NCHW/OIHW convolution without layout-conversion temporaries."""
    comptime channels64 = Int64(channels)
    comptime in_h64 = Int64(in_h)
    comptime in_w64 = Int64(in_w)
    comptime out_channels64 = Int64(out_channels)
    comptime kh64 = Int64(kh)
    comptime kw64 = Int64(kw)
    comptime out_h64 = Int64(out_h)
    comptime out_w64 = Int64(out_w)
    comptime pad_h64 = Int64(pad_h)
    comptime pad_w64 = Int64(pad_w)
    comptime stride_h64 = Int64(stride_h)
    comptime stride_w64 = Int64(stride_w)
    comptime dilation_h64 = Int64(dilation_h)
    comptime dilation_w64 = Int64(dilation_w)

    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var oy = i % out_w64
        var ox = (i // out_w64) % out_h64
        var out_channel = (i // (out_w64 * out_h64)) % out_channels64
        var batch_index = i // (out_w64 * out_h64 * out_channels64)
        var total = bias.unsafe_load(Int(out_channel))
        for channel in range(channels):
            for kx in range(kh):
                var iy = ox * stride_h64 - pad_h64 + Int64(kx) * dilation_h64
                if iy < 0 or iy >= in_h64:
                    continue
                comptime if stride_w == 1 and dilation_w == 1:
                    comptime vector_width = 4
                    var ky_start = max(Int64(0), pad_w64 - oy)
                    var ky_end = min(kw64, in_w64 + pad_w64 - oy)
                    var input_row = (
                        (
                            (batch_index * channels64 + Int64(channel)) * in_h64
                            + iy
                        )
                        * in_w64
                        + oy
                        - pad_w64
                    )
                    var kernel_row = (
                        (out_channel * channels64 + Int64(channel)) * kh64
                        + Int64(kx)
                    ) * kw64
                    var ky = ky_start
                    while ky + Int64(vector_width) <= ky_end:
                        total += (
                            inputs.unsafe_load[width=vector_width](
                                Int(input_row + ky)
                            )
                            * kernel.unsafe_load[width=vector_width](
                                Int(kernel_row + ky)
                            )
                        ).reduce_add()
                        ky += Int64(vector_width)
                    while ky < ky_end:
                        total += inputs.unsafe_load(
                            Int(input_row + ky)
                        ) * kernel.unsafe_load(Int(kernel_row + ky))
                        ky += 1
                else:
                    for ky in range(kw):
                        var ix = (
                            oy * stride_w64 - pad_w64 + Int64(ky) * dilation_w64
                        )
                        if ix >= 0 and ix < in_w64:
                            var input_index = (
                                (batch_index * channels64 + Int64(channel))
                                * in_h64
                                + iy
                            ) * in_w64 + ix
                            var kernel_index = (
                                (out_channel * channels64 + Int64(channel))
                                * kh64
                                + Int64(kx)
                            ) * kw64 + Int64(ky)
                            total += inputs.unsafe_load(
                                Int(input_index)
                            ) * kernel.unsafe_load(Int(kernel_index))
        output.unsafe_store(Int(i), total)


def _cached_conv2d_forward_direct_kernel[
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
]() raises -> type_of(
    _shared_device_context().compile_function[
        _conv2d_forward_direct_kernel[
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
        ]
    ]()
):
    comptime name = (
        "mantle_gpu_conv2d_forward_direct_"
        + String(batch)
        + "_"
        + String(channels)
        + "_"
        + String(in_h)
        + "_"
        + String(in_w)
        + "_"
        + String(out_channels)
        + "_"
        + String(kh)
        + "_"
        + String(kw)
        + "_"
        + String(out_h)
        + "_"
        + String(out_w)
        + "_"
        + String(pad_h)
        + "_"
        + String(pad_w)
        + "_"
        + String(stride_h)
        + "_"
        + String(stride_w)
        + "_"
        + String(dilation_h)
        + "_"
        + String(dilation_w)
    )
    comptime global_ = _Global[
        name,
        _make_kernel_fn[
            _conv2d_forward_direct_kernel[
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
            ]
        ],
    ]
    return global_.get_or_create_ptr()[unsafe_offset=0].copy()


def gpu_conv2d_forward_direct[
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
    _cached_conv2d_forward_direct_kernel[
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
    ]()._call_with_pack_checked(
        output.gpu_context(),
        output.gpu_ptr(),
        inputs.gpu_ptr(),
        kernel.gpu_ptr(),
        bias.gpu_ptr(),
        Int64(output.num_elements()),
        grid_dim=ceildiv(output.num_elements(), _BLOCK),
        block_dim=min(output.num_elements(), _BLOCK),
    )


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


def _nchw_from_spatial_kernel[
    channels: Int, out_h: Int, out_w: Int
](
    output: Pointer[Scalar[f32], MutAnyOrigin],
    spatial: Pointer[Scalar[f32], MutAnyOrigin],
    bias: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    """NHWC (spatial-major) -> NCHW plus a fused bias add.

    `channels`/`out_h`/`out_w` are compile-time parameters rather than
    runtime arguments -- like the weight-gradient kernels above, this
    kernel's index math is all division/modulo by these three, and a
    compile-time-constant divisor is far cheaper on GPU than a runtime one.
    """
    comptime channels64 = Int64(channels)
    comptime spatial64 = Int64(out_h * out_w)

    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var position = i % spatial64
        var channel = (i // spatial64) % channels64
        var batch = i // (channels64 * spatial64)
        var spatial_index = (
            batch * spatial64 + position
        ) * channels64 + channel
        output.unsafe_store(
            Int(i),
            spatial.unsafe_load(Int(spatial_index))
            + bias.unsafe_load(Int(channel)),
        )


def _cached_nchw_from_spatial_kernel[
    channels: Int, out_h: Int, out_w: Int
]() raises -> type_of(
    _shared_device_context().compile_function[
        _nchw_from_spatial_kernel[channels, out_h, out_w]
    ]()
):
    comptime name = (
        "mantle_gpu_kernel_nchw_from_spatial_"
        + String(channels)
        + "_"
        + String(out_h)
        + "_"
        + String(out_w)
    )
    comptime global_ = _Global[
        name, _make_kernel_fn[_nchw_from_spatial_kernel[channels, out_h, out_w]]
    ]
    return global_.get_or_create_ptr()[unsafe_offset=0].copy()


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
    _cached_nchw_from_spatial_kernel[
        out_channels, out_h, out_w
    ]()._call_with_pack_checked(
        ctx,
        output.gpu_ptr(),
        spatial.gpu_ptr(),
        bias.gpu_ptr(),
        Int64(output.num_elements()),
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
    """Avoid conversion for single-channel Apple inputs; use MAX otherwise."""
    comptime if CompilationTarget.is_macos() and channels == 1:
        gpu_conv2d_forward_direct[
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
    else:
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


def _spatial_from_nchw_kernel[
    channels: Int, height: Int, width: Int
](
    spatial: Pointer[Scalar[f32], MutAnyOrigin],
    src: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    """NCHW -> NHWC (spatial-major); see `_nchw_from_spatial_kernel` above
    for why `channels`/`height`/`width` are compile-time parameters."""
    comptime channels64 = Int64(channels)
    comptime spatial64 = Int64(height * width)

    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var position = i % spatial64
        var channel = (i // spatial64) % channels64
        var batch = i // (channels64 * spatial64)
        spatial.unsafe_store(
            Int((batch * spatial64 + position) * channels64 + channel),
            src.unsafe_load(Int(i)),
        )


def _cached_spatial_from_nchw_kernel[
    channels: Int, height: Int, width: Int
]() raises -> type_of(
    _shared_device_context().compile_function[
        _spatial_from_nchw_kernel[channels, height, width]
    ]()
):
    comptime name = (
        "mantle_gpu_kernel_spatial_from_nchw_"
        + String(channels)
        + "_"
        + String(height)
        + "_"
        + String(width)
    )
    comptime global_ = _Global[
        name,
        _make_kernel_fn[_spatial_from_nchw_kernel[channels, height, width]],
    ]
    return global_.get_or_create_ptr()[unsafe_offset=0].copy()


def _filter_oihw_to_rscf_kernel[
    channels: Int, out_channels: Int, kh: Int, kw: Int
](
    output: Pointer[Scalar[f32], MutAnyOrigin],
    kernel: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    """Convert Mantle OIHW filters to MAX's RSCF layout."""
    comptime channels64 = Int64(channels)
    comptime out_channels64 = Int64(out_channels)
    comptime kh64 = Int64(kh)
    comptime kw64 = Int64(kw)
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var out_channel = i % out_channels64
        var remaining = i // out_channels64
        var channel = remaining % channels64
        remaining //= channels64
        var kernel_x = remaining % kw64
        var kernel_y = remaining // kw64
        var source = (
            (out_channel * channels64 + channel) * kh64 + kernel_y
        ) * kw64 + kernel_x
        output.unsafe_store(Int(i), kernel.unsafe_load(Int(source)))


def _cached_filter_oihw_to_rscf_kernel[
    channels: Int, out_channels: Int, kh: Int, kw: Int
]() raises -> type_of(
    _shared_device_context().compile_function[
        _filter_oihw_to_rscf_kernel[channels, out_channels, kh, kw]
    ]()
):
    comptime name = (
        "mantle_gpu_kernel_filter_oihw_to_rscf_"
        + String(channels)
        + "_"
        + String(out_channels)
        + "_"
        + String(kh)
        + "_"
        + String(kw)
    )
    comptime global_ = _Global[
        name,
        _make_kernel_fn[
            _filter_oihw_to_rscf_kernel[channels, out_channels, kh, kw]
        ],
    ]
    return global_.get_or_create_ptr()[unsafe_offset=0].copy()


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


comptime _CONV_WEIGHT_TARGET_THREADS = 32768
"""Target thread count for the weight/bias-gradient reductions below.

Both `_conv_kernel_gradient_partial_kernel` and
`_conv_bias_gradient_partial_kernel` reduce over the batch axis in two
passes: a first pass gives every (output element, batch-*chunk*) pair its
own thread, then `_sum_rows_kernel` folds the (small) chunk axis down to the
final gradient.

Notes:
    This constant balances occupancy across layers with varying output
    element counts, avoiding hand-picked per-layer thresholds.
"""


def _conv_kernel_gradient_partial_kernel[
    batch_size: Int,
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
    num_chunks: Int,
    chunk_size: Int,
](
    partial: Pointer[Scalar[f32], MutAnyOrigin],
    inputs: Pointer[Scalar[f32], MutAnyOrigin],
    upper_grad: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    """One thread per (output-weight, batch-chunk) pair.

    Each thread only reduces over `chunk_size` batch samples (times the
    spatial output positions), rather than the whole batch -- see
    `_CONV_WEIGHT_TARGET_THREADS` for how `chunk_size` is picked.
    `_sum_rows_kernel` folds the chunk axis afterwards.

    Notes:
        Every shape dimension is a compile-time parameter, not a runtime
        argument, because integer division/modulo by a compile-time-constant
        divisor compiles down to cheap multiply-shift, while GPUs execute a
        division by a runtime value as a slow instruction. Trade-off: one
        compiled kernel per distinct conv-layer shape instead of one shared
        kernel for all shapes.
    """
    comptime kh64 = Int64(kh)
    comptime kw64 = Int64(kw)
    comptime channels64 = Int64(channels)
    comptime in_h64 = Int64(in_h)
    comptime in_w64 = Int64(in_w)
    comptime out_channels64 = Int64(out_channels)
    comptime out_h64 = Int64(out_h)
    comptime out_w64 = Int64(out_w)
    comptime pad_h64 = Int64(pad_h)
    comptime pad_w64 = Int64(pad_w)
    comptime stride_h64 = Int64(stride_h)
    comptime stride_w64 = Int64(stride_w)
    comptime dilation_h64 = Int64(dilation_h)
    comptime dilation_w64 = Int64(dilation_w)
    comptime batch_size64 = Int64(batch_size)

    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var feature = i // Int64(num_chunks)
        var chunk_id = i % Int64(num_chunks)
        var batch_start = chunk_id * Int64(chunk_size)
        var within_feature = feature % (channels64 * kh64 * kw64)
        var out_channel = feature // (channels64 * kh64 * kw64)
        var channel = within_feature // (kh64 * kw64)
        var kernel_position = within_feature % (kh64 * kw64)
        var kx = kernel_position // kw64
        var ky = kernel_position % kw64
        var total: Scalar[f32] = 0.0
        for bo in range(chunk_size):
            var batch = batch_start + Int64(bo)
            if batch >= batch_size64:
                continue
            for ox in range(out_h):
                var iy = Int64(ox) * stride_h64 - pad_h64 + kx * dilation_h64
                if iy < 0 or iy >= in_h64:
                    continue
                comptime if stride_w == 1 and dilation_w == 1:
                    # Across an output row both source arrays are contiguous.
                    # Wider rows amortize eight-lane loads; short rows retain
                    # better occupancy with four lanes.
                    comptime vector_width = 8 if out_w >= 28 else 4
                    var input_x_offset = ky - pad_w64
                    var oy_start = max(Int64(0), -input_x_offset)
                    var oy_end = min(out_w64, in_w64 - input_x_offset)
                    var input_row = (
                        (batch * channels64 + channel) * in_h64 + iy
                    ) * in_w64 + input_x_offset
                    var grad_row = (
                        (batch * out_channels64 + out_channel) * out_h64
                        + Int64(ox)
                    ) * out_w64
                    var oy = oy_start
                    while oy + Int64(vector_width) <= oy_end:
                        total += (
                            inputs.unsafe_load[width=vector_width](
                                Int(input_row + oy)
                            )
                            * upper_grad.unsafe_load[width=vector_width](
                                Int(grad_row + oy)
                            )
                        ).reduce_add()
                        oy += Int64(vector_width)
                    while oy < oy_end:
                        total += inputs.unsafe_load(
                            Int(input_row + oy)
                        ) * upper_grad.unsafe_load(Int(grad_row + oy))
                        oy += 1
                else:
                    for oy in range(out_w):
                        var ix = (
                            Int64(oy) * stride_w64 - pad_w64 + ky * dilation_w64
                        )
                        if ix >= 0 and ix < in_w64:
                            var input_index = (
                                (batch * channels64 + channel) * in_h64 + iy
                            ) * in_w64 + ix
                            var grad_index = (
                                (batch * out_channels64 + out_channel) * out_h64
                                + Int64(ox)
                            ) * out_w64 + Int64(oy)
                            total += inputs.unsafe_load(
                                Int(input_index)
                            ) * upper_grad.unsafe_load(Int(grad_index))
        partial.unsafe_store(Int(i), total)


def _cached_conv_kernel_gradient_partial_kernel[
    batch_size: Int,
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
    num_chunks: Int,
    chunk_size: Int,
]() raises -> type_of(
    _shared_device_context().compile_function[
        _conv_kernel_gradient_partial_kernel[
            batch_size,
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
            num_chunks,
            chunk_size,
        ]
    ]()
):
    comptime name = (
        "mantle_gpu_kernel_conv_wgrad_"
        + String(batch_size)
        + "_"
        + String(channels)
        + "_"
        + String(in_h)
        + "_"
        + String(in_w)
        + "_"
        + String(out_channels)
        + "_"
        + String(kh)
        + "_"
        + String(kw)
        + "_"
        + String(out_h)
        + "_"
        + String(out_w)
        + "_"
        + String(pad_h)
        + "_"
        + String(pad_w)
        + "_"
        + String(stride_h)
        + "_"
        + String(stride_w)
        + "_"
        + String(dilation_h)
        + "_"
        + String(dilation_w)
        + "_"
        + String(num_chunks)
        + "_"
        + String(chunk_size)
    )
    comptime global_ = _Global[
        name,
        _make_kernel_fn[
            _conv_kernel_gradient_partial_kernel[
                batch_size,
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
                num_chunks,
                chunk_size,
            ]
        ],
    ]
    return global_.get_or_create_ptr()[unsafe_offset=0].copy()


def _conv_bias_gradient_partial_kernel[
    batch_size: Int,
    out_channels: Int,
    out_h: Int,
    out_w: Int,
    num_chunks: Int,
    chunk_size: Int,
](
    partial: Pointer[Scalar[f32], MutAnyOrigin],
    upper_grad: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    """One thread per (out-channel, batch-chunk) pair; see the kernel-
    gradient partial kernel above for why this is split out of the batch
    loop, and why every shape dimension is a compile-time parameter."""
    comptime out_channels64 = Int64(out_channels)
    comptime spatial64 = Int64(out_h * out_w)
    comptime batch_size64 = Int64(batch_size)

    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var channel = i // Int64(num_chunks)
        var chunk_id = i % Int64(num_chunks)
        var batch_start = chunk_id * Int64(chunk_size)
        var total: Scalar[f32] = 0.0
        for bo in range(chunk_size):
            var batch = batch_start + Int64(bo)
            if batch >= batch_size64:
                continue
            var base = (batch * out_channels64 + channel) * spatial64
            for position in range(out_h * out_w):
                total += upper_grad.unsafe_load(Int(base + Int64(position)))
        partial.unsafe_store(Int(i), total)


def _cached_conv_bias_gradient_partial_kernel[
    batch_size: Int,
    out_channels: Int,
    out_h: Int,
    out_w: Int,
    num_chunks: Int,
    chunk_size: Int,
]() raises -> type_of(
    _shared_device_context().compile_function[
        _conv_bias_gradient_partial_kernel[
            batch_size, out_channels, out_h, out_w, num_chunks, chunk_size
        ]
    ]()
):
    comptime name = (
        "mantle_gpu_kernel_conv_bgrad_"
        + String(batch_size)
        + "_"
        + String(out_channels)
        + "_"
        + String(out_h)
        + "_"
        + String(out_w)
        + "_"
        + String(num_chunks)
        + "_"
        + String(chunk_size)
    )
    comptime global_ = _Global[
        name,
        _make_kernel_fn[
            _conv_bias_gradient_partial_kernel[
                batch_size, out_channels, out_h, out_w, num_chunks, chunk_size
            ]
        ],
    ]
    return global_.get_or_create_ptr()[unsafe_offset=0].copy()


def _sum_rows_kernel(
    output: Pointer[Scalar[f32], MutAnyOrigin],
    partial: Pointer[Scalar[f32], MutAnyOrigin],
    rows: Int64,
    chunks: Int64,
):
    """Folds a `(rows, chunks)` buffer's last axis into `output[rows]`.

    Used to finish the batch-axis reduction the two partial kernels above
    split off -- `chunks` is small (the batch size), so this second pass is
    cheap even though it's back to one thread per row.
    """
    var row = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if row < rows:
        var total: Scalar[f32] = 0.0
        var base = row * chunks
        for c in range(Int(chunks)):
            total += partial.unsafe_load(Int(base + Int64(c)))
        output.unsafe_store(Int(row), total)


comptime _sum_rows_kernel_global = _Global[
    "mantle_gpu_kernel_sum_rows", _make_kernel_fn[_sum_rows_kernel]
]


def _cached_sum_rows_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_sum_rows_kernel]())
):
    return _sum_rows_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


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
    _cached_spatial_from_nchw_kernel[
        out_channels, out_h, out_w
    ]()._call_with_pack_checked(
        ctx,
        spatial.gpu_ptr(),
        upper_grad.gpu_ptr(),
        Int64(upper_grad.num_elements()),
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
    var ctx = output.gpu_context()
    comptime out_elems = channels * kh * kw * out_channels
    comptime num_chunks = min(
        batch, max(1, _CONV_WEIGHT_TARGET_THREADS // out_elems)
    )
    comptime chunk_size = ceildiv(batch, num_chunks)

    var partial = Tensor[f32, Device.gpu](
        TensorShape(out_elems, num_chunks), uninitialized=True
    )
    _cached_conv_kernel_gradient_partial_kernel[
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
        num_chunks,
        chunk_size,
    ]()._call_with_pack_checked(
        ctx,
        partial.gpu_ptr(),
        inputs.gpu_ptr(),
        upper_grad.gpu_ptr(),
        Int64(partial.num_elements()),
        grid_dim=ceildiv(partial.num_elements(), _BLOCK),
        block_dim=min(partial.num_elements(), _BLOCK),
    )
    _cached_sum_rows_kernel()._call_with_pack_checked(
        ctx,
        output.gpu_ptr(),
        partial.gpu_ptr(),
        Int64(out_elems),
        Int64(num_chunks),
        grid_dim=ceildiv(out_elems, _BLOCK),
        block_dim=min(out_elems, _BLOCK),
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
    var ctx = output.gpu_context()
    comptime num_chunks = min(
        batch, max(1, _CONV_WEIGHT_TARGET_THREADS // out_channels)
    )
    comptime chunk_size = ceildiv(batch, num_chunks)

    var partial = Tensor[f32, Device.gpu](
        TensorShape(out_channels, num_chunks), uninitialized=True
    )
    _cached_conv_bias_gradient_partial_kernel[
        batch, out_channels, out_h, out_w, num_chunks, chunk_size
    ]()._call_with_pack_checked(
        ctx,
        partial.gpu_ptr(),
        upper_grad.gpu_ptr(),
        Int64(partial.num_elements()),
        grid_dim=ceildiv(partial.num_elements(), _BLOCK),
        block_dim=min(partial.num_elements(), _BLOCK),
    )
    _cached_sum_rows_kernel()._call_with_pack_checked(
        ctx,
        output.gpu_ptr(),
        partial.gpu_ptr(),
        Int64(out_channels),
        Int64(num_chunks),
        grid_dim=ceildiv(out_channels, _BLOCK),
        block_dim=min(out_channels, _BLOCK),
    )


def gpu_conv2d_parameter_backward_direct[
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
    mut kernel_grad: Tensor[f32, Device.gpu],
    mut bias_grad: Tensor[f32, Device.gpu],
    inputs: Tensor[f32, Device.gpu],
    upper_grad: Tensor[f32, Device.gpu],
) raises:
    """Compute Conv2d parameter gradients without materializing im2col.

    Uses a two-pass, batch-parallel reduction where a first pass gives each
    (weight, batch-chunk) pair its own thread, then a second pass folds the
    chunk axis to the final gradient.
    """
    var ctx = kernel_grad.gpu_context()
    comptime out_elems = channels * kh * kw * out_channels
    comptime num_chunks = min(
        batch, max(1, _CONV_WEIGHT_TARGET_THREADS // out_elems)
    )
    comptime chunk_size = ceildiv(batch, num_chunks)

    var kernel_partial = Tensor[f32, Device.gpu](
        TensorShape(out_elems, num_chunks), uninitialized=True
    )
    _cached_conv_kernel_gradient_partial_kernel[
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
        num_chunks,
        chunk_size,
    ]()._call_with_pack_checked(
        ctx,
        kernel_partial.gpu_ptr(),
        inputs.gpu_ptr(),
        upper_grad.gpu_ptr(),
        Int64(kernel_partial.num_elements()),
        grid_dim=ceildiv(kernel_partial.num_elements(), _BLOCK),
        block_dim=min(kernel_partial.num_elements(), _BLOCK),
    )
    _cached_sum_rows_kernel()._call_with_pack_checked(
        ctx,
        kernel_grad.gpu_ptr(),
        kernel_partial.gpu_ptr(),
        Int64(out_elems),
        Int64(num_chunks),
        grid_dim=ceildiv(out_elems, _BLOCK),
        block_dim=min(out_elems, _BLOCK),
    )

    comptime bias_num_chunks = min(
        batch, max(1, _CONV_WEIGHT_TARGET_THREADS // out_channels)
    )
    comptime bias_chunk_size = ceildiv(batch, bias_num_chunks)

    var bias_partial = Tensor[f32, Device.gpu](
        TensorShape(out_channels, bias_num_chunks), uninitialized=True
    )
    _cached_conv_bias_gradient_partial_kernel[
        batch, out_channels, out_h, out_w, bias_num_chunks, bias_chunk_size
    ]()._call_with_pack_checked(
        ctx,
        bias_partial.gpu_ptr(),
        upper_grad.gpu_ptr(),
        Int64(bias_partial.num_elements()),
        grid_dim=ceildiv(bias_partial.num_elements(), _BLOCK),
        block_dim=min(bias_partial.num_elements(), _BLOCK),
    )
    _cached_sum_rows_kernel()._call_with_pack_checked(
        ctx,
        bias_grad.gpu_ptr(),
        bias_partial.gpu_ptr(),
        Int64(out_channels),
        Int64(bias_num_chunks),
        grid_dim=ceildiv(out_channels, _BLOCK),
        block_dim=min(out_channels, _BLOCK),
    )

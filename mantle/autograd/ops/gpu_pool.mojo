# ===----------------------------------------------------------------------=== #
# Mantle: GPU max pooling
# ===----------------------------------------------------------------------=== #
"""Device-resident NCHW MaxPool2d kernels."""
from std.math import ceildiv
from max.gpu import thread_idx, block_idx, block_dim
from std.ffi import _Global
from std.os import abort

from mantle import f32
from mantle.core.tensor import Tensor, _shared_device_context
from mantle.core.device import Device

comptime _BLOCK = 256


def _make_kernel_fn[
    declared_arg_types: TypeList[Trait=AnyType, ...],
    //,
    func: def(* args: * declared_arg_types) thin -> None,
]() -> type_of(_shared_device_context().compile_function[func]()):
    try:
        return _shared_device_context().compile_function[func]()
    except e:
        abort("Mantle: GPU pooling kernel compile failed: " + String(e))


def _maxpool_forward_kernel(
    output: Pointer[Scalar[f32], MutAnyOrigin],
    inputs: Pointer[Scalar[f32], MutAnyOrigin],
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
        var oy = i % out_w
        var ox = (i // out_w) % out_h
        var channel = (i // (out_h * out_w)) % channels
        var batch = i // (channels * out_h * out_w)
        var found = False
        var maximum: Scalar[f32] = 0.0
        for kx in range(Int(kh)):
            for ky in range(Int(kw)):
                var iy = ox * stride_h - pad_h + Int64(kx) * dilation_h
                var ix = oy * stride_w - pad_w + Int64(ky) * dilation_w
                if iy >= 0 and ix >= 0 and iy < in_h and ix < in_w:
                    var value = inputs.unsafe_load(
                        Int(
                            ((batch * channels + channel) * in_h + iy) * in_w
                            + ix
                        )
                    )
                    if not found or value > maximum:
                        maximum = value
                        found = True
        output.unsafe_store(Int(i), maximum)


comptime _maxpool_forward_kernel_global = _Global[
    "mantle_gpu_kernel_maxpool_forward",
    _make_kernel_fn[_maxpool_forward_kernel],
]


def _cached_maxpool_forward_kernel() raises -> (
    type_of(
        _shared_device_context().compile_function[_maxpool_forward_kernel]()
    )
):
    return _maxpool_forward_kernel_global.get_or_create_ptr()[
        unsafe_offset=0
    ].copy()


def _maxpool_backward_kernel(
    output: Pointer[Scalar[f32], MutAnyOrigin],
    upper_grad: Pointer[Scalar[f32], MutAnyOrigin],
    inputs: Pointer[Scalar[f32], MutAnyOrigin],
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
        var ix = i % in_w
        var iy = (i // in_w) % in_h
        var channel = (i // (in_h * in_w)) % channels
        var batch = i // (channels * in_h * in_w)
        var total: Scalar[f32] = 0.0
        # The overwhelmingly common CNN pool is a non-overlapping 2x2 (or
        # kxk) window.  Every input then has exactly one owning output, so
        # find that one window instead of re-scanning every output window for
        # every input element.  This changes the work from O(H²W²k²) to
        # O(HWk²), while retaining the generic overlap-safe path below.
        if (
            pad_h == 0
            and pad_w == 0
            and dilation_h == 1
            and dilation_w == 1
            and stride_h == kh
            and stride_w == kw
        ):
            var ox = iy // kh
            var oy = ix // kw
            if ox < out_h and oy < out_w:
                var selected: Int64 = -1
                var found = False
                var maximum: Scalar[f32] = 0.0
                for kx in range(Int(kh)):
                    for ky in range(Int(kw)):
                        var py = ox * stride_h + Int64(kx)
                        var px = oy * stride_w + Int64(ky)
                        if py >= 0 and px >= 0 and py < in_h and px < in_w:
                            var candidate = (
                                (batch * channels + channel) * in_h + py
                            ) * in_w + px
                            var value = inputs.unsafe_load(Int(candidate))
                            if not found or value > maximum:
                                maximum = value
                                selected = candidate
                                found = True
                if selected == i:
                    total = upper_grad.unsafe_load(
                        Int(
                            ((batch * channels + channel) * out_h + ox) * out_w
                            + oy
                        )
                    )
        else:
            for ox in range(Int(out_h)):
                for oy in range(Int(out_w)):
                    var selected: Int64 = -1
                    var found = False
                    var maximum: Scalar[f32] = 0.0
                    for kx in range(Int(kh)):
                        for ky in range(Int(kw)):
                            var py = (
                                Int64(ox) * stride_h
                                - pad_h
                                + Int64(kx) * dilation_h
                            )
                            var px = (
                                Int64(oy) * stride_w
                                - pad_w
                                + Int64(ky) * dilation_w
                            )
                            if py >= 0 and px >= 0 and py < in_h and px < in_w:
                                var candidate = (
                                    (batch * channels + channel) * in_h + py
                                ) * in_w + px
                                var value = inputs.unsafe_load(Int(candidate))
                                if not found or value > maximum:
                                    maximum = value
                                    selected = candidate
                                    found = True
                    if selected == i:
                        total += upper_grad.unsafe_load(
                            Int(
                                (
                                    (batch * channels + channel) * out_h
                                    + Int64(ox)
                                )
                                * out_w
                                + Int64(oy)
                            )
                        )
        output.unsafe_store(Int(i), total)


comptime _maxpool_backward_kernel_global = _Global[
    "mantle_gpu_kernel_maxpool_backward",
    _make_kernel_fn[_maxpool_backward_kernel],
]


def _cached_maxpool_backward_kernel() raises -> (
    type_of(
        _shared_device_context().compile_function[_maxpool_backward_kernel]()
    )
):
    return _maxpool_backward_kernel_global.get_or_create_ptr()[
        unsafe_offset=0
    ].copy()


def gpu_maxpool2d_forward[
    batch: Int,
    channels: Int,
    in_h: Int,
    in_w: Int,
    out_h: Int,
    out_w: Int,
    kh: Int,
    kw: Int,
    pad_h: Int,
    pad_w: Int,
    stride_h: Int,
    stride_w: Int,
    dilation_h: Int,
    dilation_w: Int,
](mut output: Tensor[f32, Device.gpu], inputs: Tensor[f32, Device.gpu]) raises:
    var ctx = output.gpu_context()
    _cached_maxpool_forward_kernel()._call_with_pack_checked(
        ctx,
        output.gpu_ptr(),
        inputs.gpu_ptr(),
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


def gpu_maxpool2d_backward[
    batch: Int,
    channels: Int,
    in_h: Int,
    in_w: Int,
    out_h: Int,
    out_w: Int,
    kh: Int,
    kw: Int,
    pad_h: Int,
    pad_w: Int,
    stride_h: Int,
    stride_w: Int,
    dilation_h: Int,
    dilation_w: Int,
](
    mut output: Tensor[f32, Device.gpu],
    upper_grad: Tensor[f32, Device.gpu],
    inputs: Tensor[f32, Device.gpu],
) raises:
    var ctx = output.gpu_context()
    _cached_maxpool_backward_kernel()._call_with_pack_checked(
        ctx,
        output.gpu_ptr(),
        upper_grad.gpu_ptr(),
        inputs.gpu_ptr(),
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

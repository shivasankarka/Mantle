# ===----------------------------------------------------------------------=== #
# Mantle: GPU elementwise kernels
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""GPU elementwise kernels (mantle.autograd.ops.gpu_elementwise)
------------------------------------------------
GPU kernels for ADD/SUB/MUL/DIV/RELU forward+backward (same-shape
operands), trailing-dim-broadcast bias-add (for Linear layers), RNG, and
the Adam optimizer step.
"""
from std.math import ceildiv, sqrt, exp, log, tanh, cos, pi, pow
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from std.ffi import _Global
from std.os import abort
from std.random.philox import Random
from std.random import random_ui64

from mantle import f32
from mantle.core.tensor import Tensor, TensorShape, _shared_device_context
from mantle.core.device import Device

comptime _BLOCK = 256


def _make_kernel_fn[
    declared_arg_types: TypeList[Trait=AnyType, ...],
    //,
    func: def(* args: * declared_arg_types) thin -> None,
]() -> type_of(_shared_device_context().compile_function[func]()):
    """A kernel's `DeviceFunction` (compiled pipeline state) is expensive to
    create and constant for the process, so each kernel gets exactly one,
    cached the same way `_shared_device_context` caches the `DeviceContext`
    itself — otherwise every single launch would recompile it."""
    try:
        return _shared_device_context().compile_function[func]()
    except e:
        abort("Mantle: GPU kernel compile failed: " + String(e))


def _unary_fwd_kernel[
    f: def(Scalar[f32]) thin -> Scalar[f32]
](
    res: Pointer[Scalar[f32], MutAnyOrigin],
    a: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    """Generic `res[i] = f(a[i])` — shared by every elementwise unary math
    op (sqrt/exp/log/gelu/...) that differs only in `f`."""
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        res.unsafe_store(Int(i), f(a.unsafe_load(Int(i))))


def _unary_bwd_kernel[
    df: def(Scalar[f32]) thin -> Scalar[f32]
](
    res: Pointer[Scalar[f32], MutAnyOrigin],
    a: Pointer[Scalar[f32], MutAnyOrigin],
    ug: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    """Generic `res[i] = ug[i] * df(a[i])` — the chain-rule backward shared
    by every unary math op whose derivative is itself a function of just
    the forward input."""
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        res.unsafe_store(
            Int(i), ug.unsafe_load(Int(i)) * df(a.unsafe_load(Int(i)))
        )


def _add_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    a: Pointer[Scalar[f32], MutAnyOrigin],
    b: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        res.unsafe_store(Int(i), a.unsafe_load(Int(i)) + b.unsafe_load(Int(i)))


comptime _add_kernel_global = _Global[
    "mantle_gpu_kernel_add", _make_kernel_fn[_add_kernel]
]


def _cached_add_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_add_kernel]())
):
    return _add_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def _sub_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    a: Pointer[Scalar[f32], MutAnyOrigin],
    b: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        res.unsafe_store(Int(i), a.unsafe_load(Int(i)) - b.unsafe_load(Int(i)))


comptime _sub_kernel_global = _Global[
    "mantle_gpu_kernel_sub", _make_kernel_fn[_sub_kernel]
]


def _cached_sub_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_sub_kernel]())
):
    return _sub_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def _mul_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    a: Pointer[Scalar[f32], MutAnyOrigin],
    b: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        res.unsafe_store(Int(i), a.unsafe_load(Int(i)) * b.unsafe_load(Int(i)))


comptime _mul_kernel_global = _Global[
    "mantle_gpu_kernel_mul", _make_kernel_fn[_mul_kernel]
]


def _cached_mul_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_mul_kernel]())
):
    return _mul_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def _div_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    a: Pointer[Scalar[f32], MutAnyOrigin],
    b: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        res.unsafe_store(Int(i), a.unsafe_load(Int(i)) / b.unsafe_load(Int(i)))


comptime _div_kernel_global = _Global[
    "mantle_gpu_kernel_div", _make_kernel_fn[_div_kernel]
]


def _cached_div_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_div_kernel]())
):
    return _div_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


# `mode=0` broadcasts a trailing vector, `mode=1` a trailing singleton, and
# `mode=2` a smaller right-aligned tensor across leading dimensions.
def _broadcast_binary_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    a: Pointer[Scalar[f32], MutAnyOrigin],
    b: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
    b_len: Int64,
    repeat: Int64,
    mode: Int64,
    op: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var bi = i % b_len if mode == 0 or mode == 2 else i // repeat
        var av = a.unsafe_load(Int(i))
        var bv = b.unsafe_load(Int(bi))
        if op == 0:
            res.unsafe_store(Int(i), av + bv)
        elif op == 1:
            res.unsafe_store(Int(i), av - bv)
        elif op == 2:
            res.unsafe_store(Int(i), av * bv)
        else:
            res.unsafe_store(Int(i), av / bv)


comptime _broadcast_binary_kernel_global = _Global[
    "mantle_gpu_kernel_broadcast_binary",
    _make_kernel_fn[_broadcast_binary_kernel],
]


def _cached_broadcast_binary_kernel() raises -> (
    type_of(
        _shared_device_context().compile_function[_broadcast_binary_kernel]()
    )
):
    return _broadcast_binary_kernel_global.get_or_create_ptr()[
        unsafe_offset=0
    ].copy()


def _broadcast_binary_bw_t1_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    ug: Pointer[Scalar[f32], MutAnyOrigin],
    a: Pointer[Scalar[f32], MutAnyOrigin],
    b: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
    b_len: Int64,
    repeat: Int64,
    mode: Int64,
    op: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var bi = i % b_len if mode == 0 or mode == 2 else i // repeat
        var g = ug.unsafe_load(Int(i))
        var bv = b.unsafe_load(Int(bi))
        if op == 0 or op == 1:
            res.unsafe_store(Int(i), g)
        elif op == 2:
            res.unsafe_store(Int(i), g * bv)
        else:
            res.unsafe_store(Int(i), g / bv)


comptime _broadcast_binary_bw_t1_kernel_global = _Global[
    "mantle_gpu_kernel_broadcast_binary_bw_t1",
    _make_kernel_fn[_broadcast_binary_bw_t1_kernel],
]


def _cached_broadcast_binary_bw_t1_kernel() raises -> (
    type_of(
        _shared_device_context().compile_function[
            _broadcast_binary_bw_t1_kernel
        ]()
    )
):
    return _broadcast_binary_bw_t1_kernel_global.get_or_create_ptr()[
        unsafe_offset=0
    ].copy()


def _broadcast_binary_bw_t2_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    ug: Pointer[Scalar[f32], MutAnyOrigin],
    a: Pointer[Scalar[f32], MutAnyOrigin],
    b: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
    b_len: Int64,
    repeat: Int64,
    mode: Int64,
    op: Int64,
):
    var bi = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if bi < b_len:
        var total: Scalar[f32] = 0
        if mode == 0 or mode == 2:
            for block in range(Int(n // b_len)):
                var i = Int64(block) * b_len + bi
                var g = ug.unsafe_load(Int(i))
                var av = a.unsafe_load(Int(i))
                var bv = b.unsafe_load(Int(bi))
                if op == 0:
                    total += g
                elif op == 1:
                    total -= g
                elif op == 2:
                    total += g * av
                else:
                    total -= g * av / (bv * bv)
        else:
            for offset in range(Int(repeat)):
                var i = bi * repeat + Int64(offset)
                var g = ug.unsafe_load(Int(i))
                var av = a.unsafe_load(Int(i))
                var bv = b.unsafe_load(Int(bi))
                if op == 0:
                    total += g
                elif op == 1:
                    total -= g
                elif op == 2:
                    total += g * av
                else:
                    total -= g * av / (bv * bv)
        res.unsafe_store(Int(bi), total)


comptime _broadcast_binary_bw_t2_kernel_global = _Global[
    "mantle_gpu_kernel_broadcast_binary_bw_t2",
    _make_kernel_fn[_broadcast_binary_bw_t2_kernel],
]


def _cached_broadcast_binary_bw_t2_kernel() raises -> (
    type_of(
        _shared_device_context().compile_function[
            _broadcast_binary_bw_t2_kernel
        ]()
    )
):
    return _broadcast_binary_bw_t2_kernel_global.get_or_create_ptr()[
        unsafe_offset=0
    ].copy()


# One thread handles one contiguous trailing row. Transformer reductions are
# short (head sequence length or model width), so this avoids temporary global
# buffers and, crucially, avoids a host round-trip for Softmax and LayerNorm.
# mode: 0=sum, 1=mean, 2=max.
def _reduce_last_kernel(
    dst: Pointer[Scalar[f32], MutAnyOrigin],
    src: Pointer[Scalar[f32], MutAnyOrigin],
    groups: Int64,
    axis_len: Int64,
    mode: Int64,
):
    var group = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if group < groups:
        var base = group * axis_len
        var result: Scalar[f32] = 0.0
        if mode == 2:
            result = src.unsafe_load(Int(base))
            for offset in range(1, Int(axis_len)):
                var value = src.unsafe_load(Int(base + Int64(offset)))
                if value > result:
                    result = value
        else:
            for offset in range(Int(axis_len)):
                result += src.unsafe_load(Int(base + Int64(offset)))
            if mode == 1:
                result /= Scalar[f32](axis_len)
        dst.unsafe_store(Int(group), result)


comptime _reduce_last_kernel_global = _Global[
    "mantle_gpu_kernel_reduce_last", _make_kernel_fn[_reduce_last_kernel]
]


def _cached_reduce_last_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_reduce_last_kernel]())
):
    return _reduce_last_kernel_global.get_or_create_ptr()[
        unsafe_offset=0
    ].copy()


def _reduce_last_bw_kernel(
    dst: Pointer[Scalar[f32], MutAnyOrigin],
    ug: Pointer[Scalar[f32], MutAnyOrigin],
    src: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
    axis_len: Int64,
    mode: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var group = i // axis_len
        var gradient = ug.unsafe_load(Int(group))
        if mode == 0:
            dst.unsafe_store(Int(i), gradient)
        elif mode == 1:
            dst.unsafe_store(Int(i), gradient / Scalar[f32](axis_len))
        else:
            var base = group * axis_len
            var maximum = src.unsafe_load(Int(base))
            var matches: Scalar[f32] = 0.0
            for offset in range(Int(axis_len)):
                var value = src.unsafe_load(Int(base + Int64(offset)))
                if value > maximum:
                    maximum = value
            for offset in range(Int(axis_len)):
                if src.unsafe_load(Int(base + Int64(offset))) == maximum:
                    matches += 1.0
            if src.unsafe_load(Int(i)) == maximum:
                dst.unsafe_store(Int(i), gradient / matches)
            else:
                dst.unsafe_store(Int(i), 0.0)


comptime _reduce_last_bw_kernel_global = _Global[
    "mantle_gpu_kernel_reduce_last_bw", _make_kernel_fn[_reduce_last_bw_kernel]
]


def _cached_reduce_last_bw_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_reduce_last_bw_kernel]())
):
    return _reduce_last_bw_kernel_global.get_or_create_ptr()[
        unsafe_offset=0
    ].copy()


def gpu_reduce_last_forward(
    mut dst: Tensor[f32, Device.gpu],
    src: Tensor[f32, Device.gpu],
    axis_len: Int,
    mode: Int,
) raises:
    var ctx = dst.gpu_context()
    var groups = src.num_elements() // axis_len
    _cached_reduce_last_kernel()._call_with_pack_checked(
        ctx,
        dst.gpu_ptr(),
        src.gpu_ptr(),
        Int64(groups),
        Int64(axis_len),
        Int64(mode),
        grid_dim=(ceildiv(groups, _BLOCK),),
        block_dim=(_BLOCK,),
    )


def gpu_reduce_last_backward_into(
    mut dst: Tensor[f32, Device.gpu],
    ug: Tensor[f32, Device.gpu],
    src: Tensor[f32, Device.gpu],
    axis_len: Int,
    mode: Int,
) raises:
    var ctx = dst.gpu_context()
    var n = src.num_elements()
    _cached_reduce_last_bw_kernel()._call_with_pack_checked(
        ctx,
        dst.gpu_ptr(),
        ug.gpu_ptr(),
        src.gpu_ptr(),
        Int64(n),
        Int64(axis_len),
        Int64(mode),
        grid_dim=(ceildiv(n, _BLOCK),),
        block_dim=(_BLOCK,),
    )


def gpu_reduce_last_backward(
    ug: Tensor[f32, Device.gpu],
    src: Tensor[f32, Device.gpu],
    axis_len: Int,
    mode: Int,
) raises -> Tensor[f32, Device.gpu]:
    var dst = Tensor[f32, Device.gpu](src.shape(), uninitialized=True)
    gpu_reduce_last_backward_into(dst, ug, src, axis_len, mode)
    return dst^


def _gather_kernel(
    dst: Pointer[Scalar[f32], MutAnyOrigin],
    table: Pointer[Scalar[f32], MutAnyOrigin],
    indices: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
    width: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var index_offset = i // width
        var row = Int(indices.unsafe_load(Int(index_offset)))
        dst.unsafe_store(
            Int(i), table.unsafe_load(row * Int(width) + Int(i % width))
        )


comptime _gather_kernel_global = _Global[
    "mantle_gpu_kernel_gather", _make_kernel_fn[_gather_kernel]
]


def _cached_gather_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_gather_kernel]())
):
    return _gather_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def _gather_bw_kernel(
    dst: Pointer[Scalar[f32], MutAnyOrigin],
    ug: Pointer[Scalar[f32], MutAnyOrigin],
    indices: Pointer[Scalar[f32], MutAnyOrigin],
    table_n: Int64,
    index_n: Int64,
    width: Int64,
):
    """Gather's table gradient without atomics: one thread per table cell.

    Embedding vocabularies can contain repeated token ids. Scanning the small
    index batch per table cell is deterministic and avoids racing atomic adds.
    """
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < table_n:
        var row = i // width
        var column = i % width
        var total: Scalar[f32] = 0.0
        for index_offset in range(Int(index_n)):
            if Int(indices.unsafe_load(index_offset)) == Int(row):
                total += ug.unsafe_load(index_offset * Int(width) + Int(column))
        dst.unsafe_store(Int(i), total)


comptime _gather_bw_kernel_global = _Global[
    "mantle_gpu_kernel_gather_bw", _make_kernel_fn[_gather_bw_kernel]
]


def _cached_gather_bw_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_gather_bw_kernel]())
):
    return _gather_bw_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def gpu_gather_forward(
    mut dst: Tensor[f32, Device.gpu],
    table: Tensor[f32, Device.gpu],
    indices: Tensor[f32, Device.gpu],
    width: Int,
) raises:
    var ctx = dst.gpu_context()
    var n = dst.num_elements()
    _cached_gather_kernel()._call_with_pack_checked(
        ctx,
        dst.gpu_ptr(),
        table.gpu_ptr(),
        indices.gpu_ptr(),
        Int64(n),
        Int64(width),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_gather_backward(
    ug: Tensor[f32, Device.gpu],
    table: Tensor[f32, Device.gpu],
    indices: Tensor[f32, Device.gpu],
    width: Int,
) raises -> Tensor[f32, Device.gpu]:
    var dst = Tensor[f32, Device.gpu](table.shape(), uninitialized=True)
    var ctx = dst.gpu_context()
    var n = dst.num_elements()
    _cached_gather_bw_kernel()._call_with_pack_checked(
        ctx,
        dst.gpu_ptr(),
        ug.gpu_ptr(),
        indices.gpu_ptr(),
        Int64(n),
        Int64(indices.num_elements()),
        Int64(width),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )
    return dst^


def _gelu_f(x: Scalar[f32]) -> Scalar[f32]:
    var u = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + tanh(u))


def _gelu_df(x: Scalar[f32]) -> Scalar[f32]:
    var c0: Scalar[f32] = 0.7978845608028654
    var c1: Scalar[f32] = 0.044715
    var t = tanh(c0 * (x + c1 * x * x * x))
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * c0 * (
        1.0 + 3.0 * c1 * x * x
    )


comptime _gelu_kernel_global = _Global[
    "mantle_gpu_kernel_gelu", _make_kernel_fn[_unary_fwd_kernel[_gelu_f]]
]


def _cached_gelu_kernel() raises -> (
    type_of(
        _shared_device_context().compile_function[_unary_fwd_kernel[_gelu_f]]()
    )
):
    return _gelu_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


comptime _gelu_bw_kernel_global = _Global[
    "mantle_gpu_kernel_gelu_bw", _make_kernel_fn[_unary_bwd_kernel[_gelu_df]]
]


def _cached_gelu_bw_kernel() raises -> (
    type_of(
        _shared_device_context().compile_function[_unary_bwd_kernel[_gelu_df]]()
    )
):
    return _gelu_bw_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def gpu_gelu_forward(
    mut res: Tensor[f32, Device.gpu], t1: Tensor[f32, Device.gpu]
) raises:
    var ctx = res.gpu_context()
    var n = res.num_elements()
    _cached_gelu_kernel()._call_with_pack_checked(
        ctx,
        res.gpu_ptr(),
        t1.gpu_ptr(),
        Int64(n),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_gelu_backward_into(
    mut grad: Tensor[f32, Device.gpu],
    ug: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
) raises:
    var ctx = grad.gpu_context()
    var n = grad.num_elements()
    _cached_gelu_bw_kernel()._call_with_pack_checked(
        ctx,
        grad.gpu_ptr(),
        t1.gpu_ptr(),
        ug.gpu_ptr(),
        Int64(n),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_gelu_backward(
    ug: Tensor[f32, Device.gpu], t1: Tensor[f32, Device.gpu]
) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](t1.shape(), uninitialized=True)
    gpu_gelu_backward_into(res_grad, ug, t1)
    return res_grad^


# LayerNorm is row-wise over the contiguous last axis. One thread owns one
# row, which keeps the mean/variance and input-gradient reductions local and
# avoids materializing the composite graph's mean, variance, and normalized
# activation tensors.
def _layernorm_forward_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    src: Pointer[Scalar[f32], MutAnyOrigin],
    gamma: Pointer[Scalar[f32], MutAnyOrigin],
    beta: Pointer[Scalar[f32], MutAnyOrigin],
    groups: Int64,
    width: Int64,
    epsilon: Scalar[f32],
):
    var group = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if group < groups:
        var base = group * width
        var mean: Scalar[f32] = 0.0
        for j in range(Int(width)):
            mean += src.unsafe_load(Int(base + Int64(j)))
        mean /= Scalar[f32](width)

        var variance: Scalar[f32] = 0.0
        for j in range(Int(width)):
            var delta = src.unsafe_load(Int(base + Int64(j))) - mean
            variance += delta * delta
        var inv_std = 1.0 / sqrt(variance / Scalar[f32](width) + epsilon)

        for j in range(Int(width)):
            var xhat = (src.unsafe_load(Int(base + Int64(j))) - mean) * inv_std
            res.unsafe_store(
                Int(base + Int64(j)),
                xhat * gamma.unsafe_load(j) + beta.unsafe_load(j),
            )


comptime _layernorm_forward_kernel_global = _Global[
    "mantle_gpu_kernel_layernorm_forward",
    _make_kernel_fn[_layernorm_forward_kernel],
]


def _cached_layernorm_forward_kernel() raises -> (
    type_of(
        _shared_device_context().compile_function[_layernorm_forward_kernel]()
    )
):
    return _layernorm_forward_kernel_global.get_or_create_ptr()[
        unsafe_offset=0
    ].copy()


def _layernorm_input_backward_kernel(
    dst: Pointer[Scalar[f32], MutAnyOrigin],
    ug: Pointer[Scalar[f32], MutAnyOrigin],
    src: Pointer[Scalar[f32], MutAnyOrigin],
    gamma: Pointer[Scalar[f32], MutAnyOrigin],
    groups: Int64,
    width: Int64,
    epsilon: Scalar[f32],
):
    var group = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if group < groups:
        var base = group * width
        var mean: Scalar[f32] = 0.0
        for j in range(Int(width)):
            mean += src.unsafe_load(Int(base + Int64(j)))
        mean /= Scalar[f32](width)

        var variance: Scalar[f32] = 0.0
        for j in range(Int(width)):
            var delta = src.unsafe_load(Int(base + Int64(j))) - mean
            variance += delta * delta
        var inv_std = 1.0 / sqrt(variance / Scalar[f32](width) + epsilon)

        var sum_dy: Scalar[f32] = 0.0
        var sum_dy_xhat: Scalar[f32] = 0.0
        for j in range(Int(width)):
            var dy = ug.unsafe_load(Int(base + Int64(j))) * gamma.unsafe_load(j)
            var xhat = (src.unsafe_load(Int(base + Int64(j))) - mean) * inv_std
            sum_dy += dy
            sum_dy_xhat += dy * xhat

        for j in range(Int(width)):
            var dy = ug.unsafe_load(Int(base + Int64(j))) * gamma.unsafe_load(j)
            var xhat = (src.unsafe_load(Int(base + Int64(j))) - mean) * inv_std
            dst.unsafe_store(
                Int(base + Int64(j)),
                inv_std
                * (
                    dy
                    - sum_dy / Scalar[f32](width)
                    - xhat * sum_dy_xhat / Scalar[f32](width)
                ),
            )


comptime _layernorm_input_backward_kernel_global = _Global[
    "mantle_gpu_kernel_layernorm_input_backward",
    _make_kernel_fn[_layernorm_input_backward_kernel],
]


def _cached_layernorm_input_backward_kernel() raises -> (
    type_of(
        _shared_device_context().compile_function[
            _layernorm_input_backward_kernel
        ]()
    )
):
    return _layernorm_input_backward_kernel_global.get_or_create_ptr()[
        unsafe_offset=0
    ].copy()


def _layernorm_affine_backward_kernel(
    grad: Pointer[Scalar[f32], MutAnyOrigin],
    ug: Pointer[Scalar[f32], MutAnyOrigin],
    src: Pointer[Scalar[f32], MutAnyOrigin],
    groups: Int64,
    width: Int64,
    epsilon: Scalar[f32],
    affine_id: Int64,
):
    var j = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if j < width:
        var gamma_sum: Scalar[f32] = 0.0
        var beta_sum: Scalar[f32] = 0.0
        for group in range(Int(groups)):
            var base = Int64(group) * width
            var mean: Scalar[f32] = 0.0
            for k in range(Int(width)):
                mean += src.unsafe_load(Int(base + Int64(k)))
            mean /= Scalar[f32](width)
            var variance: Scalar[f32] = 0.0
            for k in range(Int(width)):
                var delta = src.unsafe_load(Int(base + Int64(k))) - mean
                variance += delta * delta
            var xhat = (src.unsafe_load(Int(base + j)) - mean) / sqrt(
                variance / Scalar[f32](width) + epsilon
            )
            var upper = ug.unsafe_load(Int(base + j))
            gamma_sum += upper * xhat
            beta_sum += upper
        grad.unsafe_store(Int(j), gamma_sum if affine_id == 0 else beta_sum)


comptime _layernorm_affine_backward_kernel_global = _Global[
    "mantle_gpu_kernel_layernorm_affine_backward",
    _make_kernel_fn[_layernorm_affine_backward_kernel],
]


def _cached_layernorm_affine_backward_kernel() raises -> (
    type_of(
        _shared_device_context().compile_function[
            _layernorm_affine_backward_kernel
        ]()
    )
):
    return _layernorm_affine_backward_kernel_global.get_or_create_ptr()[
        unsafe_offset=0
    ].copy()


def gpu_layernorm_forward(
    mut res: Tensor[f32, Device.gpu],
    src: Tensor[f32, Device.gpu],
    gamma: Tensor[f32, Device.gpu],
    beta: Tensor[f32, Device.gpu],
    epsilon: Scalar[f32],
) raises:
    var ctx = res.gpu_context()
    var width = gamma.num_elements()
    var groups = res.num_elements() // width
    _cached_layernorm_forward_kernel()._call_with_pack_checked(
        ctx,
        res.gpu_ptr(),
        src.gpu_ptr(),
        gamma.gpu_ptr(),
        beta.gpu_ptr(),
        Int64(groups),
        Int64(width),
        epsilon,
        grid_dim=ceildiv(groups, _BLOCK),
        block_dim=min(groups, _BLOCK),
    )


def gpu_layernorm_input_backward(
    mut grad: Tensor[f32, Device.gpu],
    ug: Tensor[f32, Device.gpu],
    src: Tensor[f32, Device.gpu],
    gamma: Tensor[f32, Device.gpu],
    epsilon: Scalar[f32],
) raises:
    var ctx = grad.gpu_context()
    var width = gamma.num_elements()
    var groups = grad.num_elements() // width
    _cached_layernorm_input_backward_kernel()._call_with_pack_checked(
        ctx,
        grad.gpu_ptr(),
        ug.gpu_ptr(),
        src.gpu_ptr(),
        gamma.gpu_ptr(),
        Int64(groups),
        Int64(width),
        epsilon,
        grid_dim=ceildiv(groups, _BLOCK),
        block_dim=min(groups, _BLOCK),
    )


def gpu_layernorm_affine_backward(
    mut grad: Tensor[f32, Device.gpu],
    ug: Tensor[f32, Device.gpu],
    src: Tensor[f32, Device.gpu],
    epsilon: Scalar[f32],
    affine_id: Int64,
) raises:
    var ctx = grad.gpu_context()
    var width = grad.num_elements()
    var groups = ug.num_elements() // width
    _cached_layernorm_affine_backward_kernel()._call_with_pack_checked(
        ctx,
        grad.gpu_ptr(),
        ug.gpu_ptr(),
        src.gpu_ptr(),
        Int64(groups),
        Int64(width),
        epsilon,
        affine_id,
        grid_dim=ceildiv(width, _BLOCK),
        block_dim=min(width, _BLOCK),
    )


def _dropout_kernel(
    dst: Pointer[Scalar[f32], MutAnyOrigin],
    src: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
    keep: Scalar[f32],
    scale: Scalar[f32],
    seed: UInt64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var gen = Random(seed=seed, offset=UInt64(i))
        var random = gen.step_uniform()
        if random[0] < keep:
            dst.unsafe_store(Int(i), src.unsafe_load(Int(i)) * scale)
        else:
            dst.unsafe_store(Int(i), 0.0)


comptime _dropout_kernel_global = _Global[
    "mantle_gpu_kernel_dropout", _make_kernel_fn[_dropout_kernel]
]


def _cached_dropout_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_dropout_kernel]())
):
    return _dropout_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def gpu_dropout_forward(
    mut res: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    p: Scalar[f32],
    seed: UInt64,
    training: Bool,
) raises:
    if not training:
        res.copy_from(t1)
        return
    var ctx = res.gpu_context()
    var n = res.num_elements()
    _cached_dropout_kernel()._call_with_pack_checked(
        ctx,
        res.gpu_ptr(),
        t1.gpu_ptr(),
        Int64(n),
        1.0 - p,
        1.0 / (1.0 - p),
        seed,
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_dropout_backward(
    ug: Tensor[f32, Device.gpu],
    p: Scalar[f32],
    seed: UInt64,
    training: Bool,
) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](ug.shape(), uninitialized=True)
    gpu_dropout_backward_into(res_grad, ug, p, seed, training)
    return res_grad^


def gpu_dropout_backward_into(
    mut grad: Tensor[f32, Device.gpu],
    ug: Tensor[f32, Device.gpu],
    p: Scalar[f32],
    seed: UInt64,
    training: Bool,
) raises:
    gpu_dropout_forward(grad, ug, p, seed, training)


def _accumulate_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    other: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        res.unsafe_store(
            Int(i), res.unsafe_load(Int(i)) + other.unsafe_load(Int(i))
        )


comptime _accumulate_kernel_global = _Global[
    "mantle_gpu_kernel_accumulate", _make_kernel_fn[_accumulate_kernel]
]


def _cached_accumulate_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_accumulate_kernel]())
):
    return _accumulate_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def _neg_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    a: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        res.unsafe_store(Int(i), -a.unsafe_load(Int(i)))


comptime _neg_kernel_global = _Global[
    "mantle_gpu_kernel_neg", _make_kernel_fn[_neg_kernel]
]


def _cached_neg_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_neg_kernel]())
):
    return _neg_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def _relu_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    a: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var v = a.unsafe_load(Int(i))
        res.unsafe_store(Int(i), v if v > 0 else Scalar[f32](0))


comptime _relu_kernel_global = _Global[
    "mantle_gpu_kernel_relu", _make_kernel_fn[_relu_kernel]
]


def _cached_relu_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_relu_kernel]())
):
    return _relu_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def _relu_bw_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    t1: Pointer[Scalar[f32], MutAnyOrigin],
    ug: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var mask = Scalar[f32](1) if t1.unsafe_load(Int(i)) > 0 else Scalar[
            f32
        ](0)
        res.unsafe_store(Int(i), mask * ug.unsafe_load(Int(i)))


comptime _relu_bw_kernel_global = _Global[
    "mantle_gpu_kernel_relu_bw", _make_kernel_fn[_relu_bw_kernel]
]


def _cached_relu_bw_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_relu_bw_kernel]())
):
    return _relu_bw_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def _div_bw_t2_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    t1: Pointer[Scalar[f32], MutAnyOrigin],
    t2: Pointer[Scalar[f32], MutAnyOrigin],
    ug: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var t2v = t2.unsafe_load(Int(i))
        res.unsafe_store(
            Int(i),
            -t1.unsafe_load(Int(i)) / (t2v * t2v) * ug.unsafe_load(Int(i)),
        )


comptime _div_bw_t2_kernel_global = _Global[
    "mantle_gpu_kernel_div_bw_t2", _make_kernel_fn[_div_bw_t2_kernel]
]


def _cached_div_bw_t2_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_div_bw_t2_kernel]())
):
    return _div_bw_t2_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def _add_bias_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    a: Pointer[Scalar[f32], MutAnyOrigin],
    bias: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
    total: Int64,
):
    """res[i] = a[i] + bias[i % n], for a trailing-dim broadcast bias."""
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < total:
        res.unsafe_store(
            Int(i), a.unsafe_load(Int(i)) + bias.unsafe_load(Int(i % n))
        )


comptime _add_bias_kernel_global = _Global[
    "mantle_gpu_kernel_add_bias", _make_kernel_fn[_add_bias_kernel]
]


def _cached_add_bias_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_add_bias_kernel]())
):
    return _add_bias_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def _bias_grad_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    ug: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
    outer: Int64,
):
    """res[j] = sum over the broadcast dim of ug[..., j]."""
    var j = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if j < n:
        var s: Scalar[f32] = 0
        var idx = j
        for _ in range(outer):
            s += ug.unsafe_load(Int(idx))
            idx += n
        res.unsafe_store(Int(j), s)


comptime _bias_grad_kernel_global = _Global[
    "mantle_gpu_kernel_bias_grad", _make_kernel_fn[_bias_grad_kernel]
]


def _cached_bias_grad_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_bias_grad_kernel]())
):
    return _bias_grad_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def _pow_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    a: Pointer[Scalar[f32], MutAnyOrigin],
    exponent: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    """`exponent` is read from its own device buffer (not a host scalar
    argument) so this needs no host round-trip: `exponent` is itself a
    graph tensor (POW's second operand), and reading it on the host would
    reintroduce the exact sync we're trying to eliminate."""
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var e = Int(exponent.unsafe_load(0))
        res.unsafe_store(Int(i), pow(a.unsafe_load(Int(i)), e))


comptime _pow_kernel_global = _Global[
    "mantle_gpu_kernel_pow", _make_kernel_fn[_pow_kernel]
]


def _cached_pow_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_pow_kernel]())
):
    return _pow_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def _pow_bw_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    a: Pointer[Scalar[f32], MutAnyOrigin],
    ug: Pointer[Scalar[f32], MutAnyOrigin],
    exponent: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var ev = exponent.unsafe_load(0)
        var e = Int(ev)
        res.unsafe_store(
            Int(i),
            ev * pow(a.unsafe_load(Int(i)), e - 1) * ug.unsafe_load(Int(i)),
        )


comptime _pow_bw_kernel_global = _Global[
    "mantle_gpu_kernel_pow_bw", _make_kernel_fn[_pow_bw_kernel]
]


def _cached_pow_bw_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_pow_bw_kernel]())
):
    return _pow_bw_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def _sqrt_f(x: Scalar[f32]) -> Scalar[f32]:
    return sqrt(x)


def _sqrt_df(x: Scalar[f32]) -> Scalar[f32]:
    return 1 / (2 * sqrt(x))


comptime _sqrt_kernel_global = _Global[
    "mantle_gpu_kernel_sqrt", _make_kernel_fn[_unary_fwd_kernel[_sqrt_f]]
]


def _cached_sqrt_kernel() raises -> (
    type_of(
        _shared_device_context().compile_function[_unary_fwd_kernel[_sqrt_f]]()
    )
):
    return _sqrt_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


comptime _sqrt_bw_kernel_global = _Global[
    "mantle_gpu_kernel_sqrt_bw", _make_kernel_fn[_unary_bwd_kernel[_sqrt_df]]
]


def _cached_sqrt_bw_kernel() raises -> (
    type_of(
        _shared_device_context().compile_function[_unary_bwd_kernel[_sqrt_df]]()
    )
):
    return _sqrt_bw_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def _exp_f(x: Scalar[f32]) -> Scalar[f32]:
    return exp(x)


comptime _exp_kernel_global = _Global[
    "mantle_gpu_kernel_exp", _make_kernel_fn[_unary_fwd_kernel[_exp_f]]
]


def _cached_exp_kernel() raises -> (
    type_of(
        _shared_device_context().compile_function[_unary_fwd_kernel[_exp_f]]()
    )
):
    return _exp_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


comptime _exp_bw_kernel_global = _Global[
    # exp's own derivative is itself, so the forward math fn doubles as the
    # backward one.
    "mantle_gpu_kernel_exp_bw",
    _make_kernel_fn[_unary_bwd_kernel[_exp_f]],
]


def _cached_exp_bw_kernel() raises -> (
    type_of(
        _shared_device_context().compile_function[_unary_bwd_kernel[_exp_f]]()
    )
):
    return _exp_bw_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def _log_f(x: Scalar[f32]) -> Scalar[f32]:
    return log(x)


def _log_df(x: Scalar[f32]) -> Scalar[f32]:
    return 1 / x


comptime _log_kernel_global = _Global[
    "mantle_gpu_kernel_log", _make_kernel_fn[_unary_fwd_kernel[_log_f]]
]


def _cached_log_kernel() raises -> (
    type_of(
        _shared_device_context().compile_function[_unary_fwd_kernel[_log_f]]()
    )
):
    return _log_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


comptime _log_bw_kernel_global = _Global[
    "mantle_gpu_kernel_log_bw", _make_kernel_fn[_unary_bwd_kernel[_log_df]]
]


def _cached_log_bw_kernel() raises -> (
    type_of(
        _shared_device_context().compile_function[_unary_bwd_kernel[_log_df]]()
    )
):
    return _log_bw_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def _mean_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    a: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    """Single-thread full reduction: correct for any n, only fast for the
    small loss-sized tensors this is meant for (a handful of thousand
    elements) — not a general large-tensor reduction kernel."""
    var s: Scalar[f32] = 0
    for i in range(Int(n)):
        s += a.unsafe_load(i)
    res.unsafe_store(0, s / Scalar[f32](n))


comptime _mean_kernel_global = _Global[
    "mantle_gpu_kernel_mean", _make_kernel_fn[_mean_kernel]
]


def _cached_mean_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_mean_kernel]())
):
    return _mean_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def _mean_bw_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    ug: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        res.unsafe_store(Int(i), ug.unsafe_load(0) / Scalar[f32](n))


comptime _mean_bw_kernel_global = _Global[
    "mantle_gpu_kernel_mean_bw", _make_kernel_fn[_mean_bw_kernel]
]


def _cached_mean_bw_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_mean_bw_kernel]())
):
    return _mean_bw_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def _sum_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    a: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    """Single-thread full reduction, same shape as `_mean_kernel` but
    without the final divide-by-n — see its docstring for the scale this is
    meant for."""
    var s: Scalar[f32] = 0
    for i in range(Int(n)):
        s += a.unsafe_load(i)
    res.unsafe_store(0, s)


comptime _sum_kernel_global = _Global[
    "mantle_gpu_kernel_sum", _make_kernel_fn[_sum_kernel]
]


def _cached_sum_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_sum_kernel]())
):
    return _sum_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def _sum_bw_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    ug: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        res.unsafe_store(Int(i), ug.unsafe_load(0))


comptime _sum_bw_kernel_global = _Global[
    "mantle_gpu_kernel_sum_bw", _make_kernel_fn[_sum_bw_kernel]
]


def _cached_sum_bw_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_sum_bw_kernel]())
):
    return _sum_bw_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def gpu_pow_forward(
    mut res: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    exponent: Tensor[f32, Device.gpu],
) raises:
    var ctx = res.gpu_context()
    var n = res.num_elements()
    _cached_pow_kernel()._call_with_pack_checked(
        ctx,
        res.gpu_ptr(),
        t1.gpu_ptr(),
        exponent.gpu_ptr(),
        Int64(n),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_pow_backward_into(
    mut grad: Tensor[f32, Device.gpu],
    ug: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    exponent: Tensor[f32, Device.gpu],
) raises:
    var ctx = grad.gpu_context()
    var n = grad.num_elements()
    _cached_pow_bw_kernel()._call_with_pack_checked(
        ctx,
        grad.gpu_ptr(),
        t1.gpu_ptr(),
        ug.gpu_ptr(),
        exponent.gpu_ptr(),
        Int64(n),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_pow_backward(
    ug: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    exponent: Tensor[f32, Device.gpu],
) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](t1.shape(), uninitialized=True)
    gpu_pow_backward_into(res_grad, ug, t1, exponent)
    return res_grad^


def gpu_sqrt_forward(
    mut res: Tensor[f32, Device.gpu], t1: Tensor[f32, Device.gpu]
) raises:
    var ctx = res.gpu_context()
    var n = res.num_elements()
    _cached_sqrt_kernel()._call_with_pack_checked(
        ctx,
        res.gpu_ptr(),
        t1.gpu_ptr(),
        Int64(n),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_sqrt_backward_into(
    mut grad: Tensor[f32, Device.gpu],
    ug: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
) raises:
    var ctx = grad.gpu_context()
    var n = grad.num_elements()
    _cached_sqrt_bw_kernel()._call_with_pack_checked(
        ctx,
        grad.gpu_ptr(),
        t1.gpu_ptr(),
        ug.gpu_ptr(),
        Int64(n),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_sqrt_backward(
    ug: Tensor[f32, Device.gpu], t1: Tensor[f32, Device.gpu]
) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](t1.shape(), uninitialized=True)
    gpu_sqrt_backward_into(res_grad, ug, t1)
    return res_grad^


def gpu_exp_forward(
    mut res: Tensor[f32, Device.gpu], t1: Tensor[f32, Device.gpu]
) raises:
    var ctx = res.gpu_context()
    var n = res.num_elements()
    _cached_exp_kernel()._call_with_pack_checked(
        ctx,
        res.gpu_ptr(),
        t1.gpu_ptr(),
        Int64(n),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_exp_backward_into(
    mut grad: Tensor[f32, Device.gpu],
    ug: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
) raises:
    var ctx = grad.gpu_context()
    var n = grad.num_elements()
    _cached_exp_bw_kernel()._call_with_pack_checked(
        ctx,
        grad.gpu_ptr(),
        t1.gpu_ptr(),
        ug.gpu_ptr(),
        Int64(n),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_exp_backward(
    ug: Tensor[f32, Device.gpu], t1: Tensor[f32, Device.gpu]
) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](t1.shape(), uninitialized=True)
    gpu_exp_backward_into(res_grad, ug, t1)
    return res_grad^


def gpu_log_forward(
    mut res: Tensor[f32, Device.gpu], t1: Tensor[f32, Device.gpu]
) raises:
    var ctx = res.gpu_context()
    var n = res.num_elements()
    _cached_log_kernel()._call_with_pack_checked(
        ctx,
        res.gpu_ptr(),
        t1.gpu_ptr(),
        Int64(n),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_log_backward_into(
    mut grad: Tensor[f32, Device.gpu],
    ug: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
) raises:
    var ctx = grad.gpu_context()
    var n = grad.num_elements()
    _cached_log_bw_kernel()._call_with_pack_checked(
        ctx,
        grad.gpu_ptr(),
        t1.gpu_ptr(),
        ug.gpu_ptr(),
        Int64(n),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_log_backward(
    ug: Tensor[f32, Device.gpu], t1: Tensor[f32, Device.gpu]
) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](t1.shape(), uninitialized=True)
    gpu_log_backward_into(res_grad, ug, t1)
    return res_grad^


def gpu_mean_forward(
    mut res: Tensor[f32, Device.gpu], t1: Tensor[f32, Device.gpu]
) raises:
    var ctx = res.gpu_context()
    var n = t1.num_elements()
    _cached_mean_kernel()._call_with_pack_checked(
        ctx,
        res.gpu_ptr(),
        t1.gpu_ptr(),
        Int64(n),
        grid_dim=1,
        block_dim=1,
    )


def gpu_mean_backward_into(
    mut grad: Tensor[f32, Device.gpu], ug: Tensor[f32, Device.gpu]
) raises:
    var ctx = grad.gpu_context()
    var n = grad.num_elements()
    _cached_mean_bw_kernel()._call_with_pack_checked(
        ctx,
        grad.gpu_ptr(),
        ug.gpu_ptr(),
        Int64(n),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_mean_backward(
    ug: Tensor[f32, Device.gpu], t_shape: TensorShape
) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](t_shape, uninitialized=True)
    gpu_mean_backward_into(res_grad, ug)
    return res_grad^


def gpu_sum_forward(
    mut res: Tensor[f32, Device.gpu], t1: Tensor[f32, Device.gpu]
) raises:
    var ctx = res.gpu_context()
    var n = t1.num_elements()
    _cached_sum_kernel()._call_with_pack_checked(
        ctx,
        res.gpu_ptr(),
        t1.gpu_ptr(),
        Int64(n),
        grid_dim=1,
        block_dim=1,
    )


def gpu_sum_backward_into(
    mut grad: Tensor[f32, Device.gpu], ug: Tensor[f32, Device.gpu]
) raises:
    var ctx = grad.gpu_context()
    var n = grad.num_elements()
    _cached_sum_bw_kernel()._call_with_pack_checked(
        ctx,
        grad.gpu_ptr(),
        ug.gpu_ptr(),
        Int64(n),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_sum_backward(
    ug: Tensor[f32, Device.gpu], t_shape: TensorShape
) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](t_shape, uninitialized=True)
    gpu_sum_backward_into(res_grad, ug)
    return res_grad^


def _strided_block_copy_kernel(
    dst: Pointer[Scalar[f32], MutAnyOrigin],
    src: Pointer[Scalar[f32], MutAnyOrigin],
    count: Int64,
    src_chunk_stride: Int64,
    dst_chunk_stride: Int64,
    src_offset: Int64,
    dst_offset: Int64,
    total: Int64,
):
    """dst[dst_offset + c*dst_chunk_stride + e] = src[src_offset +
    c*src_chunk_stride + e], for c in [0, total/count) and e in [0, count) —
    the CONCAT/SPLIT chunked copy pattern (see dynamics.mojo), done as one
    kernel launch instead of `total/count` separate device memcpys."""
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < total:
        var c = i // count
        var e = i % count
        var src_idx = src_offset + c * src_chunk_stride + e
        var dst_idx = dst_offset + c * dst_chunk_stride + e
        dst.unsafe_store(Int(dst_idx), src.unsafe_load(Int(src_idx)))


comptime _strided_block_copy_kernel_global = _Global[
    "mantle_gpu_kernel_strided_block_copy",
    _make_kernel_fn[_strided_block_copy_kernel],
]


def _cached_strided_block_copy_kernel() raises -> (
    type_of(
        _shared_device_context().compile_function[
            _strided_block_copy_kernel
        ]()
    )
):
    return _strided_block_copy_kernel_global.get_or_create_ptr()[
        unsafe_offset=0
    ].copy()


def gpu_strided_block_copy(
    mut dst: Tensor[f32, Device.gpu],
    src: Tensor[f32, Device.gpu],
    n_chunks: Int,
    count: Int,
    src_chunk_stride: Int,
    dst_chunk_stride: Int,
    src_offset: Int,
    dst_offset: Int,
) raises:
    var ctx = dst.gpu_context()
    var total = n_chunks * count
    _cached_strided_block_copy_kernel()._call_with_pack_checked(
        ctx,
        dst.gpu_ptr(),
        src.gpu_ptr(),
        Int64(count),
        Int64(src_chunk_stride),
        Int64(dst_chunk_stride),
        Int64(src_offset),
        Int64(dst_offset),
        Int64(total),
        grid_dim=ceildiv(total, _BLOCK),
        block_dim=min(total, _BLOCK),
    )


def gpu_add_bias_forward(
    mut res: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    bias: Tensor[f32, Device.gpu],
) raises:
    """Res = t1 + bias, broadcasting `bias` (rank 1) over t1's trailing dim."""
    var ctx = res.gpu_context()
    var total = res.num_elements()
    var n = bias.num_elements()
    _cached_add_bias_kernel()._call_with_pack_checked(
        ctx,
        res.gpu_ptr(),
        t1.gpu_ptr(),
        bias.gpu_ptr(),
        Int64(n),
        Int64(total),
        grid_dim=ceildiv(total, _BLOCK),
        block_dim=min(total, _BLOCK),
    )


def _channel_bias_add_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    a: Pointer[Scalar[f32], MutAnyOrigin],
    bias: Pointer[Scalar[f32], MutAnyOrigin],
    channels: Int64,
    spatial: Int64,
    total: Int64,
):
    """res[i] = a[i] + bias[(i / spatial) % channels], for an NCHW tensor
    broadcasting a per-channel bias over dim 1."""
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < total:
        var ch = (i // spatial) % channels
        res.unsafe_store(
            Int(i), a.unsafe_load(Int(i)) + bias.unsafe_load(Int(ch))
        )


comptime _channel_bias_add_kernel_global = _Global[
    "mantle_gpu_kernel_channel_bias_add",
    _make_kernel_fn[_channel_bias_add_kernel],
]


def _cached_channel_bias_add_kernel() raises -> (
    type_of(
        _shared_device_context().compile_function[_channel_bias_add_kernel]()
    )
):
    return _channel_bias_add_kernel_global.get_or_create_ptr()[
        unsafe_offset=0
    ].copy()


def gpu_channel_bias_add_forward[
    channels: Int, spatial: Int
](
    mut res: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    bias: Tensor[f32, Device.gpu],
) raises:
    """Res = t1 + bias, broadcasting a rank-1 `bias` over dim 1 of an NCHW
    tensor (i.e. every `spatial` contiguous elements share one bias value)."""
    var ctx = res.gpu_context()
    var total = res.num_elements()
    _cached_channel_bias_add_kernel()._call_with_pack_checked(
        ctx,
        res.gpu_ptr(),
        t1.gpu_ptr(),
        bias.gpu_ptr(),
        Int64(channels),
        Int64(spatial),
        Int64(total),
        grid_dim=ceildiv(total, _BLOCK),
        block_dim=min(total, _BLOCK),
    )


def gpu_bias_grad_into(
    mut res_grad: Tensor[f32, Device.gpu], ug: Tensor[f32, Device.gpu]
) raises:
    """Write the reduction of `ug` into a preallocated bias-gradient buffer."""
    var ctx = res_grad.gpu_context()
    var total = ug.num_elements()
    var n = res_grad.num_elements()
    var outer = total // n
    _cached_bias_grad_kernel()._call_with_pack_checked(
        ctx,
        res_grad.gpu_ptr(),
        ug.gpu_ptr(),
        Int64(n),
        Int64(outer),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_bias_grad(
    ug: Tensor[f32, Device.gpu], n: Int
) raises -> Tensor[f32, Device.gpu]:
    """Reduces `ug`'s gradient back down to the broadcast bias's shape."""
    var res_grad = Tensor[f32, Device.gpu](TensorShape(n), uninitialized=True)
    gpu_bias_grad_into(res_grad, ug)
    return res_grad^


def _rand_uniform_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
    seed: UInt64,
    low: Scalar[f32],
    high: Scalar[f32],
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var gen = Random(seed=seed, offset=UInt64(i))
        var u = gen.step_uniform()
        res.unsafe_store(Int(i), u[0] * (high - low) + low)


comptime _rand_uniform_kernel_global = _Global[
    "mantle_gpu_kernel_rand_uniform", _make_kernel_fn[_rand_uniform_kernel]
]


def _cached_rand_uniform_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_rand_uniform_kernel]())
):
    return _rand_uniform_kernel_global.get_or_create_ptr()[
        unsafe_offset=0
    ].copy()


def _rand_normal_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
    seed: UInt64,
    mean: Scalar[f32],
    std: Scalar[f32],
):
    """Box-Muller, using one Philox stream per thread (independent of any
    other thread's draw, unlike the CPU Mersenne-Twister path in
    `rand_utils.mojo`."""
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var gen = Random(seed=seed, offset=UInt64(i))
        var u = gen.step_uniform_unbiased()
        var r = sqrt(-2.0 * log(u[0]))
        var z0 = r * cos(Scalar[f32](2.0 * pi) * u[1])
        res.unsafe_store(Int(i), mean + std * z0)


comptime _rand_normal_kernel_global = _Global[
    "mantle_gpu_kernel_rand_normal", _make_kernel_fn[_rand_normal_kernel]
]


def _cached_rand_normal_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_rand_normal_kernel]())
):
    return _rand_normal_kernel_global.get_or_create_ptr()[
        unsafe_offset=0
    ].copy()


def gpu_rand_uniform(
    mut res: Tensor[f32, Device.gpu], low: Scalar[f32], high: Scalar[f32]
) raises:
    """Fills `res` with values drawn from a uniform distribution, entirely
    on-device (Philox counter-based RNG, one independent stream per
    element)."""
    var ctx = res.gpu_context()
    var n = res.num_elements()
    var seed = random_ui64(0, UInt64.MAX)
    _cached_rand_uniform_kernel()._call_with_pack_checked(
        ctx,
        res.gpu_ptr(),
        Int64(n),
        seed,
        low,
        high,
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_rand_normal(
    mut res: Tensor[f32, Device.gpu], mean: Scalar[f32], std: Scalar[f32]
) raises:
    """Fills `res` with values drawn from a normal distribution, entirely
    on-device (Box-Muller over a Philox stream per element)."""
    var ctx = res.gpu_context()
    var n = res.num_elements()
    var seed = random_ui64(0, UInt64.MAX)
    _cached_rand_normal_kernel()._call_with_pack_checked(
        ctx,
        res.gpu_ptr(),
        Int64(n),
        seed,
        mean,
        std,
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def _adam_step_kernel(
    param: Pointer[Scalar[f32], MutAnyOrigin],
    momentum: Pointer[Scalar[f32], MutAnyOrigin],
    rms: Pointer[Scalar[f32], MutAnyOrigin],
    grad: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
    lr: Scalar[f32],
    beta1: Scalar[f32],
    beta2: Scalar[f32],
    epsilon: Scalar[f32],
    one_minus_beta1_pow_t: Scalar[f32],
    one_minus_beta2_pow_t: Scalar[f32],
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var m = momentum.unsafe_load(Int(i))
        var v = rms.unsafe_load(Int(i))
        var g = grad.unsafe_load(Int(i))
        var p = param.unsafe_load(Int(i))

        m = beta1 * m + (1 - beta1) * g
        momentum.unsafe_store(Int(i), m)
        var m_hat = m / one_minus_beta1_pow_t

        v = beta2 * v + (1 - beta2) * g * g
        rms.unsafe_store(Int(i), v)
        var v_hat = v / one_minus_beta2_pow_t

        p = p - lr * (m_hat / (sqrt(v_hat) + epsilon))
        param.unsafe_store(Int(i), p)


comptime _adam_step_kernel_global = _Global[
    "mantle_gpu_kernel_adam_step", _make_kernel_fn[_adam_step_kernel]
]


def _cached_adam_step_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_adam_step_kernel]())
):
    return _adam_step_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def gpu_adam_step(
    mut param: Tensor[f32, Device.gpu],
    mut momentum: Tensor[f32, Device.gpu],
    mut rms: Tensor[f32, Device.gpu],
    grad: Tensor[f32, Device.gpu],
    lr: Scalar[f32],
    beta1: Scalar[f32],
    beta2: Scalar[f32],
    epsilon: Scalar[f32],
    one_minus_beta1_pow_t: Scalar[f32],
    one_minus_beta2_pow_t: Scalar[f32],
) raises:
    """One Adam update (momentum/rms/bias-correction/param step)."""
    var ctx = param.gpu_context()
    var n = param.num_elements()
    _cached_adam_step_kernel()._call_with_pack_checked(
        ctx,
        param.gpu_ptr(),
        momentum.gpu_ptr(),
        rms.gpu_ptr(),
        grad.gpu_ptr(),
        Int64(n),
        lr,
        beta1,
        beta2,
        epsilon,
        one_minus_beta1_pow_t,
        one_minus_beta2_pow_t,
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def _adamw_step_kernel(
    param: Pointer[Scalar[f32], MutAnyOrigin],
    momentum: Pointer[Scalar[f32], MutAnyOrigin],
    rms: Pointer[Scalar[f32], MutAnyOrigin],
    grad: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
    lr: Scalar[f32],
    beta1: Scalar[f32],
    beta2: Scalar[f32],
    epsilon: Scalar[f32],
    weight_decay: Scalar[f32],
    one_minus_beta1_pow_t: Scalar[f32],
    one_minus_beta2_pow_t: Scalar[f32],
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var m = momentum.unsafe_load(Int(i))
        var v = rms.unsafe_load(Int(i))
        var g = grad.unsafe_load(Int(i))
        var p = param.unsafe_load(Int(i))

        m = beta1 * m + (1 - beta1) * g
        momentum.unsafe_store(Int(i), m)
        v = beta2 * v + (1 - beta2) * g * g
        rms.unsafe_store(Int(i), v)

        # AdamW decays the parameter independently of the adaptive update.
        p = p - lr * weight_decay * p
        p = p - lr * (
            (m / one_minus_beta1_pow_t)
            / (sqrt(v / one_minus_beta2_pow_t) + epsilon)
        )
        param.unsafe_store(Int(i), p)


comptime _adamw_step_kernel_global = _Global[
    "mantle_gpu_kernel_adamw_step", _make_kernel_fn[_adamw_step_kernel]
]


def _cached_adamw_step_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_adamw_step_kernel]())
):
    return _adamw_step_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def gpu_adamw_step(
    mut param: Tensor[f32, Device.gpu],
    mut momentum: Tensor[f32, Device.gpu],
    mut rms: Tensor[f32, Device.gpu],
    grad: Tensor[f32, Device.gpu],
    lr: Scalar[f32],
    beta1: Scalar[f32],
    beta2: Scalar[f32],
    epsilon: Scalar[f32],
    weight_decay: Scalar[f32],
    one_minus_beta1_pow_t: Scalar[f32],
    one_minus_beta2_pow_t: Scalar[f32],
) raises:
    """One fully on-device AdamW update."""
    var ctx = param.gpu_context()
    var n = param.num_elements()
    _cached_adamw_step_kernel()._call_with_pack_checked(
        ctx,
        param.gpu_ptr(),
        momentum.gpu_ptr(),
        rms.gpu_ptr(),
        grad.gpu_ptr(),
        Int64(n),
        lr,
        beta1,
        beta2,
        epsilon,
        weight_decay,
        one_minus_beta1_pow_t,
        one_minus_beta2_pow_t,
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_write_from_host(
    mut gpu_t: Tensor[f32, Device.gpu], host_t: Tensor[f32, Device.cpu]
) raises:
    """Overwrites `gpu_t`'s existing device buffer with `host_t`'s data —
    used by `ops.mojo`'s host round-trip fallback for ops without a native
    GPU kernel yet (unlike `Tensor.to_gpu()`, this writes into the buffer
    `gpu_t` already owns rather than allocating a new one, so callers that
    hold a `.share()` of `gpu_t` still see the update)."""
    var ctx = gpu_t.gpu_context()
    ctx.enqueue_copy(gpu_t.gpu_ptr(), host_t.ptr(), gpu_t.num_elements())
    ctx.synchronize()


def gpu_add_forward(
    mut res: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    t2: Tensor[f32, Device.gpu],
) raises:
    var ctx = res.gpu_context()
    var n = res.num_elements()
    _cached_add_kernel()._call_with_pack_checked(
        ctx,
        res.gpu_ptr(),
        t1.gpu_ptr(),
        t2.gpu_ptr(),
        Int64(n),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_sub_forward(
    mut res: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    t2: Tensor[f32, Device.gpu],
) raises:
    var ctx = res.gpu_context()
    var n = res.num_elements()
    _cached_sub_kernel()._call_with_pack_checked(
        ctx,
        res.gpu_ptr(),
        t1.gpu_ptr(),
        t2.gpu_ptr(),
        Int64(n),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_mul_forward(
    mut res: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    t2: Tensor[f32, Device.gpu],
) raises:
    var ctx = res.gpu_context()
    var n = res.num_elements()
    _cached_mul_kernel()._call_with_pack_checked(
        ctx,
        res.gpu_ptr(),
        t1.gpu_ptr(),
        t2.gpu_ptr(),
        Int64(n),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_div_forward(
    mut res: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    t2: Tensor[f32, Device.gpu],
) raises:
    var ctx = res.gpu_context()
    var n = res.num_elements()
    _cached_div_kernel()._call_with_pack_checked(
        ctx,
        res.gpu_ptr(),
        t1.gpu_ptr(),
        t2.gpu_ptr(),
        Int64(n),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_broadcast_binary_forward(
    mut res: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    t2: Tensor[f32, Device.gpu],
    repeat: Int,
    mode: Int,
    op: Int,
) raises:
    var ctx = res.gpu_context()
    var n = res.num_elements()
    _cached_broadcast_binary_kernel()._call_with_pack_checked(
        ctx,
        res.gpu_ptr(),
        t1.gpu_ptr(),
        t2.gpu_ptr(),
        Int64(n),
        Int64(t2.num_elements()),
        Int64(repeat),
        Int64(mode),
        Int64(op),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_broadcast_binary_backward_t1(
    ug: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    t2: Tensor[f32, Device.gpu],
    repeat: Int,
    mode: Int,
    op: Int,
) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](t1.shape(), uninitialized=True)
    var ctx = res_grad.gpu_context()
    var n = res_grad.num_elements()
    _cached_broadcast_binary_bw_t1_kernel()._call_with_pack_checked(
        ctx,
        res_grad.gpu_ptr(),
        ug.gpu_ptr(),
        t1.gpu_ptr(),
        t2.gpu_ptr(),
        Int64(n),
        Int64(t2.num_elements()),
        Int64(repeat),
        Int64(mode),
        Int64(op),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )
    return res_grad^


def gpu_broadcast_binary_backward_t2(
    ug: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    t2: Tensor[f32, Device.gpu],
    repeat: Int,
    mode: Int,
    op: Int,
) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](t2.shape(), uninitialized=True)
    var ctx = res_grad.gpu_context()
    var n = ug.num_elements()
    var out_n = res_grad.num_elements()
    _cached_broadcast_binary_bw_t2_kernel()._call_with_pack_checked(
        ctx,
        res_grad.gpu_ptr(),
        ug.gpu_ptr(),
        t1.gpu_ptr(),
        t2.gpu_ptr(),
        Int64(n),
        Int64(out_n),
        Int64(repeat),
        Int64(mode),
        Int64(op),
        grid_dim=ceildiv(out_n, _BLOCK),
        block_dim=min(out_n, _BLOCK),
    )
    return res_grad^


def gpu_accumulate_grad(
    mut grad: Tensor[f32, Device.gpu], res_grad: Tensor[f32, Device.gpu]
) raises:
    """`grad += res_grad`, elementwise (no broadcasting)."""
    var ctx = grad.gpu_context()
    var n = grad.num_elements()
    _cached_accumulate_kernel()._call_with_pack_checked(
        ctx,
        grad.gpu_ptr(),
        res_grad.gpu_ptr(),
        Int64(n),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_relu_forward(
    mut res: Tensor[f32, Device.gpu], t1: Tensor[f32, Device.gpu]
) raises:
    var ctx = res.gpu_context()
    var n = res.num_elements()
    _cached_relu_kernel()._call_with_pack_checked(
        ctx,
        res.gpu_ptr(),
        t1.gpu_ptr(),
        Int64(n),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_relu_backward_into(
    mut grad: Tensor[f32, Device.gpu],
    ug: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
) raises:
    var ctx = grad.gpu_context()
    var n = grad.num_elements()
    _cached_relu_bw_kernel()._call_with_pack_checked(
        ctx,
        grad.gpu_ptr(),
        t1.gpu_ptr(),
        ug.gpu_ptr(),
        Int64(n),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_relu_backward(
    ug: Tensor[f32, Device.gpu], t1: Tensor[f32, Device.gpu]
) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](ug.shape(), uninitialized=True)
    gpu_relu_backward_into(res_grad, ug, t1)
    return res_grad^


def gpu_sub_backward_t2_into(
    mut grad: Tensor[f32, Device.gpu], ug: Tensor[f32, Device.gpu]
) raises:
    var ctx = grad.gpu_context()
    var n = grad.num_elements()
    _cached_neg_kernel()._call_with_pack_checked(
        ctx,
        grad.gpu_ptr(),
        ug.gpu_ptr(),
        Int64(n),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_sub_backward_t2(
    ug: Tensor[f32, Device.gpu]
) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](ug.shape(), uninitialized=True)
    gpu_sub_backward_t2_into(res_grad, ug)
    return res_grad^


def gpu_mul_backward_into(
    mut grad: Tensor[f32, Device.gpu],
    ug: Tensor[f32, Device.gpu],
    other: Tensor[f32, Device.gpu],
) raises:
    var ctx = grad.gpu_context()
    var n = grad.num_elements()
    _cached_mul_kernel()._call_with_pack_checked(
        ctx,
        grad.gpu_ptr(),
        ug.gpu_ptr(),
        other.gpu_ptr(),
        Int64(n),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_mul_backward(
    ug: Tensor[f32, Device.gpu], other: Tensor[f32, Device.gpu]
) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](ug.shape(), uninitialized=True)
    gpu_mul_backward_into(res_grad, ug, other)
    return res_grad^


def gpu_div_backward_t1_into(
    mut grad: Tensor[f32, Device.gpu],
    ug: Tensor[f32, Device.gpu],
    t2: Tensor[f32, Device.gpu],
) raises:
    var ctx = grad.gpu_context()
    var n = grad.num_elements()
    _cached_div_kernel()._call_with_pack_checked(
        ctx,
        grad.gpu_ptr(),
        ug.gpu_ptr(),
        t2.gpu_ptr(),
        Int64(n),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_div_backward_t1(
    ug: Tensor[f32, Device.gpu], t2: Tensor[f32, Device.gpu]
) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](ug.shape(), uninitialized=True)
    gpu_div_backward_t1_into(res_grad, ug, t2)
    return res_grad^


def gpu_div_backward_t2_into(
    mut grad: Tensor[f32, Device.gpu],
    ug: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    t2: Tensor[f32, Device.gpu],
) raises:
    var ctx = grad.gpu_context()
    var n = grad.num_elements()
    _cached_div_bw_t2_kernel()._call_with_pack_checked(
        ctx,
        grad.gpu_ptr(),
        t1.gpu_ptr(),
        t2.gpu_ptr(),
        ug.gpu_ptr(),
        Int64(n),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )


def gpu_div_backward_t2(
    ug: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    t2: Tensor[f32, Device.gpu],
) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](ug.shape(), uninitialized=True)
    gpu_div_backward_t2_into(res_grad, ug, t1, t2)
    return res_grad^

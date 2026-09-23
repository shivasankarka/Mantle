# ===----------------------------------------------------------------------=== #
# Mantle: GPU elementwise kernels
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""GPU elementwise kernels (mantle.autograd.ops.gpu_elementwise)
------------------------------------------------
GPU kernels for ADD/SUB/MUL/DIV/RELU forward+backward. Same-shape
operands only (no broadcast).
"""
from std.math import ceildiv
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext

from mantle import f32
from mantle.core.tensor import Tensor
from mantle.core.device import Device

comptime _BLOCK = 256


def _add_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    a: Pointer[Scalar[f32], MutAnyOrigin],
    b: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        res.unsafe_store(Int(i), a.unsafe_load(Int(i)) + b.unsafe_load(Int(i)))


def _sub_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    a: Pointer[Scalar[f32], MutAnyOrigin],
    b: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        res.unsafe_store(Int(i), a.unsafe_load(Int(i)) - b.unsafe_load(Int(i)))


def _mul_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    a: Pointer[Scalar[f32], MutAnyOrigin],
    b: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        res.unsafe_store(Int(i), a.unsafe_load(Int(i)) * b.unsafe_load(Int(i)))


def _div_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    a: Pointer[Scalar[f32], MutAnyOrigin],
    b: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        res.unsafe_store(Int(i), a.unsafe_load(Int(i)) / b.unsafe_load(Int(i)))


def _accumulate_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    other: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        res.unsafe_store(Int(i), res.unsafe_load(Int(i)) + other.unsafe_load(Int(i)))


def _neg_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    a: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        res.unsafe_store(Int(i), -a.unsafe_load(Int(i)))


def _relu_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    a: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var v = a.unsafe_load(Int(i))
        res.unsafe_store(Int(i), v if v > 0 else Scalar[f32](0))


def _relu_bw_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    t1: Pointer[Scalar[f32], MutAnyOrigin],
    ug: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var mask = Scalar[f32](1) if t1.unsafe_load(Int(i)) > 0 else Scalar[f32](0)
        res.unsafe_store(Int(i), mask * ug.unsafe_load(Int(i)))


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
            Int(i), -t1.unsafe_load(Int(i)) / (t2v * t2v) * ug.unsafe_load(Int(i))
        )


def gpu_add_forward(
    mut res: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    t2: Tensor[f32, Device.gpu],
) raises:
    var ctx = res.gpu_context()
    var n = res.num_elements()
    ctx.compile_function[_add_kernel]()._call_with_pack_checked(
        ctx, res.gpu_ptr(), t1.gpu_ptr(), t2.gpu_ptr(), Int64(n),
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
    )
    ctx.synchronize()


def gpu_sub_forward(
    mut res: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    t2: Tensor[f32, Device.gpu],
) raises:
    var ctx = res.gpu_context()
    var n = res.num_elements()
    ctx.compile_function[_sub_kernel]()._call_with_pack_checked(
        ctx, res.gpu_ptr(), t1.gpu_ptr(), t2.gpu_ptr(), Int64(n),
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
    )
    ctx.synchronize()


def gpu_mul_forward(
    mut res: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    t2: Tensor[f32, Device.gpu],
) raises:
    var ctx = res.gpu_context()
    var n = res.num_elements()
    ctx.compile_function[_mul_kernel]()._call_with_pack_checked(
        ctx, res.gpu_ptr(), t1.gpu_ptr(), t2.gpu_ptr(), Int64(n),
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
    )
    ctx.synchronize()


def gpu_div_forward(
    mut res: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    t2: Tensor[f32, Device.gpu],
) raises:
    var ctx = res.gpu_context()
    var n = res.num_elements()
    ctx.compile_function[_div_kernel]()._call_with_pack_checked(
        ctx, res.gpu_ptr(), t1.gpu_ptr(), t2.gpu_ptr(), Int64(n),
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
    )
    ctx.synchronize()


def gpu_accumulate_grad(
    mut grad: Tensor[f32, Device.gpu], res_grad: Tensor[f32, Device.gpu]
) raises:
    """`grad += res_grad`, elementwise (no broadcasting)."""
    var ctx = grad.gpu_context()
    var n = grad.num_elements()
    ctx.compile_function[_accumulate_kernel]()._call_with_pack_checked(
        ctx, grad.gpu_ptr(), res_grad.gpu_ptr(), Int64(n),
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
    )
    ctx.synchronize()


def gpu_relu_forward(
    mut res: Tensor[f32, Device.gpu], t1: Tensor[f32, Device.gpu]
) raises:
    var ctx = res.gpu_context()
    var n = res.num_elements()
    ctx.compile_function[_relu_kernel]()._call_with_pack_checked(
        ctx, res.gpu_ptr(), t1.gpu_ptr(), Int64(n),
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
    )
    ctx.synchronize()


def gpu_relu_backward(
    ug: Tensor[f32, Device.gpu], t1: Tensor[f32, Device.gpu]
) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](ug.shape())
    var ctx = res_grad.gpu_context()
    var n = res_grad.num_elements()
    ctx.compile_function[_relu_bw_kernel]()._call_with_pack_checked(
        ctx, res_grad.gpu_ptr(), t1.gpu_ptr(), ug.gpu_ptr(), Int64(n),
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
    )
    ctx.synchronize()
    return res_grad^


def gpu_sub_backward_t2(ug: Tensor[f32, Device.gpu]) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](ug.shape())
    var ctx = res_grad.gpu_context()
    var n = res_grad.num_elements()
    ctx.compile_function[_neg_kernel]()._call_with_pack_checked(
        ctx, res_grad.gpu_ptr(), ug.gpu_ptr(), Int64(n),
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
    )
    ctx.synchronize()
    return res_grad^


def gpu_mul_backward(
    ug: Tensor[f32, Device.gpu], other: Tensor[f32, Device.gpu]
) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](ug.shape())
    var ctx = res_grad.gpu_context()
    var n = res_grad.num_elements()
    ctx.compile_function[_mul_kernel]()._call_with_pack_checked(
        ctx, res_grad.gpu_ptr(), ug.gpu_ptr(), other.gpu_ptr(), Int64(n),
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
    )
    ctx.synchronize()
    return res_grad^


def gpu_div_backward_t1(
    ug: Tensor[f32, Device.gpu], t2: Tensor[f32, Device.gpu]
) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](ug.shape())
    var ctx = res_grad.gpu_context()
    var n = res_grad.num_elements()
    ctx.compile_function[_div_kernel]()._call_with_pack_checked(
        ctx, res_grad.gpu_ptr(), ug.gpu_ptr(), t2.gpu_ptr(), Int64(n),
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
    )
    ctx.synchronize()
    return res_grad^


def gpu_div_backward_t2(
    ug: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    t2: Tensor[f32, Device.gpu],
) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](ug.shape())
    var ctx = res_grad.gpu_context()
    var n = res_grad.num_elements()
    ctx.compile_function[_div_bw_t2_kernel]()._call_with_pack_checked(
        ctx,
        res_grad.gpu_ptr(),
        t1.gpu_ptr(),
        t2.gpu_ptr(),
        ug.gpu_ptr(),
        Int64(n),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )
    ctx.synchronize()
    return res_grad^

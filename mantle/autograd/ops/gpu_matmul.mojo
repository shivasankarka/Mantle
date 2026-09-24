# ===----------------------------------------------------------------------=== #
# Mantle: GPU matmul kernels
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""GPU matmul (mantle.autograd.ops.gpu_matmul)
------------------------------------------------
Rank-2 matmul + transpose variants for DOT forward/backward, via MAX's own
`linalg.matmul` (TileTensor-based, not the `max.kernels.*` namespace —
that's internal-only and not importable from this distribution).

`matmul`'s `transpose_a` parameter is unsupported (per its own docstring),
and a `.transpose()` *view* passed as the `a` operand is silently wrong
whenever the output has a single column (N=1, likely a GEMV dispatch that
ignores the transposed strides) — confirmed on both `target="cpu"` and
`target="gpu"`. `transpose_b` (a real view on `b`) is fine at every shape
tested, including the analogous thin case. So `gpu_matmul_at` (needs
`a^T`) materializes the transpose with a small copy kernel first, then
calls `matmul` untransposed; `gpu_matmul_bt` (needs `b^T`) uses
`.transpose()` directly.
"""
from std.math import ceildiv
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from std.ffi import _Global
from std.os import abort
from linalg.matmul import matmul
from layout import TileTensor, Coord
from layout.tile_layout import row_major
from std.utils import IndexList

from mantle import f32
from mantle.core.tensor import Tensor, TensorShape, _shared_device_context
from mantle.core.device import Device

comptime _BLOCK = 16


def _make_kernel_fn[
    declared_arg_types: TypeList[Trait=AnyType, ...],
    //,
    func: def(*args: *declared_arg_types) thin -> None,
]() -> type_of(_shared_device_context().compile_function[func]()):
    """Caches a kernel's compiled `DeviceFunction` (same rationale as
    `_shared_device_context` caching the `DeviceContext` itself — see
    `gpu_elementwise.mojo`)."""
    try:
        return _shared_device_context().compile_function[func]()
    except e:
        abort("Mantle: GPU kernel compile failed: " + String(e))


def _transpose_kernel(
    dst: Pointer[Scalar[f32], MutAnyOrigin],
    src: Pointer[Scalar[f32], MutAnyOrigin],
    rows: Int64,
    cols: Int64,
):
    """dst[cols,rows] = src[rows,cols]^T."""
    var r = Int64(block_idx.y * block_dim.y + thread_idx.y)
    var c = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if r < rows and c < cols:
        dst.unsafe_store(Int(c * rows + r), src.unsafe_load(Int(r * cols + c)))


comptime _transpose_kernel_global = _Global[
    "mantle_gpu_kernel_transpose", _make_kernel_fn[_transpose_kernel]
]


def _cached_transpose_kernel() raises -> type_of(_shared_device_context().compile_function[_transpose_kernel]()):
    return _transpose_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def gpu_transpose[
    rows: Int, cols: Int
](mut dst: Tensor[f32, Device.gpu], src: Tensor[f32, Device.gpu]) raises:
    var ctx = dst.gpu_context()
    _cached_transpose_kernel()._call_with_pack_checked(
        ctx,
        dst.gpu_ptr(),
        src.gpu_ptr(),
        Int64(rows),
        Int64(cols),
        grid_dim=(ceildiv(cols, _BLOCK), ceildiv(rows, _BLOCK)),
        block_dim=(_BLOCK, _BLOCK),
    )


def gpu_matmul[
    m: Int, k: Int, n: Int
](
    mut res: Tensor[f32, Device.gpu],
    a: Tensor[f32, Device.gpu],
    b: Tensor[f32, Device.gpu],
) raises:
    """Res[m,n] = a[m,k] @ b[k,n]."""
    var ctx = res.gpu_context()

    var res_tt = TileTensor(
        ptr=res.gpu_ptr().unsafe_origin_cast[MutAnyOrigin](),
        layout=row_major[m, n](),
    )
    var a_tt = TileTensor(
        ptr=a.gpu_ptr().unsafe_origin_cast[MutAnyOrigin](),
        layout=row_major[m, k](),
    )
    var b_tt = TileTensor(
        ptr=b.gpu_ptr().unsafe_origin_cast[MutAnyOrigin](),
        layout=row_major[k, n](),
    )
    matmul[target="gpu"](res_tt, a_tt, b_tt, ctx)


def gpu_matmul_bias[
    m: Int, k: Int, n: Int
](
    mut res: Tensor[f32, Device.gpu],
    a: Tensor[f32, Device.gpu],
    b: Tensor[f32, Device.gpu],
    bias: Tensor[f32, Device.gpu],
) raises:
    """Res[m,n] = a[m,k] @ b[k,n] + bias[n], the bias-add fused into
    matmul's own epilogue (`elementwise_lambda_fn`) instead of a separate
    kernel launch — for `Linear`'s forward pass. `Linear`'s bias is a
    trailing-dim broadcast, so the epilogue just adds `bias[col]` to each
    computed tile before storing it."""
    var ctx = res.gpu_context()
    var bias_ptr = bias.gpu_ptr()

    var res_tt = TileTensor(
        ptr=res.gpu_ptr().unsafe_origin_cast[MutAnyOrigin](),
        layout=row_major[m, n](),
    )
    var a_tt = TileTensor(
        ptr=a.gpu_ptr().unsafe_origin_cast[MutAnyOrigin](),
        layout=row_major[m, k](),
    )
    var b_tt = TileTensor(
        ptr=b.gpu_ptr().unsafe_origin_cast[MutAnyOrigin](),
        layout=row_major[k, n](),
    )

    @__parameter
    @inline(.always)
    @__copy_capture(res_tt, bias_ptr)
    def bias_epilogue[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]) -> None:
        var bv = bias_ptr.unsafe_load[width=width](idx[1])
        res_tt.store(Coord(idx), rebind[SIMD[f32, width]](val) + bv)

    matmul[target="gpu", elementwise_lambda_fn=bias_epilogue](
        res_tt, a_tt, b_tt, ctx
    )


def gpu_matmul_bt[
    m: Int, p: Int, k: Int
](
    mut res: Tensor[f32, Device.gpu],
    a: Tensor[f32, Device.gpu],
    b: Tensor[f32, Device.gpu],
) raises:
    """Res[m,k] = a[m,p] @ b[k,p]^T."""
    var ctx = res.gpu_context()

    var res_tt = TileTensor(
        ptr=res.gpu_ptr().unsafe_origin_cast[MutAnyOrigin](),
        layout=row_major[m, k](),
    )
    var a_tt = TileTensor(
        ptr=a.gpu_ptr().unsafe_origin_cast[MutAnyOrigin](),
        layout=row_major[m, p](),
    )
    var b_tt = TileTensor(
        ptr=b.gpu_ptr().unsafe_origin_cast[MutAnyOrigin](),
        layout=row_major[k, p](),
    )
    matmul[target="gpu"](res_tt, a_tt, b_tt.transpose(), ctx)


def gpu_matmul_at[
    p: Int, k: Int, n: Int
](
    mut res: Tensor[f32, Device.gpu],
    a: Tensor[f32, Device.gpu],
    b: Tensor[f32, Device.gpu],
) raises:
    """Res[k,n] = a[p,k]^T @ b[p,n]."""
    var ctx = res.gpu_context()

    var a_t = Tensor[f32, Device.gpu](TensorShape(k, p), uninitialized=True)
    gpu_transpose[p, k](a_t, a)

    var res_tt = TileTensor(
        ptr=res.gpu_ptr().unsafe_origin_cast[MutAnyOrigin](),
        layout=row_major[k, n](),
    )
    var a_t_tt = TileTensor(
        ptr=a_t.gpu_ptr().unsafe_origin_cast[MutAnyOrigin](),
        layout=row_major[k, p](),
    )
    var b_tt = TileTensor(
        ptr=b.gpu_ptr().unsafe_origin_cast[MutAnyOrigin](),
        layout=row_major[p, n](),
    )
    matmul[target="gpu"](res_tt, a_t_tt, b_tt, ctx)

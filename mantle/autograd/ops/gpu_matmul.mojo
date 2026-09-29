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
`linalg.matmul` (TileTensor-based).

Notes:
    `gpu_matmul_at` materializes a^T with a copy kernel before matmul;
    `gpu_matmul_bt` uses `.transpose()` directly on b.
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
    func: def(* args: * declared_arg_types) thin -> None,
]() -> type_of(_shared_device_context().compile_function[func]()):
    """Compile and cache a kernel's `DeviceFunction`."""
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


def _cached_transpose_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_transpose_kernel]())
):
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


def _transpose_4d_kernel(
    dst: Pointer[Scalar[f32], MutAnyOrigin],
    src: Pointer[Scalar[f32], MutAnyOrigin],
    d0: Int64,
    d1: Int64,
    d2: Int64,
    d3: Int64,
    a0: Int64,
    a1: Int64,
    a2: Int64,
    a3: Int64,
):
    """Permute a contiguous rank-4 tensor with runtime axes.

    The output coordinate `(c0,c1,c2,c3)` maps to source coordinates at
    axes `(a0,a1,a2,a3)`. Runtime axes keep one cached kernel usable for all
    attention permutations while avoiding a CPU materialization.
    """
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < d0 * d1 * d2 * d3:
        var out_d0 = d0
        var out_d1 = d1
        var out_d2 = d2
        var out_d3 = d3
        if a0 == 1:
            out_d0 = d1
        elif a0 == 2:
            out_d0 = d2
        elif a0 == 3:
            out_d0 = d3
        if a1 == 0:
            out_d1 = d0
        elif a1 == 2:
            out_d1 = d2
        elif a1 == 3:
            out_d1 = d3
        if a2 == 0:
            out_d2 = d0
        elif a2 == 1:
            out_d2 = d1
        elif a2 == 3:
            out_d2 = d3
        if a3 == 0:
            out_d3 = d0
        elif a3 == 1:
            out_d3 = d1
        elif a3 == 2:
            out_d3 = d2

        var remaining = i
        var c3 = remaining % out_d3
        remaining = remaining // out_d3
        var c2 = remaining % out_d2
        remaining = remaining // out_d2
        var c1 = remaining % out_d1
        var c0 = remaining // out_d1

        var source_offset: Int64 = 0
        if a0 == 0:
            source_offset += c0 * d1 * d2 * d3
        elif a0 == 1:
            source_offset += c0 * d2 * d3
        elif a0 == 2:
            source_offset += c0 * d3
        else:
            source_offset += c0
        if a1 == 0:
            source_offset += c1 * d1 * d2 * d3
        elif a1 == 1:
            source_offset += c1 * d2 * d3
        elif a1 == 2:
            source_offset += c1 * d3
        else:
            source_offset += c1
        if a2 == 0:
            source_offset += c2 * d1 * d2 * d3
        elif a2 == 1:
            source_offset += c2 * d2 * d3
        elif a2 == 2:
            source_offset += c2 * d3
        else:
            source_offset += c2
        if a3 == 0:
            source_offset += c3 * d1 * d2 * d3
        elif a3 == 1:
            source_offset += c3 * d2 * d3
        elif a3 == 2:
            source_offset += c3 * d3
        else:
            source_offset += c3
        dst.unsafe_store(Int(i), src.unsafe_load(Int(source_offset)))


comptime _transpose_4d_kernel_global = _Global[
    "mantle_gpu_kernel_transpose_4d", _make_kernel_fn[_transpose_4d_kernel]
]


def _cached_transpose_4d_kernel() raises -> (
    type_of(_shared_device_context().compile_function[_transpose_4d_kernel]())
):
    return _transpose_4d_kernel_global.get_or_create_ptr()[
        unsafe_offset=0
    ].copy()


def gpu_transpose_4d(
    mut dst: Tensor[f32, Device.gpu],
    src: Tensor[f32, Device.gpu],
    axes: TensorShape,
) raises:
    """Device-resident rank-4 transpose used by multi-head attention."""
    var shape = src.shape()
    var ctx = dst.gpu_context()
    _cached_transpose_4d_kernel()._call_with_pack_checked(
        ctx,
        dst.gpu_ptr(),
        src.gpu_ptr(),
        Int64(shape[0]),
        Int64(shape[1]),
        Int64(shape[2]),
        Int64(shape[3]),
        Int64(axes[0]),
        Int64(axes[1]),
        Int64(axes[2]),
        Int64(axes[3]),
        grid_dim=(ceildiv(src.num_elements(), _BLOCK),),
        block_dim=(_BLOCK,),
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
    """Res[k,n] = a[p,k]^T @ b[p,n] without materializing ``a^T``."""
    var ctx = res.gpu_context()

    var res_tt = TileTensor(
        ptr=res.gpu_ptr().unsafe_origin_cast[MutAnyOrigin](),
        layout=row_major[k, n](),
    )
    var a_tt = TileTensor(
        ptr=a.gpu_ptr().unsafe_origin_cast[MutAnyOrigin](),
        layout=row_major[p, k](),
    )
    var b_tt = TileTensor(
        ptr=b.gpu_ptr().unsafe_origin_cast[MutAnyOrigin](),
        layout=row_major[p, n](),
    )
    matmul[target="gpu"](res_tt, a_tt.transpose(), b_tt, ctx)


def gpu_batched_matmul[
    batches: Int, m: Int, k: Int, n: Int
](
    mut res: Tensor[f32, Device.gpu],
    a: Tensor[f32, Device.gpu],
    b: Tensor[f32, Device.gpu],
) raises:
    """Per-batch `a[m,k] @ b[k,n]`.

    Notes:
        Submits one ordered device matmul per contiguous `(B,H,*,*)` slice;
        all operands remain on GPU with no host staging.
    """
    var ctx = res.gpu_context()
    var res_ptr = res.gpu_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var a_ptr = a.gpu_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var b_ptr = b.gpu_ptr().unsafe_origin_cast[MutAnyOrigin]()
    for batch in range(batches):
        var res_tt = TileTensor(
            ptr=res_ptr.unsafe_offset(batch * m * n),
            layout=row_major[m, n](),
        )
        var a_tt = TileTensor(
            ptr=a_ptr.unsafe_offset(batch * m * k),
            layout=row_major[m, k](),
        )
        var b_tt = TileTensor(
            ptr=b_ptr.unsafe_offset(batch * k * n),
            layout=row_major[k, n](),
        )
        matmul[target="gpu"](res_tt, a_tt, b_tt, ctx)


def gpu_batched_matmul_bt[
    batches: Int, m: Int, p: Int, k: Int
](
    mut res: Tensor[f32, Device.gpu],
    a: Tensor[f32, Device.gpu],
    b: Tensor[f32, Device.gpu],
) raises:
    """Per-batch `a[m,p] @ b[k,p]^T`."""
    var ctx = res.gpu_context()
    var res_ptr = res.gpu_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var a_ptr = a.gpu_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var b_ptr = b.gpu_ptr().unsafe_origin_cast[MutAnyOrigin]()
    for batch in range(batches):
        var res_tt = TileTensor(
            ptr=res_ptr.unsafe_offset(batch * m * k),
            layout=row_major[m, k](),
        )
        var a_tt = TileTensor(
            ptr=a_ptr.unsafe_offset(batch * m * p),
            layout=row_major[m, p](),
        )
        var b_tt = TileTensor(
            ptr=b_ptr.unsafe_offset(batch * k * p),
            layout=row_major[k, p](),
        )
        matmul[target="gpu"](res_tt, a_tt, b_tt.transpose(), ctx)


def gpu_batched_matmul_at[
    batches: Int, p: Int, k: Int, n: Int
](
    mut res: Tensor[f32, Device.gpu],
    a: Tensor[f32, Device.gpu],
    b: Tensor[f32, Device.gpu],
) raises:
    """Per-batch `a[p,k]^T @ b[p,n]`."""
    var ctx = res.gpu_context()
    var res_ptr = res.gpu_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var a_ptr = a.gpu_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var b_ptr = b.gpu_ptr().unsafe_origin_cast[MutAnyOrigin]()
    for batch in range(batches):
        var res_tt = TileTensor(
            ptr=res_ptr.unsafe_offset(batch * k * n),
            layout=row_major[k, n](),
        )
        var a_tt = TileTensor(
            ptr=a_ptr.unsafe_offset(batch * p * k),
            layout=row_major[p, k](),
        )
        var b_tt = TileTensor(
            ptr=b_ptr.unsafe_offset(batch * p * n),
            layout=row_major[p, n](),
        )
        matmul[target="gpu"](res_tt, a_tt.transpose(), b_tt, ctx)

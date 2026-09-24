# ===----------------------------------------------------------------------=== #
# Mantle: Matrix Multiplication
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""MatMul (mantle.autograd.ops.matmul)
------------------------------------------------
Rank-2 matmul (via MAX's own `linalg.matmul`, CPU target) with batched and
transpose variants built on top of it.
"""
from std.memory import unsafe_memset_zero, Pointer
from linalg.matmul import matmul
from layout import TileTensor
from layout.tile_layout import row_major

from mantle import f32
from mantle.core.tensor import Tensor, TensorShape
from mantle.core.tensorutils import transpose_2D


# ===----------------------------------------------------------------------===#
# Rank-2 matmul
# ===----------------------------------------------------------------------===#


@always_inline
def dot[
    t1_shape: TensorShape, t2_shape: TensorShape
](mut res: Tensor[f32], t1: Tensor[f32], t2: Tensor[f32]) raises:
    dot[t1_shape, t2_shape](res.ptr(), t1.ptr(), t2.ptr())


@always_inline
def dot[
    mut1: Bool,
    mut2: Bool,
    origin_res: MutOrigin,
    origin_t1: Origin[mut=mut1],
    origin_t2: Origin[mut=mut2],
    //,
    t1_shape: TensorShape,
    t2_shape: TensorShape,
](
    res: Pointer[Scalar[f32], origin_res],
    t1: Pointer[Scalar[f32], origin_t1],
    t2: Pointer[Scalar[f32], origin_t2],
) raises:
    comptime M = t1_shape[0]  # t1[0]
    comptime K = t1_shape[1]  # t1[1], t2[0]
    comptime N = t2_shape[1]  # t2[1]

    var res_tt = TileTensor(
        ptr=res.unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin](),
        layout=row_major[M, N](),
    )
    var t1_tt = TileTensor(
        ptr=t1.unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin](),
        layout=row_major[M, K](),
    )
    var t2_tt = TileTensor(
        ptr=t2.unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin](),
        layout=row_major[K, N](),
    )
    matmul[target="cpu"](res_tt, t1_tt, t2_tt)


def dot_transpose_t2[
    mut1: Bool,
    mut2: Bool,
    origin_res: MutOrigin,
    origin_t1: Origin[mut=mut1],
    origin_t2: Origin[mut=mut2],
    //,
    A_shape: TensorShape,
    B_shape: TensorShape,
](
    mut C: Pointer[Scalar[f32], origin_res],
    A: Pointer[Scalar[f32], origin_t1],
    B: Pointer[Scalar[f32], origin_t2],
) raises:
    dot[A_shape, TensorShape(B_shape[1], B_shape[0])](
        C, A, transpose_2D[B_shape](B)
    )


def dot_transpose_t2[
    A_shape: TensorShape, B_shape: TensorShape
](mut C: Tensor[f32], A: Tensor[f32], B: Tensor[f32]) raises:
    unsafe_memset_zero(C.ptr(), C.num_elements())

    dot[A_shape, TensorShape(B_shape[1], B_shape[0])](
        C, A, transpose_2D[B_shape](B)
    )


# ===----------------------------------------------------------------------===#
# Batched matmul
# ===----------------------------------------------------------------------===#
#
# Semantics: matmul on the last two dims of t1/t2. Leading (batch) dims
# either match exactly, or one operand is rank-2 and is broadcast (shared)
# across every batch slice of the other operand — this covers both
# `(B,T,D)@(D,K)` (Linear on batched input) and `(B,H,T,d)@(B,H,d,T)`
# (attention, matching batch dims).


@always_inline
def num_batches(shape: TensorShape) -> Int:
    var n = 1
    for i in range(shape.rank() - 2):
        n *= shape[i]
    return n


def batched_dot[
    t1_shape: TensorShape, t2_shape: TensorShape
](mut res: Tensor[f32], t1: Tensor[f32], t2: Tensor[f32]) raises:
    comptime M = t1_shape[-2]
    comptime K = t1_shape[-1]
    comptime N = t2_shape[-1]
    comptime t1_batches = num_batches(t1_shape)
    comptime t2_batches = num_batches(t2_shape)
    comptime batches = max(t1_batches, t2_batches)

    comptime t1_step = 0 if t1_batches == 1 else M * K
    comptime t2_step = 0 if t2_batches == 1 else K * N

    var res_ptr = res.ptr()
    var t1_ptr = t1.ptr()
    var t2_ptr = t2.ptr()

    for b in range(batches):
        dot[TensorShape(M, K), TensorShape(K, N)](
            res_ptr.unsafe_offset(b * M * N),
            t1_ptr.unsafe_offset(b * t1_step),
            t2_ptr.unsafe_offset(b * t2_step),
        )


def batched_dot_transpose_t2[
    A_shape: TensorShape, B_shape: TensorShape
](mut C: Tensor[f32], A: Tensor[f32], B: Tensor[f32]) raises:
    """Batched dot(A, B^T) over the last two dims."""
    comptime M = A_shape[-2]
    comptime K = A_shape[-1]
    comptime N = B_shape[-2]
    comptime A_batches = num_batches(A_shape)
    comptime B_batches = num_batches(B_shape)
    comptime batches = max(A_batches, B_batches)

    comptime A_step = 0 if A_batches == 1 else M * K
    comptime B_step = 0 if B_batches == 1 else N * K

    unsafe_memset_zero(C.ptr(), C.num_elements())

    var C_ptr = C.ptr()
    var A_ptr = A.ptr()
    var B_ptr = B.ptr()

    for b in range(batches):
        var B_t = transpose_2D[TensorShape(N, K)](
            B_ptr.unsafe_offset(b * B_step)
        )
        dot[TensorShape(M, K), TensorShape(K, N)](
            C_ptr.unsafe_offset(b * M * N), A_ptr.unsafe_offset(b * A_step), B_t
        )
        B_t.unsafe_free()


def batched_dot_transpose_t1[
    A_shape: TensorShape, B_shape: TensorShape
](mut C: Tensor[f32], A: Tensor[f32], B: Tensor[f32]) raises:
    """Batched dot(A^T, B) over the last two dims."""
    comptime M = A_shape[-1]
    comptime K = A_shape[-2]
    comptime N = B_shape[-1]
    comptime A_batches = num_batches(A_shape)
    comptime B_batches = num_batches(B_shape)
    comptime batches = max(A_batches, B_batches)

    comptime A_step = 0 if A_batches == 1 else K * M
    comptime B_step = 0 if B_batches == 1 else K * N

    unsafe_memset_zero(C.ptr(), C.num_elements())

    var C_ptr = C.ptr()
    var A_ptr = A.ptr()
    var B_ptr = B.ptr()

    for b in range(batches):
        var A_t = transpose_2D[TensorShape(K, M)](
            A_ptr.unsafe_offset(b * A_step)
        )
        dot[TensorShape(M, K), TensorShape(K, N)](
            C_ptr.unsafe_offset(b * M * N), A_t, B_ptr.unsafe_offset(b * B_step)
        )
        A_t.unsafe_free()


def dot_transpose_t1[
    A_shape: TensorShape, B_shape: TensorShape
](mut C: Tensor[f32], A: Tensor[f32], B: Tensor[f32]) raises:
    unsafe_memset_zero(C.ptr(), C.num_elements())

    dot[TensorShape(A_shape[1], A_shape[0]), B_shape](
        C, transpose_2D[A_shape](A), B
    )

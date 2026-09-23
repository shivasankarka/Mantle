# ===----------------------------------------------------------------------=== #
# Mantle: Matrix Multiplication
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""MatMul (mantle.autograd.ops.matmul)
------------------------------------------------
Tiled, parallelized matrix multiplication with transpose variants.
"""
from std.algorithm import vectorize, parallelize
from std.memory import memset_zero, stack_allocation, UnsafePointer
from std.sys.info import simd_width_of

from mantle import f32
from mantle.core.tensor import Tensor, TensorShape
from mantle.core.tensorutils import transpose_2D


# ===----------------------------------------------------------------------===#
# Block Helpers
# ===----------------------------------------------------------------------===#


@always_inline
def calculate_block[
    mut1: Bool,
    mut2: Bool,
    origin_res: MutOrigin,
    origin_t1: Origin[mut=mut1],
    origin_t2: Origin[mut=mut2], //,
    M: Int,
    N: Int,
    K: Int,
    BLOCK_M: Int,
    BLOCK_N: Int,
    nelts: Int,
](
    res: UnsafePointer[Scalar[f32], origin_res],
    t1: UnsafePointer[Scalar[f32], origin_t1],
    t2: UnsafePointer[Scalar[f32], origin_t2],
    bm: Int,
    bn: Int,
):
    # Compute tile
    var acc = stack_allocation[BLOCK_M * BLOCK_N, f32]()
    memset_zero(acc, BLOCK_M * BLOCK_N)

    for k in range(K):
        comptime for m in range(BLOCK_M):

            def inner_n[
                nelts: Int
            ](n: Int) {mut acc, read t1, read t2, read bm, read bn, read k}:
                acc.store(
                    m * BLOCK_N + n,
                    SIMD[f32, nelts](t1[(bm + m) * K + k]).fma(
                        t2.load[width=nelts](k * N + (bn + n)),
                        acc.load[width=nelts](m * BLOCK_N + n),
                    ),
                )

            vectorize[nelts](BLOCK_N, inner_n)

    # Store tile
    for m in range(BLOCK_M):

        def vec_store[
            nelts: Int
        ](n: Int) {read res, read acc, read bm, read bn, read m}:
            res.store(
                (bm + m) * N + (bn + n), acc.load[width=nelts](m * BLOCK_N + n)
            )

        vectorize[nelts](BLOCK_N, vec_store)


@always_inline
def dot[
    t1_shape: TensorShape, t2_shape: TensorShape
](mut res: Tensor[f32], t1: Tensor[f32], t2: Tensor[f32]):
    dot[t1_shape, t2_shape](res.mut_ptr(), t1.ptr(), t2.ptr())


@always_inline
def dot[
    mut1: Bool,
    mut2: Bool,
    origin_res: MutOrigin,
    origin_t1: Origin[mut=mut1],
    origin_t2: Origin[mut=mut2], //,
    t1_shape: TensorShape,
    t2_shape: TensorShape,
](
    res: UnsafePointer[Scalar[f32], origin_res],
    t1: UnsafePointer[Scalar[f32], origin_t1],
    t2: UnsafePointer[Scalar[f32], origin_t2],
):
    comptime M = t1_shape[0]  # t1[0]
    comptime K = t1_shape[1]  # t1[1], t2[0]
    comptime N = t2_shape[1]  # t2[1]

    # simd_width_of[f32]() = 8 for float32
    comptime nelts = simd_width_of[f32]()
    comptime BLOCK_N = 8 * 2
    comptime BLOCK_M = 6
    comptime THREADS = 6  # num_logical_cores()

    comptime BLOCK_N_REMAINDER = N % BLOCK_N
    comptime BLOCK_M_REMAINDER = M % BLOCK_M

    @parameter
    def bm_par(m_outer: Int):
        var bm = m_outer * BLOCK_M

        for n_outer in range(0, N // BLOCK_N):
            var bn = n_outer * BLOCK_N

            calculate_block[M, N, K, BLOCK_M, BLOCK_N, nelts](
                res, t1, t2, bm, bn
            )

        # Handle the remainder of N
        comptime if BLOCK_N_REMAINDER > 0:
            var bn = N - BLOCK_N_REMAINDER

            calculate_block[M, N, K, BLOCK_M, BLOCK_N_REMAINDER, nelts](
                res, t1, t2, bm, bn
            )

    parallelize[bm_par](M // BLOCK_M, M // BLOCK_M)

    # Handle the remainder of M
    comptime if BLOCK_M_REMAINDER > 0:
        var bm = M - BLOCK_M_REMAINDER

        # comptime for?
        for n_outer in range(0, N // BLOCK_N):
            var bn = n_outer * BLOCK_N

            calculate_block[M, N, K, BLOCK_M_REMAINDER, BLOCK_N, nelts](
                res, t1, t2, bm, bn
            )

        # Handle corner remainder
        comptime if BLOCK_N_REMAINDER > 0:
            var bn = N - BLOCK_N_REMAINDER

            calculate_block[
                M, N, K, BLOCK_M_REMAINDER, BLOCK_N_REMAINDER, nelts
            ](res, t1, t2, bm, bn)


def dot_transpose_t2[
    mut1: Bool,
    mut2: Bool,
    origin_res: MutOrigin,
    origin_t1: Origin[mut=mut1],
    origin_t2: Origin[mut=mut2], //,
    A_shape: TensorShape,
    B_shape: TensorShape,
](
    mut C: UnsafePointer[Scalar[f32], origin_res],
    A: UnsafePointer[Scalar[f32], origin_t1],
    B: UnsafePointer[Scalar[f32], origin_t2],
):
    dot[A_shape, TensorShape(B_shape[1], B_shape[0])](
        C, A, transpose_2D[B_shape](B)
    )


def dot_transpose_t2[
    A_shape: TensorShape, B_shape: TensorShape
](mut C: Tensor[f32], A: Tensor[f32], B: Tensor[f32]):
    memset_zero(C.mut_ptr(), C.num_elements())

    dot[A_shape, TensorShape(B_shape[1], B_shape[0])](
        C, A, transpose_2D[B_shape](B)
    )

    # @parameter
    # def calc_row(i: Int):
    #     for j in range(B_shape[0]):

    #         @parameter
    #         def calc_row_A_B[nelts: Int](k: Int):
    #             var A_pos = i * A.dim(1) + k
    #             var B_pos = j * A.dim(1) + k
    #             var t_new_pos = i * C.dim(1) + j

    #             C[t_new_pos] += (
    #                 A.load[nelts](A_pos) * B.load[nelts](B_pos)
    #             ).reduce_add()

    #         vectorize[calc_row_A_B, nelts, size=A_shape[1]]()

    # parallelize[calc_row](A_shape[0], 1)


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
](mut res: Tensor[f32], t1: Tensor[f32], t2: Tensor[f32]):
    comptime M = t1_shape[-2]
    comptime K = t1_shape[-1]
    comptime N = t2_shape[-1]
    comptime t1_batches = num_batches(t1_shape)
    comptime t2_batches = num_batches(t2_shape)
    comptime batches = max(t1_batches, t2_batches)

    comptime t1_step = 0 if t1_batches == 1 else M * K
    comptime t2_step = 0 if t2_batches == 1 else K * N

    var res_ptr = res.mut_ptr()
    var t1_ptr = t1.ptr()
    var t2_ptr = t2.ptr()

    for b in range(batches):
        dot[TensorShape(M, K), TensorShape(K, N)](
            res_ptr + b * M * N,
            t1_ptr + b * t1_step,
            t2_ptr + b * t2_step,
        )


def batched_dot_transpose_t2[
    A_shape: TensorShape, B_shape: TensorShape
](mut C: Tensor[f32], A: Tensor[f32], B: Tensor[f32]):
    """Batched dot(A, B^T) over the last two dims."""
    comptime M = A_shape[-2]
    comptime K = A_shape[-1]
    comptime N = B_shape[-2]
    comptime A_batches = num_batches(A_shape)
    comptime B_batches = num_batches(B_shape)
    comptime batches = max(A_batches, B_batches)

    comptime A_step = 0 if A_batches == 1 else M * K
    comptime B_step = 0 if B_batches == 1 else N * K

    memset_zero(C.mut_ptr(), C.num_elements())

    var C_ptr = C.mut_ptr()
    var A_ptr = A.ptr()
    var B_ptr = B.ptr()

    for b in range(batches):
        var B_t = transpose_2D[TensorShape(N, K)](B_ptr + b * B_step)
        dot[TensorShape(M, K), TensorShape(K, N)](
            C_ptr + b * M * N, A_ptr + b * A_step, B_t
        )
        B_t.free()


def batched_dot_transpose_t1[
    A_shape: TensorShape, B_shape: TensorShape
](mut C: Tensor[f32], A: Tensor[f32], B: Tensor[f32]):
    """Batched dot(A^T, B) over the last two dims."""
    comptime M = A_shape[-1]
    comptime K = A_shape[-2]
    comptime N = B_shape[-1]
    comptime A_batches = num_batches(A_shape)
    comptime B_batches = num_batches(B_shape)
    comptime batches = max(A_batches, B_batches)

    comptime A_step = 0 if A_batches == 1 else K * M
    comptime B_step = 0 if B_batches == 1 else K * N

    memset_zero(C.mut_ptr(), C.num_elements())

    var C_ptr = C.mut_ptr()
    var A_ptr = A.ptr()
    var B_ptr = B.ptr()

    for b in range(batches):
        var A_t = transpose_2D[TensorShape(K, M)](A_ptr + b * A_step)
        dot[TensorShape(M, K), TensorShape(K, N)](
            C_ptr + b * M * N, A_t, B_ptr + b * B_step
        )
        A_t.free()


def dot_transpose_t1[
    A_shape: TensorShape, B_shape: TensorShape
](mut C: Tensor[f32], A: Tensor[f32], B: Tensor[f32]):
    memset_zero(C.mut_ptr(), C.num_elements())

    dot[TensorShape(A_shape[1], A_shape[0]), B_shape](
        C, transpose_2D[A_shape](A), B
    )

    # @parameter
    # def calc_row(i: Int):
    #     for j in range(A_shape[0]):

    #         @parameter
    #         def calc_row_t_new_B[nelts: Int](k: Int):
    #             var A_pos = j * A.dim(1) + i
    #             var B_pos = j * B.dim(1) + k
    #             var t_new_pos = i * C.dim(1) + k

    #             C.store[nelts](
    #                 t_new_pos,
    #                 C.load[nelts](t_new_pos)
    #                 + A[A_pos] * B.load[nelts](B_pos),
    #             )

    #         vectorize[calc_row_t_new_B, nelts, size=B_shape[1]]()

    # parallelize[calc_row](A_shape[1], 1)

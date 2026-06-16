from std.algorithm import vectorize, parallelize
from std.memory import memset_zero, stack_allocation, UnsafePointer
from std.sys.info import simd_width_of

from basalt.utils.tensorutils import transpose_2D


@always_inline
def calculate_block[
    M: Int, N: Int, K: Int, BLOCK_M: Int, BLOCK_N: Int, nelts: Int
](
    res: UnsafePointer[Scalar[dtype], _],
    t1: UnsafePointer[Scalar[dtype], _],
    t2: UnsafePointer[Scalar[dtype], _],
    bm: Int,
    bn: Int,
):
    # Compute tile
    var acc = stack_allocation[BLOCK_M * BLOCK_N, dtype]()
    memset_zero(acc, BLOCK_M * BLOCK_N)

    for k in range(K):
        comptime for m in range(BLOCK_M):

            def inner_n[
                nelts: Int
            ](n: Int) {
                mut acc, read t1, read t2, read acc, read bm, read bn, read k
            }:
                acc.store(
                    m * BLOCK_N + n,
                    SIMD[dtype, nelts](t1[(bm + m) * K + k]).fma(
                        t2.load[width=nelts](k * N + (bn + n)),
                        acc.load[width=nelts](m * BLOCK_N + n),
                    ),
                )

            vectorize[nelts](BLOCK_N, inner_n)

    # Store tile
    for m in range(BLOCK_M):

        def vec_store[
            nelts: Int
        ](n: Int) {mut res, read acc, read bm, read bn, read m, read n}:
            res.store(
                (bm + m) * N + (bn + n), acc.load[width=nelts](m * BLOCK_N + n)
            )

        vectorize[vec_store, nelts](BLOCK_N)


@parameter
@always_inline
def dot[
    t1_shape: TensorShape, t2_shape: TensorShape
](mut res: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]):
    dot[t1_shape, t2_shape](res.data(), t1.data(), t2.data())


@parameter
@always_inline
def dot[
    t1_shape: TensorShape, t2_shape: TensorShape
](
    res: UnsafePointer[Scalar[dtype]],
    t1: UnsafePointer[Scalar[dtype]],
    t2: UnsafePointer[Scalar[dtype]],
):
    comptime M = t1_shape[0]  # t1[0]
    comptime K = t1_shape[1]  # t1[1], t2[0]
    comptime N = t2_shape[1]  # t2[1]

    # simdwidthof[dtype]() = 8 for float32
    comptime nelts = simdwidthof[dtype]()
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
    A_shape: TensorShape, B_shape: TensorShape
](
    mut C: UnsafePointer[Scalar[dtype]],
    A: UnsafePointer[Scalar[dtype]],
    B: UnsafePointer[Scalar[dtype]],
):
    dot[A_shape, TensorShape(B_shape[1], B_shape[0])](
        C, A, transpose_2D[B_shape](B)
    )


def dot_transpose_t2[
    A_shape: TensorShape, B_shape: TensorShape
](mut C: Tensor[dtype], A: Tensor[dtype], B: Tensor[dtype]):
    memset_zero(C.data(), C.num_elements())

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


def dot_transpose_t1[
    A_shape: TensorShape, B_shape: TensorShape
](mut C: Tensor[dtype], A: Tensor[dtype], B: Tensor[dtype]):
    memset_zero(C.data(), C.num_elements())

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

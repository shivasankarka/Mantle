from std.random import rand
from std.testing import assert_equal, assert_almost_equal
from std.math import sqrt

from std.utils.index import IndexList

from mantle import f32, nelts
from mantle.autograd.ops.matmul import dot
from mantle.core.tensorutils import (
    fill,
    elwise_transform,
    elwise_pow,
    elwise_op,
    broadcast_shapes,
    broadcast_elwise_op,
    get_reduce_shape,
    accumulate_grad,
    tsum,
    tmean,
    tstd,
    tmax,
    transpose,
)
from mantle.core.tensor import Tensor, TensorShape
from mantle.core.math_util import add, sub, mul, div, round_simd, exp, sqrt_simd

from tests import assert_tensors_equal


def test_zero() raises:
    var A = Tensor[f32](2, 3)
    var B = Tensor[f32](2, 3)
    rand[f32](B.mut_ptr(), B.num_elements())
    B.zero()
    assert_tensors_equal(A, B)


def test_fill() raises:
    var A = Tensor[f32](2, 3)
    var B = Tensor[f32](2, 3)
    for i in range(A.num_elements()):
        A[i] = 1.0
    fill(B, 1.0)
    assert_tensors_equal(A, B)


def test_dot() raises:
    comptime a_shape = TensorShape(2, 3)
    comptime b_shape = TensorShape(3, 2)
    var A = Tensor[f32](a_shape)
    var B = Tensor[f32](b_shape)
    fill(A, 1.0)
    fill(B, 1.0)

    var C = Tensor[f32](2, 2)
    dot[a_shape, b_shape](C, A, B)
    var C_expected = Tensor[f32](2, 2)
    fill(C_expected, 3.0)
    assert_tensors_equal(C, C_expected)

    var D = Tensor[f32](3, 3)
    dot[b_shape, a_shape](D, B, A)
    var D_expected = Tensor[f32](3, 3)
    fill(D_expected, 2.0)
    assert_tensors_equal(D, D_expected)


def test_elwise_transform() raises:
    var A = Tensor[f32](2, 10)
    var B = Tensor[f32](2, 10)
    var C = Tensor[f32](2, 10)
    var D = Tensor[f32](2, 10)
    fill(A, 4)
    fill(B, 2)
    fill(C, exp[f32, 1](SIMD[f32, 1](2.0)))
    fill(D, 7)

    var A_res = Tensor[f32](2, 10)
    elwise_transform[sqrt_simd](A_res, A)
    assert_tensors_equal(A_res, B)

    var B_res = Tensor[f32](2, 10)
    elwise_transform[exp](B_res, B)
    assert_tensors_equal(B_res, C)

    var C_res = Tensor[f32](2, 10)
    elwise_transform[round_simd](C_res, C)
    assert_tensors_equal(C_res, D)


def test_elwise_pow() raises:
    var A = Tensor[f32](1, 10)
    var B = Tensor[f32](1, 10)
    for i in range(10):
        A[i] = Float32(i)
        B[i] = Float32(i**2)

    var A_res = Tensor[f32](1, 10)
    elwise_pow(A_res, A, 2)
    assert_tensors_equal(A_res, B)


def test_elwise_tensor_tensor() raises:
    comptime t1_shape = TensorShape(2, 10)
    comptime t2_shape = TensorShape(2, 10)
    var t1 = Tensor[f32](t1_shape)
    var t2 = Tensor[f32](t2_shape)
    fill(t1, 3.0)
    fill(t2, 3.0)

    var result1 = Tensor[f32](2, 10)
    elwise_op[t1_shape, t2_shape, add](result1, t1, t2)
    var result1_expected = Tensor[f32](2, 10)
    fill(result1_expected, 6.0)
    assert_tensors_equal(result1, result1_expected)

    var result2 = Tensor[f32](2, 10)
    elwise_op[t1_shape, t2_shape, sub](result2, t1, t2)
    var result2_expected = Tensor[f32](2, 10)
    assert_tensors_equal(result2, result2_expected)

    var result3 = Tensor[f32](2, 10)
    elwise_op[t1_shape, t2_shape, mul](result3, t1, t2)
    var result3_expected = Tensor[f32](2, 10)
    fill(result3_expected, 9.0)
    assert_tensors_equal(result3, result3_expected)

    var result4 = Tensor[f32](2, 10)
    elwise_op[t1_shape, t2_shape, div](result4, t1, t2)
    var result4_expected = Tensor[f32](2, 10)
    fill(result4_expected, 1.0)
    assert_tensors_equal(result4, result4_expected)


def test_elwise_tensor_scalar() raises:
    var a: Scalar[f32] = 2.0
    var t1 = Tensor[f32](2, 10)
    fill(t1, 1.0)
    var result = Tensor[f32](2, 10)

    elwise_op[add](result, t1, a)
    var result1_expected = Tensor[f32](2, 10)
    fill(result1_expected, 3.0)
    assert_tensors_equal(result, result1_expected)

    elwise_op[add](result, a, t1)
    assert_tensors_equal(result, result1_expected)

    elwise_op[sub](result, t1, a)
    var result3_expected = Tensor[f32](2, 10)
    fill(result3_expected, -1)
    assert_tensors_equal(result, result3_expected)

    elwise_op[mul](result, a, t1)
    var result4_expected = Tensor[f32](2, 10)
    fill(result4_expected, 2)
    assert_tensors_equal(result, result4_expected)

    elwise_op[div](result, t1, a)
    var result5_expected = Tensor[f32](2, 10)
    fill(result5_expected, 0.5)
    assert_tensors_equal(result, result5_expected)


def test_elwise_broadcast_tensor() raises:
    comptime t1_shape = TensorShape(2, 3, 4)
    comptime t2_shape = TensorShape(5, 2, 1, 4)
    comptime res_shape = broadcast_shapes(t1_shape, t2_shape)
    var t1 = Tensor[f32](t1_shape)
    var t2 = Tensor[f32](t2_shape)

    fill(t1, 3.0)
    for i in range(40):
        t2[i] = Float32(i + 1)

    var result1 = Tensor[f32](res_shape)
    elwise_op[t1_shape, t2_shape, add](result1, t1, t2)
    var result1_expected = Tensor[f32](5, 2, 3, 4)
    # fill expected tensor
    for i in range(40):
        for j in range(3):
            var index = (i % 4) + ((i // 4) * 12) + j * 4
            result1_expected[index] = Float32(3.0) + Float32(i + 1)
    assert_tensors_equal(result1, result1_expected)


from test_tensorutils_data import SumMeanStdData


def test_sum_mean_std() raises:
    var t = Tensor[f32](2, 10)
    var s: Scalar[f32] = 0
    for i in range(20):
        t[i] = Float32(i + 1)
        s += Float32(i + 1)

    # Not specifying the axis takes all elements regardless of the shape
    var tensor_sum = tsum(t)
    assert_equal(tensor_sum, s)

    var tensor_mean = tmean(t)
    assert_equal(tensor_mean, Float32(s) / 20)

    var tensor_std = tstd(t)
    var expected_std: Scalar[f32] = 0
    for i in range(20):
        expected_std += (Float32(i + 1) - tensor_mean) ** 2
    expected_std = sqrt(expected_std / 20)
    assert_equal(tensor_std, expected_std)

    # When specifying the axis you can sum across batches
    # Axis 0
    var batch_sum_0 = Tensor[f32](get_reduce_shape(t.shape(), axis=0))
    tsum(batch_sum_0, t, axis=0)
    var expected_batch_sum_0 = Tensor[f32](1, 10)
    for i in range(10):
        expected_batch_sum_0[i] = Float32((i + 1) + (i + 1 + 10))
    assert_tensors_equal(batch_sum_0, expected_batch_sum_0)

    var batch_mean_0 = Tensor[f32](get_reduce_shape(t.shape(), axis=0))
    tmean(batch_mean_0, t, axis=0)
    var expected_batch_mean_0 = Tensor[f32](1, 10)
    for i in range(10):
        expected_batch_mean_0[i] = expected_batch_sum_0[i] / 2
    assert_tensors_equal(batch_mean_0, expected_batch_mean_0)

    var batch_std_0 = Tensor[f32](get_reduce_shape(t.shape(), axis=0))
    tstd(batch_std_0, t, axis=0)
    var expected_batch_std_0 = Tensor[f32](1, 10)
    fill(expected_batch_std_0, 5)
    assert_tensors_equal(batch_std_0, expected_batch_std_0)

    # Axis 1
    var batch_sum_1 = Tensor[f32](get_reduce_shape(t.shape(), axis=1))
    tsum(batch_sum_1, t, axis=1)
    var expected_batch_sum_1 = Tensor[f32](2, 1)
    expected_batch_sum_1[0] = Float32(1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10)
    expected_batch_sum_1[1] = Float32(11 + 12 + 13 + 14 + 15 + 16 + 17 + 18 + 19 + 20)
    assert_tensors_equal(batch_sum_1, expected_batch_sum_1)

    var batch_mean_1 = Tensor[f32](get_reduce_shape(t.shape(), axis=1))
    tmean(batch_mean_1, t, axis=1)
    var expected_batch_mean_1 = Tensor[f32](2, 1)
    expected_batch_mean_1[0] = expected_batch_sum_1[0] / 10
    expected_batch_mean_1[1] = expected_batch_sum_1[1] / 10
    assert_tensors_equal(batch_mean_1, expected_batch_mean_1)

    var batch_std_1 = Tensor[f32](get_reduce_shape(t.shape(), axis=1))
    tstd(batch_std_1, t, axis=1)
    var expected_batch_std_1 = Tensor[f32](2, 1)
    fill(expected_batch_std_1, 2.8722813129425049)
    assert_tensors_equal(batch_std_1, expected_batch_std_1)


def test_sum_mean_std_n() raises:
    var t = Tensor[f32](3, 4, 5)
    var s: Scalar[f32] = 0
    for i in range(60):
        t[i] = Float32(i + 1)
        s += Float32(i + 1)

    # Not specifying the axis takes all elements regardless of the shape
    var tensor_sum = tsum(t)
    assert_equal(tensor_sum, s)

    var tensor_mean = tmean(t)
    assert_equal(tensor_mean, Float32(s) / 60)

    var tensor_std = tstd(t)
    var expected_std: Scalar[f32] = 0
    for i in range(60):
        expected_std += (Float32(i + 1) - tensor_mean) ** 2
    expected_std = sqrt(expected_std / 60)
    assert_equal(tensor_std, expected_std)

    # When specifying the axis you can sum across batches
    # Axis 0
    var data = SumMeanStdData.generate_3d_axis_0()
    var batch_sum_0 = Tensor[f32](get_reduce_shape(t.shape(), axis=0))
    tsum(batch_sum_0, t, axis=0)
    assert_tensors_equal(batch_sum_0, data.expected_sum)

    var batch_mean_0 = Tensor[f32](get_reduce_shape(t.shape(), axis=0))
    tmean(batch_mean_0, t, axis=0)
    assert_tensors_equal(batch_mean_0, data.expected_mean)

    var batch_std_0 = Tensor[f32](get_reduce_shape(t.shape(), axis=0))
    tstd(batch_std_0, t, axis=0)
    assert_tensors_equal(batch_std_0, data.expected_std)

    # When specifying the axis you can sum across batches
    # Axis 1
    data = SumMeanStdData.generate_3d_axis_1()
    var batch_sum_1 = Tensor[f32](get_reduce_shape(t.shape(), axis=1))
    tsum(batch_sum_1, t, axis=1)
    assert_tensors_equal(batch_sum_1, data.expected_sum)

    var batch_mean_1 = Tensor[f32](get_reduce_shape(t.shape(), axis=1))
    tmean(batch_mean_1, t, axis=1)
    assert_tensors_equal(batch_mean_1, data.expected_mean)

    var batch_std_1 = Tensor[f32](get_reduce_shape(t.shape(), axis=1))
    tstd(batch_std_1, t, axis=1)
    assert_tensors_equal(batch_std_1, data.expected_std)

    # When specifying the axis you can sum across batches
    # Axis 2
    data = SumMeanStdData.generate_3d_axis_2()
    var batch_sum_2 = Tensor[f32](get_reduce_shape(t.shape(), axis=2))
    tsum(batch_sum_2, t, axis=2)
    assert_tensors_equal(batch_sum_2, data.expected_sum)

    var batch_mean_2 = Tensor[f32](get_reduce_shape(t.shape(), axis=2))
    tmean(batch_mean_2, t, axis=2)
    assert_tensors_equal(batch_mean_2, data.expected_mean)

    var batch_std_2 = Tensor[f32](get_reduce_shape(t.shape(), axis=2))
    tstd(batch_std_2, t, axis=2)
    assert_tensors_equal(batch_std_2, data.expected_std)


def test_max() raises:
    var t = Tensor[f32](2, 3, 2)
    for i in range(12):
        t[i] = Float32(i + 1)

    var tensor_max = tmax(t)
    assert_equal(tensor_max, Float32(12))

    def fill_tensor[
        size: Int
    ](mut tensor: Tensor[f32], values: IndexList[size]):
        for i in range(tensor.num_elements()):
            tensor[i] = Float32(values[i])

    var tensor_max_axis_0 = Tensor[f32](get_reduce_shape(t.shape(), axis=0))
    tmax(tensor_max_axis_0, t, axis=0)
    var expected_max_axis_0_temp = IndexList[6](7, 8, 9, 10, 11, 12)
    var expected_max_axis_0 = Tensor[f32](1, 3, 2)
    fill_tensor(expected_max_axis_0, expected_max_axis_0_temp)
    assert_tensors_equal(tensor_max_axis_0, expected_max_axis_0)

    var tensor_max_axis_1 = Tensor[f32](get_reduce_shape(t.shape(), axis=1))
    tmax(tensor_max_axis_1, t, axis=1)
    var expected_max_axis_1_temp = IndexList[4](5, 6, 11, 12)
    var expected_max_axis_1 = Tensor[f32](2, 1, 2)
    fill_tensor(expected_max_axis_1, expected_max_axis_1_temp)
    assert_tensors_equal(tensor_max_axis_1, expected_max_axis_1)

    var tensor_max_axis_2 = Tensor[f32](get_reduce_shape(t.shape(), axis=2))
    tmax(tensor_max_axis_2, t, axis=2)
    var expected_max_axis_2_temp = IndexList[6](2, 4, 6, 8, 10, 12)
    var expected_max_axis_2 = Tensor[f32](2, 3, 1)
    fill_tensor(expected_max_axis_2, expected_max_axis_2_temp)
    assert_tensors_equal(tensor_max_axis_2, expected_max_axis_2)


from test_tensorutils_data import TransposeData


def test_transpose() raises:
    # Transpose 2D
    var data = TransposeData.generate_1_2dim_test_case()
    var transposed = transpose(data.A, TensorShape(data.transpose_dims))
    assert_tensors_equal(transposed, data.expected)

    # Transpose 2 dimensions
    data = TransposeData.generate_2_2dim_test_case()
    transposed = transpose(data.A, TensorShape(data.transpose_dims))
    assert_tensors_equal(transposed, data.expected)

    data = TransposeData.generate_3_2dim_test_case()
    transposed = transpose(data.A, TensorShape(data.transpose_dims))
    assert_tensors_equal(transposed, data.expected)

    data = TransposeData.generate_4_2dim_test_case()
    transposed = transpose(data.A, TensorShape(data.transpose_dims))
    assert_tensors_equal(transposed, data.expected)

    # Transpose all dimensions
    data = TransposeData.generate_1_alldim_test_case()

    transposed = transpose(data.A, TensorShape(data.transpose_dims))
    assert_tensors_equal(transposed, data.expected)

    data = TransposeData.generate_2_alldim_test_case()

    transposed = transpose(data.A, TensorShape(data.transpose_dims))
    assert_tensors_equal(transposed, data.expected)

    # Transpose (reverse)
    data = TransposeData.generate_1_transpose_test_case()
    transposed = transpose(data.A, TensorShape(data.transpose_dims))
    assert_tensors_equal(transposed, data.expected)


def test_accumulate_grad() raises:
    comptime A_shape = TensorShape(2, 3, 4)
    comptime B_shape = TensorShape(2, 1, 1)
    var A = Tensor[f32](A_shape)
    var B = Tensor[f32](B_shape)
    fill(A, 3.0)

    accumulate_grad[B_shape, A_shape](B, A)
    var expected = Tensor[f32](2, 1, 1)
    fill(expected, 36)
    assert_tensors_equal(B, expected)

    comptime B_shape_2 = TensorShape(2, 1)
    B = Tensor[f32](B_shape_2)
    accumulate_grad[B_shape_2, A_shape](B, A)
    expected = Tensor[f32](2, 1)
    fill(expected, 24)
    assert_tensors_equal(B, expected)


# from test_tensorutils_data import PaddingData

# def test_padding() raises:
#     # 1D padding (only after)
#     var data = PaddingData.generate_1d_test_case_after()
#     var padded_data = pad_zeros[f32, nelts](data.A, data.pad_with)
#     assert_tensors_equal(padded_data, data.expected)

#     # 1D padding (before and after)
#     data = PaddingData.generate_1d_test_case_before_after()
#     padded_data = pad_zeros[f32, nelts](data.A, data.pad_with)
#     assert_tensors_equal(padded_data, data.expected)

#     # 2D padding
#     data = PaddingData.generate_2d_test_case()
#     padded_data = pad_zeros[f32, nelts](data.A, data.pad_with)
#     assert_tensors_equal(padded_data, data.expected)

#     # 3D padding (simple)
#     data = PaddingData.generate_3d_test_case_simple()
#     padded_data = pad_zeros[f32, nelts](data.A, data.pad_with)
#     assert_tensors_equal(padded_data, data.expected)

#     # 3D padding
#     data = PaddingData.generate_3d_test_case()
#     padded_data = pad_zeros[f32, nelts](data.A, data.pad_with)
#     assert_tensors_equal(padded_data, data.expected)

#     # 4D padding
#     data = PaddingData.generate_4d_test_case()
#     padded_data = pad_zeros[f32, nelts](data.A, data.pad_with)
#     assert_tensors_equal(padded_data, data.expected)


def main():
    try:
        test_zero()
        test_fill()
        test_dot()
        test_elwise_transform()
        test_elwise_pow()
        test_elwise_tensor_tensor()
        test_elwise_tensor_scalar()
        test_elwise_broadcast_tensor()
        test_sum_mean_std()
        test_sum_mean_std_n()
        test_max()
        test_transpose()
        test_accumulate_grad()
        # # test_padding()
    except e:
        print("[ERROR] Error in tensorutils.py")
        print(e)

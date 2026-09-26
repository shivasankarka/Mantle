"""Phase benchmark for the GPU CNN backward kernels."""
from std.random import rand
from std.time import perf_counter_ns as now

from mantle import Tensor, TensorShape, f32
from mantle.core.device import Device
from mantle.autograd.ops.gpu_conv import (
    gpu_conv2d_forward,
    gpu_conv2d_input_backward,
    gpu_conv2d_parameter_backward_direct,
)
from mantle.autograd.ops.gpu_pool import (
    gpu_maxpool2d_forward,
    gpu_maxpool2d_backward,
)


comptime STEPS = 20


def random_gpu(shape: TensorShape) raises -> Tensor[f32, Device.gpu]:
    var host = Tensor[f32](shape)
    rand[f32](host.ptr(), host.num_elements())
    return host.to_gpu()


def main() raises:
    var conv1_inputs = random_gpu(TensorShape(64, 1, 28, 28))
    var conv1_upper_grad = random_gpu(TensorShape(64, 16, 28, 28))
    var conv1_kernel = random_gpu(TensorShape(16, 1, 5, 5))
    var conv1_bias = random_gpu(TensorShape(16))
    var conv1_output = Tensor[f32, Device.gpu](
        TensorShape(64, 16, 28, 28), uninitialized=True
    )
    var conv1_kernel_grad = Tensor[f32, Device.gpu](
        TensorShape(16, 1, 5, 5), uninitialized=True
    )
    var conv1_bias_grad = Tensor[f32, Device.gpu](
        TensorShape(16), uninitialized=True
    )
    gpu_conv2d_forward[
        64,
        1,
        28,
        28,
        16,
        5,
        5,
        28,
        28,
        2,
        2,
        1,
        1,
        1,
        1,
    ](conv1_output, conv1_inputs, conv1_kernel, conv1_bias)
    conv1_inputs.gpu_context().synchronize()
    var conv1_forward_start = now()
    for _ in range(STEPS):
        gpu_conv2d_forward[
            64,
            1,
            28,
            28,
            16,
            5,
            5,
            28,
            28,
            2,
            2,
            1,
            1,
            1,
            1,
        ](conv1_output, conv1_inputs, conv1_kernel, conv1_bias)
    conv1_inputs.gpu_context().synchronize()
    print(
        "KERNEL, conv1-forward,",
        Float64(now() - conv1_forward_start) / 1e9,
    )
    gpu_conv2d_parameter_backward_direct[
        64,
        1,
        28,
        28,
        16,
        5,
        5,
        28,
        28,
        2,
        2,
        1,
        1,
        1,
        1,
    ](
        conv1_kernel_grad,
        conv1_bias_grad,
        conv1_inputs,
        conv1_upper_grad,
    )
    conv1_inputs.gpu_context().synchronize()
    var conv1_start = now()
    for _ in range(STEPS):
        gpu_conv2d_parameter_backward_direct[
            64,
            1,
            28,
            28,
            16,
            5,
            5,
            28,
            28,
            2,
            2,
            1,
            1,
            1,
            1,
        ](
            conv1_kernel_grad,
            conv1_bias_grad,
            conv1_inputs,
            conv1_upper_grad,
        )
    conv1_inputs.gpu_context().synchronize()
    print(
        "KERNEL, conv1-parameter-backward-direct,",
        Float64(now() - conv1_start) / 1e9,
    )

    # Second MNIST convolution: NCHW 64x16x14x14 -> 64x32x14x14.
    var inputs = random_gpu(TensorShape(64, 16, 14, 14))
    var kernel = random_gpu(TensorShape(32, 16, 5, 5))
    var upper_grad = random_gpu(TensorShape(64, 32, 14, 14))
    var bias = random_gpu(TensorShape(32))
    var output = Tensor[f32, Device.gpu](
        TensorShape(64, 32, 14, 14), uninitialized=True
    )
    var input_grad = Tensor[f32, Device.gpu](
        TensorShape(64, 16, 14, 14), uninitialized=True
    )
    var kernel_grad = Tensor[f32, Device.gpu](
        TensorShape(32, 16, 5, 5), uninitialized=True
    )
    var bias_grad = Tensor[f32, Device.gpu](TensorShape(32), uninitialized=True)

    gpu_conv2d_forward[
        64,
        16,
        14,
        14,
        32,
        5,
        5,
        14,
        14,
        2,
        2,
        1,
        1,
        1,
        1,
    ](output, inputs, kernel, bias)
    inputs.gpu_context().synchronize()
    var conv2_forward_start = now()
    for _ in range(STEPS):
        gpu_conv2d_forward[
            64,
            16,
            14,
            14,
            32,
            5,
            5,
            14,
            14,
            2,
            2,
            1,
            1,
            1,
            1,
        ](output, inputs, kernel, bias)
    inputs.gpu_context().synchronize()
    print(
        "KERNEL, conv2-forward,",
        Float64(now() - conv2_forward_start) / 1e9,
    )

    gpu_conv2d_input_backward[
        64,
        16,
        14,
        14,
        32,
        5,
        5,
        14,
        14,
        2,
        2,
        1,
        1,
        1,
        1,
    ](input_grad, upper_grad, kernel)
    inputs.gpu_context().synchronize()

    var start = now()
    for _ in range(STEPS):
        gpu_conv2d_input_backward[
            64,
            16,
            14,
            14,
            32,
            5,
            5,
            14,
            14,
            2,
            2,
            1,
            1,
            1,
            1,
        ](input_grad, upper_grad, kernel)
    inputs.gpu_context().synchronize()
    print("KERNEL, conv2-input-backward,", Float64(now() - start) / 1e9)

    gpu_conv2d_parameter_backward_direct[
        64,
        16,
        14,
        14,
        32,
        5,
        5,
        14,
        14,
        2,
        2,
        1,
        1,
        1,
        1,
    ](kernel_grad, bias_grad, inputs, upper_grad)
    inputs.gpu_context().synchronize()
    start = now()
    for _ in range(STEPS):
        gpu_conv2d_parameter_backward_direct[
            64,
            16,
            14,
            14,
            32,
            5,
            5,
            14,
            14,
            2,
            2,
            1,
            1,
            1,
            1,
        ](kernel_grad, bias_grad, inputs, upper_grad)
    inputs.gpu_context().synchronize()
    print(
        "KERNEL, conv2-parameter-backward-direct,",
        Float64(now() - start) / 1e9,
    )

    var pool_inputs = random_gpu(TensorShape(64, 32, 14, 14))
    var pool_upper_grad = random_gpu(TensorShape(64, 32, 7, 7))
    var pool_input_grad = Tensor[f32, Device.gpu](
        TensorShape(64, 32, 14, 14), uninitialized=True
    )
    var pool_output = Tensor[f32, Device.gpu](
        TensorShape(64, 32, 7, 7), uninitialized=True
    )
    gpu_maxpool2d_forward[
        64,
        32,
        14,
        14,
        7,
        7,
        2,
        2,
        0,
        0,
        2,
        2,
        1,
        1,
    ](pool_output, pool_inputs)
    inputs.gpu_context().synchronize()
    start = now()
    for _ in range(STEPS):
        gpu_maxpool2d_forward[
            64,
            32,
            14,
            14,
            7,
            7,
            2,
            2,
            0,
            0,
            2,
            2,
            1,
            1,
        ](pool_output, pool_inputs)
    inputs.gpu_context().synchronize()
    print("KERNEL, pool2-forward,", Float64(now() - start) / 1e9)
    gpu_maxpool2d_backward[
        64,
        32,
        14,
        14,
        7,
        7,
        2,
        2,
        0,
        0,
        2,
        2,
        1,
        1,
    ](pool_input_grad, pool_upper_grad, pool_inputs)
    inputs.gpu_context().synchronize()
    start = now()
    for _ in range(STEPS):
        gpu_maxpool2d_backward[
            64,
            32,
            14,
            14,
            7,
            7,
            2,
            2,
            0,
            0,
            2,
            2,
            1,
            1,
        ](pool_input_grad, pool_upper_grad, pool_inputs)
    inputs.gpu_context().synchronize()
    print("KERNEL, pool2-backward,", Float64(now() - start) / 1e9)

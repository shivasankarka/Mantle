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
from std.math import ceildiv, sqrt, log, cos, pi
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from std.random.philox import Random
from std.random import random_ui64

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


def gpu_rand_uniform(
    mut res: Tensor[f32, Device.gpu], low: Scalar[f32], high: Scalar[f32]
) raises:
    """Fills `res` with values drawn from a uniform distribution, entirely
    on-device (Philox counter-based RNG, one independent stream per
    element)."""
    var ctx = res.gpu_context()
    var n = res.num_elements()
    var seed = random_ui64(0, UInt64.MAX)
    ctx.compile_function[_rand_uniform_kernel]()._call_with_pack_checked(
        ctx, res.gpu_ptr(), Int64(n), seed, low, high,
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
    )
    ctx.synchronize()


def gpu_rand_normal(
    mut res: Tensor[f32, Device.gpu], mean: Scalar[f32], std: Scalar[f32]
) raises:
    """Fills `res` with values drawn from a normal distribution, entirely
    on-device (Box-Muller over a Philox stream per element)."""
    var ctx = res.gpu_context()
    var n = res.num_elements()
    var seed = random_ui64(0, UInt64.MAX)
    ctx.compile_function[_rand_normal_kernel]()._call_with_pack_checked(
        ctx, res.gpu_ptr(), Int64(n), seed, mean, std,
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
    )
    ctx.synchronize()


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
    ctx.compile_function[_adam_step_kernel]()._call_with_pack_checked(
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
    ctx.synchronize()


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

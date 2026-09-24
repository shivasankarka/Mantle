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
from std.math import ceildiv, sqrt, log, cos, pi, pow
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
    func: def(*args: *declared_arg_types) thin -> None,
]() -> type_of(_shared_device_context().compile_function[func]()):
    """A kernel's `DeviceFunction` (compiled pipeline state) is expensive to
    create and constant for the process, so each kernel gets exactly one,
    cached the same way `_shared_device_context` caches the `DeviceContext`
    itself — otherwise every single launch would recompile it."""
    try:
        return _shared_device_context().compile_function[func]()
    except e:
        abort("Mantle: GPU kernel compile failed: " + String(e))


def _add_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    a: Pointer[Scalar[f32], MutAnyOrigin],
    b: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        res.unsafe_store(Int(i), a.unsafe_load(Int(i)) + b.unsafe_load(Int(i)))


comptime _add_kernel_global = _Global["mantle_gpu_kernel_add", _make_kernel_fn[_add_kernel]]


def _cached_add_kernel() raises -> type_of(_shared_device_context().compile_function[_add_kernel]()):
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


comptime _sub_kernel_global = _Global["mantle_gpu_kernel_sub", _make_kernel_fn[_sub_kernel]]


def _cached_sub_kernel() raises -> type_of(_shared_device_context().compile_function[_sub_kernel]()):
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


comptime _mul_kernel_global = _Global["mantle_gpu_kernel_mul", _make_kernel_fn[_mul_kernel]]


def _cached_mul_kernel() raises -> type_of(_shared_device_context().compile_function[_mul_kernel]()):
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


comptime _div_kernel_global = _Global["mantle_gpu_kernel_div", _make_kernel_fn[_div_kernel]]


def _cached_div_kernel() raises -> type_of(_shared_device_context().compile_function[_div_kernel]()):
    return _div_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def _accumulate_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    other: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        res.unsafe_store(Int(i), res.unsafe_load(Int(i)) + other.unsafe_load(Int(i)))


comptime _accumulate_kernel_global = _Global[
    "mantle_gpu_kernel_accumulate", _make_kernel_fn[_accumulate_kernel]
]


def _cached_accumulate_kernel() raises -> type_of(_shared_device_context().compile_function[_accumulate_kernel]()):
    return _accumulate_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def _neg_kernel(
    res: Pointer[Scalar[f32], MutAnyOrigin],
    a: Pointer[Scalar[f32], MutAnyOrigin],
    n: Int64,
):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        res.unsafe_store(Int(i), -a.unsafe_load(Int(i)))


comptime _neg_kernel_global = _Global["mantle_gpu_kernel_neg", _make_kernel_fn[_neg_kernel]]


def _cached_neg_kernel() raises -> type_of(_shared_device_context().compile_function[_neg_kernel]()):
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


comptime _relu_kernel_global = _Global["mantle_gpu_kernel_relu", _make_kernel_fn[_relu_kernel]]


def _cached_relu_kernel() raises -> type_of(_shared_device_context().compile_function[_relu_kernel]()):
    return _relu_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


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


comptime _relu_bw_kernel_global = _Global[
    "mantle_gpu_kernel_relu_bw", _make_kernel_fn[_relu_bw_kernel]
]


def _cached_relu_bw_kernel() raises -> type_of(_shared_device_context().compile_function[_relu_bw_kernel]()):
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
            Int(i), -t1.unsafe_load(Int(i)) / (t2v * t2v) * ug.unsafe_load(Int(i))
        )


comptime _div_bw_t2_kernel_global = _Global[
    "mantle_gpu_kernel_div_bw_t2", _make_kernel_fn[_div_bw_t2_kernel]
]


def _cached_div_bw_t2_kernel() raises -> type_of(_shared_device_context().compile_function[_div_bw_t2_kernel]()):
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


def _cached_add_bias_kernel() raises -> type_of(_shared_device_context().compile_function[_add_bias_kernel]()):
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


def _cached_bias_grad_kernel() raises -> type_of(_shared_device_context().compile_function[_bias_grad_kernel]()):
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


comptime _pow_kernel_global = _Global["mantle_gpu_kernel_pow", _make_kernel_fn[_pow_kernel]]


def _cached_pow_kernel() raises -> type_of(_shared_device_context().compile_function[_pow_kernel]()):
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


def _cached_pow_bw_kernel() raises -> type_of(_shared_device_context().compile_function[_pow_bw_kernel]()):
    return _pow_bw_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


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


comptime _mean_kernel_global = _Global["mantle_gpu_kernel_mean", _make_kernel_fn[_mean_kernel]]


def _cached_mean_kernel() raises -> type_of(_shared_device_context().compile_function[_mean_kernel]()):
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


def _cached_mean_bw_kernel() raises -> type_of(_shared_device_context().compile_function[_mean_bw_kernel]()):
    return _mean_bw_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


def gpu_pow_forward(
    mut res: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    exponent: Tensor[f32, Device.gpu],
) raises:
    var ctx = res.gpu_context()
    var n = res.num_elements()
    _cached_pow_kernel()._call_with_pack_checked(
        ctx, res.gpu_ptr(), t1.gpu_ptr(), exponent.gpu_ptr(), Int64(n),
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
    )


def gpu_pow_backward(
    ug: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    exponent: Tensor[f32, Device.gpu],
) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](t1.shape(), uninitialized=True)
    var ctx = res_grad.gpu_context()
    var n = res_grad.num_elements()
    _cached_pow_bw_kernel()._call_with_pack_checked(
        ctx, res_grad.gpu_ptr(), t1.gpu_ptr(), ug.gpu_ptr(), exponent.gpu_ptr(), Int64(n),
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
    )
    return res_grad^


def gpu_mean_forward(
    mut res: Tensor[f32, Device.gpu], t1: Tensor[f32, Device.gpu]
) raises:
    var ctx = res.gpu_context()
    var n = t1.num_elements()
    _cached_mean_kernel()._call_with_pack_checked(
        ctx, res.gpu_ptr(), t1.gpu_ptr(), Int64(n),
        grid_dim=1, block_dim=1,
    )


def gpu_mean_backward(
    ug: Tensor[f32, Device.gpu], t_shape: TensorShape
) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](t_shape, uninitialized=True)
    var ctx = res_grad.gpu_context()
    var n = res_grad.num_elements()
    _cached_mean_bw_kernel()._call_with_pack_checked(
        ctx, res_grad.gpu_ptr(), ug.gpu_ptr(), Int64(n),
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
    )
    return res_grad^


def gpu_add_bias_forward(
    mut res: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    bias: Tensor[f32, Device.gpu],
) raises:
    """res = t1 + bias, broadcasting `bias` (rank 1) over t1's trailing
    dim."""
    var ctx = res.gpu_context()
    var total = res.num_elements()
    var n = bias.num_elements()
    _cached_add_bias_kernel()._call_with_pack_checked(
        ctx, res.gpu_ptr(), t1.gpu_ptr(), bias.gpu_ptr(), Int64(n), Int64(total),
        grid_dim=ceildiv(total, _BLOCK), block_dim=min(total, _BLOCK),
    )


def gpu_bias_grad(
    ug: Tensor[f32, Device.gpu], n: Int
) raises -> Tensor[f32, Device.gpu]:
    """Reduces `ug`'s gradient back down to the broadcast bias's shape."""
    var res_grad = Tensor[f32, Device.gpu](TensorShape(n), uninitialized=True)
    var ctx = res_grad.gpu_context()
    var total = ug.num_elements()
    var outer = total // n
    _cached_bias_grad_kernel()._call_with_pack_checked(
        ctx, res_grad.gpu_ptr(), ug.gpu_ptr(), Int64(n), Int64(outer),
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
    )
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


def _cached_rand_uniform_kernel() raises -> type_of(_shared_device_context().compile_function[_rand_uniform_kernel]()):
    return _rand_uniform_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


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


def _cached_rand_normal_kernel() raises -> type_of(_shared_device_context().compile_function[_rand_normal_kernel]()):
    return _rand_normal_kernel_global.get_or_create_ptr()[unsafe_offset=0].copy()


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
        ctx, res.gpu_ptr(), Int64(n), seed, low, high,
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
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
        ctx, res.gpu_ptr(), Int64(n), seed, mean, std,
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
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


def _cached_adam_step_kernel() raises -> type_of(_shared_device_context().compile_function[_adam_step_kernel]()):
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
        ctx, res.gpu_ptr(), t1.gpu_ptr(), t2.gpu_ptr(), Int64(n),
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
    )


def gpu_sub_forward(
    mut res: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    t2: Tensor[f32, Device.gpu],
) raises:
    var ctx = res.gpu_context()
    var n = res.num_elements()
    _cached_sub_kernel()._call_with_pack_checked(
        ctx, res.gpu_ptr(), t1.gpu_ptr(), t2.gpu_ptr(), Int64(n),
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
    )


def gpu_mul_forward(
    mut res: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    t2: Tensor[f32, Device.gpu],
) raises:
    var ctx = res.gpu_context()
    var n = res.num_elements()
    _cached_mul_kernel()._call_with_pack_checked(
        ctx, res.gpu_ptr(), t1.gpu_ptr(), t2.gpu_ptr(), Int64(n),
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
    )


def gpu_div_forward(
    mut res: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    t2: Tensor[f32, Device.gpu],
) raises:
    var ctx = res.gpu_context()
    var n = res.num_elements()
    _cached_div_kernel()._call_with_pack_checked(
        ctx, res.gpu_ptr(), t1.gpu_ptr(), t2.gpu_ptr(), Int64(n),
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
    )


def gpu_accumulate_grad(
    mut grad: Tensor[f32, Device.gpu], res_grad: Tensor[f32, Device.gpu]
) raises:
    """`grad += res_grad`, elementwise (no broadcasting)."""
    var ctx = grad.gpu_context()
    var n = grad.num_elements()
    _cached_accumulate_kernel()._call_with_pack_checked(
        ctx, grad.gpu_ptr(), res_grad.gpu_ptr(), Int64(n),
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
    )


def gpu_relu_forward(
    mut res: Tensor[f32, Device.gpu], t1: Tensor[f32, Device.gpu]
) raises:
    var ctx = res.gpu_context()
    var n = res.num_elements()
    _cached_relu_kernel()._call_with_pack_checked(
        ctx, res.gpu_ptr(), t1.gpu_ptr(), Int64(n),
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
    )


def gpu_relu_backward(
    ug: Tensor[f32, Device.gpu], t1: Tensor[f32, Device.gpu]
) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](ug.shape(), uninitialized=True)
    var ctx = res_grad.gpu_context()
    var n = res_grad.num_elements()
    _cached_relu_bw_kernel()._call_with_pack_checked(
        ctx, res_grad.gpu_ptr(), t1.gpu_ptr(), ug.gpu_ptr(), Int64(n),
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
    )
    return res_grad^


def gpu_sub_backward_t2(ug: Tensor[f32, Device.gpu]) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](ug.shape(), uninitialized=True)
    var ctx = res_grad.gpu_context()
    var n = res_grad.num_elements()
    _cached_neg_kernel()._call_with_pack_checked(
        ctx, res_grad.gpu_ptr(), ug.gpu_ptr(), Int64(n),
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
    )
    return res_grad^


def gpu_mul_backward(
    ug: Tensor[f32, Device.gpu], other: Tensor[f32, Device.gpu]
) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](ug.shape(), uninitialized=True)
    var ctx = res_grad.gpu_context()
    var n = res_grad.num_elements()
    _cached_mul_kernel()._call_with_pack_checked(
        ctx, res_grad.gpu_ptr(), ug.gpu_ptr(), other.gpu_ptr(), Int64(n),
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
    )
    return res_grad^


def gpu_div_backward_t1(
    ug: Tensor[f32, Device.gpu], t2: Tensor[f32, Device.gpu]
) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](ug.shape(), uninitialized=True)
    var ctx = res_grad.gpu_context()
    var n = res_grad.num_elements()
    _cached_div_kernel()._call_with_pack_checked(
        ctx, res_grad.gpu_ptr(), ug.gpu_ptr(), t2.gpu_ptr(), Int64(n),
        grid_dim=ceildiv(n, _BLOCK), block_dim=min(n, _BLOCK),
    )
    return res_grad^


def gpu_div_backward_t2(
    ug: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    t2: Tensor[f32, Device.gpu],
) raises -> Tensor[f32, Device.gpu]:
    var res_grad = Tensor[f32, Device.gpu](ug.shape(), uninitialized=True)
    var ctx = res_grad.gpu_context()
    var n = res_grad.num_elements()
    _cached_div_bw_t2_kernel()._call_with_pack_checked(
        ctx,
        res_grad.gpu_ptr(),
        t1.gpu_ptr(),
        t2.gpu_ptr(),
        ug.gpu_ptr(),
        Int64(n),
        grid_dim=ceildiv(n, _BLOCK),
        block_dim=min(n, _BLOCK),
    )
    return res_grad^

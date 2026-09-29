# ===----------------------------------------------------------------------=== #
# Mantle: Optimizers
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Optim (mantle.nn.optim)
------------------------------------------------
Optimizer implementations (Adam, AdamW, SGD), LR schedulers, and gradient
utilities.
"""
from std.math import sqrt, cos
from std.algorithm import vectorize
from max.algorithm import parallelize
from std.sys.info import simd_width_of

comptime PI = Float64(3.14159265358979323846)

from mantle import f32
from mantle.nn.parameters import Parameters
from mantle.autograd.graph import Graph
from mantle.autograd.symbol import Symbol
from mantle.core.tensor import Tensor, TensorShape
from mantle.autograd.collection import Collection
from mantle.autograd.ops import OP
from mantle.core.device import Device
from mantle.core.math_util import add, sub, mul, div
from mantle.autograd.ops.gpu_elementwise import gpu_adam_step, gpu_adamw_step


trait Optimizer:
    """Minimal optimizer interface used by model training helpers."""

    def zero_grad(mut self) raises:
        ...

    def step(mut self) raises:
        ...


# ===----------------------------------------------------------------------===#
# Helpers
# ===----------------------------------------------------------------------===#


def get_trainable_parameters(g: Graph) -> List[Symbol]:
    """
    Get all symbols of trainable parameters.

    Args:
        g: The computational graph.

    Returns:
        A list of symbols corresponding to trainable parameters.
    """

    var trainable_parameters = List[Symbol]()

    for i in range(len(g.params)):
        if g.params.symbols[i].trainable:
            trainable_parameters.append(g.params.symbols[i])

    return trainable_parameters^


def get_direct_overwrite_gradients(g: Graph) -> List[Symbol]:
    """Return gradients written directly by a single native GPU backward op.

    A gradient buffer may skip ``zero_grad`` only if one downstream node
    consumes its value and that exact backward path overwrites the buffer.
    This deliberately excludes shared values, broadcasts, and every operator
    still using a temporary-plus-accumulate implementation.
    """
    var direct_gradients = List[Symbol]()
    for node in g.nodes:
        for input_id in range(len(node.inputs)):
            var symbol = node.inputs[input_id]
            if not symbol.trainable:
                continue

            var supports_overwrite = (
                node.operator == OP.LINEAR
                or (
                    (
                        node.operator == OP.RELU
                        or node.operator == OP.EXP
                        or node.operator == OP.LOG
                        or node.operator == OP.GELU
                        or node.operator == OP.DROPOUT
                        or node.operator == OP.SQRT
                        or node.operator == OP.RESHAPE
                        or node.operator == OP.FLATTEN
                    )
                    and input_id == 0
                )
                or (
                    node.operator == OP.MEAN
                    and input_id == 0
                    and not node.attributes["axis"]
                )
                or (
                    (
                        node.operator == OP.SUM
                        or node.operator == OP.MEAN
                        or node.operator == OP.MAX
                    )
                    and input_id == 0
                    and node.attributes["axis"]
                    and node.attributes["axis"].value().to_int()
                    == node.inputs[0].shape.rank() - 1
                )
                or (node.operator == OP.POW and input_id == 0)
                or (
                    (
                        node.operator == OP.SUB
                        or node.operator == OP.MUL
                        or node.operator == OP.DIV
                    )
                    and node.inputs[0].shape == node.inputs[1].shape
                )
                or (
                    node.operator == OP.ADD
                    and node.inputs[0].shape == node.inputs[1].shape
                )
                or (
                    node.operator == OP.ADD
                    and input_id == 0
                    and node.inputs[0].shape.rank() >= 2
                    and node.inputs[1].shape.rank() == 1
                    and node.inputs[1].shape[0] == node.inputs[0].shape[-1]
                )
                or (
                    node.operator == OP.ADD
                    and input_id == 1
                    and node.inputs[0].shape.rank() >= 2
                    and node.inputs[1].shape.rank() == 1
                    and node.inputs[1].shape[0] == node.inputs[0].shape[-1]
                )
                or (
                    node.operator == OP.DOT
                    and node.inputs[0].shape.rank() == 2
                    and node.inputs[1].shape.rank() == 2
                )
                or (
                    node.operator == OP.TRANSPOSE
                    and input_id == 0
                    and node.inputs[0].shape.rank() == 4
                )
                or node.operator == OP.MAXPOOL2D
                or node.operator == OP.CONV2D
            )
            if not supports_overwrite:
                continue

            var users = 0
            for other_node in g.nodes:
                for other_input in other_node.inputs:
                    if other_input == symbol:
                        users += 1
            if users == 1:
                direct_gradients.append(symbol)
    return direct_gradients^


# ===----------------------------------------------------------------------===#
# Gradient Clipping
# ===----------------------------------------------------------------------===#


def clip_grad_norm[
    g: Graph,
    trainable_parameters: List[Symbol] = get_trainable_parameters(g),
](mut parameters: Parameters[Device.cpu], max_norm: Scalar[f32]) -> Scalar[f32]:
    """
    Clip gradients by global L2 norm (in-place).

    Computes the total L2 norm of all trainable parameter gradients, then
    rescales every gradient by `max_norm / norm` when `norm > max_norm`.
    This is the standard PyTorch `clip_grad_norm_` behaviour.

    Args:
        parameters: The model's parameter/gradient storage.
        max_norm:   Maximum allowed gradient norm.

    Returns:
        The pre-clipping global gradient norm (useful for monitoring).
    """
    var tr = materialize[trainable_parameters]()

    # --- compute global L2 norm ---
    var total_norm: Scalar[f32] = 0.0

    for i in range(len(tr)):
        var param = tr[i]
        var n = param.shape.num_elements()

        def v_norm[
            nelts: Int
        ](j: Int) {mut total_norm, imm param, imm parameters}:
            var g_vec = parameters.grads[param].load[nelts](j)
            total_norm += (g_vec * g_vec).reduce_add()

        vectorize[simd_width_of[f32]()](n, v_norm)

    total_norm = sqrt(total_norm)

    # --- rescale if norm exceeds max_norm ---
    if total_norm > max_norm:
        var scale = max_norm / (total_norm + Scalar[f32](1e-6))

        for i in range(len(tr)):
            var param = tr[i]
            var n = param.shape.num_elements()

            def v_scale[
                nelts: Int
            ](j: Int) {mut parameters, imm param, imm scale}:
                var g_vec = parameters.grads[param].load[nelts](j)
                parameters.grads[param].store[nelts](j, g_vec * scale)

            vectorize[simd_width_of[f32]()](n, v_scale)

    return total_norm


# ===----------------------------------------------------------------------===#
# Adam
# ===----------------------------------------------------------------------===#


struct Adam[
    g: Graph,
    trainable_parameters: List[Symbol] = get_trainable_parameters(g),
    device: Device = Device.cpu,
](Optimizer):
    var parameters: Pointer[Parameters[Self.device], MutUntrackedOrigin]

    var lr: Scalar[f32]
    var beta1: Scalar[f32]
    var beta2: Scalar[f32]
    var epsilon: Scalar[f32]
    var iter: Int

    var rms_grads: Collection[Self.device]
    var momentum_grads: Collection[Self.device]
    var direct_overwrite_gradients: List[Symbol]

    def __init__(
        out self,
        ref[MutAnyOrigin] parameters: Parameters[Self.device],
        lr: Scalar[f32] = 0.001,
        beta1: Scalar[f32] = 0.9,
        beta2: Scalar[f32] = 0.999,
        epsilon: Scalar[f32] = 1e-8,
    ):
        self.parameters = Pointer(to=parameters).unsafe_origin_cast[
            MutUntrackedOrigin
        ]()

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iter = 0

        var tr = materialize[Self.trainable_parameters]()
        # Capacity of the collections should be the n of trainable parameters
        self.rms_grads = Collection[Self.device](capacity=len(tr))
        self.momentum_grads = Collection[Self.device](capacity=len(tr))
        self.direct_overwrite_gradients = get_direct_overwrite_gradients(Self.g)

        self.allocate_rms_and_momentum()

    def zero_grad(mut self) raises:
        """Set all gradients to zero."""
        comptime if Self.device.id == Device.cpu.id:
            self.parameters[].grads.set_zero()
        else:
            self.parameters[].grads.set_zero_except(
                self.direct_overwrite_gradients
            )

    def zero_grad(mut self, mut parameters: Parameters[Self.device]) raises:
        """Set gradients to zero using an explicit model storage reference.

        This is the stable GPU path for large mixed-operator graphs.  It
        avoids relying on a raw pointer through a deeply-specialized kernel
        call while preserving the usual all-device execution.
        """
        comptime if Self.device.id == Device.cpu.id:
            parameters.grads.set_zero()
        else:
            parameters.grads.set_zero_except(self.direct_overwrite_gradients)

    def step(mut self) raises:
        """Update model parameters."""
        self.iter += 1

        comptime if Self.device.id == Device.cpu.id:
            comptime assert Self.device.id == Device.cpu.id
            var tr = materialize[Self.trainable_parameters]()

            # Loop-invariant across every element of every param — hoisted
            # out of `v_step` so `pow` runs once per `step()` call instead
            # of once per SIMD group per param (was previously the single
            # biggest cost in the whole optimizer step).
            var one_minus_beta1_pow_t = 1 - self.beta1**self.iter
            var one_minus_beta2_pow_t = 1 - self.beta2**self.iter

            # Parallelizing only over parameters leaves one worker processing
            # the large middle-layer weight matrix while the other workers
            # finish its small peers.  Split each parameter into independent
            # contiguous chunks instead; all four buffers are disjoint per
            # chunk, so the updates remain race-free.
            comptime chunk_elements = 32768
            for i in range(len(tr)):
                var param = tr[i]
                var momentum_t = self.momentum_grads[param]
                var rms_t = self.rms_grads[param]
                var grad_t = self.parameters[].grads[param]
                var param_t = self.parameters[].tensors[param]
                var n = param.shape.num_elements()

                def chunk_step(
                    chunk: Int,
                ) {
                    imm self,
                    mut momentum_t,
                    mut rms_t,
                    imm grad_t,
                    mut param_t,
                    imm one_minus_beta1_pow_t,
                    imm one_minus_beta2_pow_t,
                    imm n,
                }:
                    var offset = chunk * chunk_elements

                    def v_step[
                        nelts: Int
                    ](j: Int) {
                        imm self,
                        mut momentum_t,
                        mut rms_t,
                        imm grad_t,
                        mut param_t,
                        imm one_minus_beta1_pow_t,
                        imm one_minus_beta2_pow_t,
                        imm offset,
                    }:
                        var index = offset + j
                        var momentum_grads = momentum_t.load[nelts](index)
                        var rms_grads = rms_t.load[nelts](index)
                        var grads = grad_t.load[nelts](index)
                        var params = param_t.load[nelts](index)

                        # Momentum beta 1
                        # f1 = beta1 * momentum + (1 - beta1) * grad
                        momentum_grads = (
                            self.beta1 * momentum_grads
                            + (1 - self.beta1) * grads
                        )
                        momentum_t.store[nelts](index, momentum_grads)

                        # Bias correction
                        # f2 = f1 / (1 - beta1 ** iter)
                        momentum_grads = momentum_grads / one_minus_beta1_pow_t

                        # RMS beta 2
                        # f1 = beta2 * rms + (1 - beta2) * grad ** 2
                        rms_grads = (
                            self.beta2 * rms_grads
                            + (1 - self.beta2) * grads * grads
                        )
                        rms_t.store[nelts](index, rms_grads)

                        # Bias correction
                        # f2 = f1 / (1 - beta2 ** iter)
                        rms_grads = rms_grads / one_minus_beta2_pow_t

                        # tensor = tensor - lr * (f2 / (sqrt(rms) + epsilon))
                        params = params - self.lr * (
                            momentum_grads / (sqrt(rms_grads) + self.epsilon)
                        )
                        param_t.store[nelts](index, params)

                    vectorize[simd_width_of[f32]()](
                        min(chunk_elements, n - offset), v_step
                    )

                var num_chunks = (n + chunk_elements - 1) // chunk_elements
                if num_chunks == 1:
                    # parallelize() dispatches through the MAX thread pool
                    # even for a single task; that fixed dispatch cost swamps
                    # small parameters (most models), so call directly.
                    chunk_step(0)
                else:
                    parallelize(chunk_step, num_chunks)
        else:
            # Native on-device Adam-update kernel — no host round-trip
            # (each parameter element updates independently, so this is a
            # pure elementwise kernel, same family as the ADD/SUB/MUL/DIV/
            # RELU kernels in gpu_elementwise.mojo).
            var one_minus_beta1_pow_t = 1 - self.beta1**self.iter
            var one_minus_beta2_pow_t = 1 - self.beta2**self.iter
            # Use the graph's runtime symbol list on GPU.  Besides avoiding
            # a large specialization for mixed Conv/Linear graphs, this keeps
            # every update on the device just like the CPU path above.
            var trainable = get_trainable_parameters(Self.g)
            for i in range(len(trainable)):
                var param = trainable[i]
                var momentum_gpu = self.momentum_grads[param]
                var rms_gpu = self.rms_grads[param]
                var params_gpu = self.parameters[].tensors[param]
                var grad_gpu = self.parameters[].grads[param]

                gpu_adam_step(
                    rebind[Tensor[f32, Device.gpu]](params_gpu),
                    rebind[Tensor[f32, Device.gpu]](momentum_gpu),
                    rebind[Tensor[f32, Device.gpu]](rms_gpu),
                    rebind[Tensor[f32, Device.gpu]](grad_gpu),
                    self.lr,
                    self.beta1,
                    self.beta2,
                    self.epsilon,
                    one_minus_beta1_pow_t,
                    one_minus_beta2_pow_t,
                )

    def step(mut self, mut parameters: Parameters[Self.device]) raises:
        """Update parameters through an explicit model storage reference."""
        comptime if Self.device.id == Device.cpu.id:
            # Keep the established CPU implementation as the single source
            # of truth.  Its pointer path is safe on CPU.
            self.step()
        else:
            self.iter += 1
            var one_minus_beta1_pow_t = 1 - self.beta1**self.iter
            var one_minus_beta2_pow_t = 1 - self.beta2**self.iter
            var trainable = get_trainable_parameters(Self.g)
            for i in range(len(trainable)):
                var param = trainable[i]
                var momentum_gpu = self.momentum_grads[param]
                var rms_gpu = self.rms_grads[param]
                var params_gpu = parameters.tensors[param]
                var grad_gpu = parameters.grads[param]
                gpu_adam_step(
                    rebind[Tensor[f32, Device.gpu]](params_gpu),
                    rebind[Tensor[f32, Device.gpu]](momentum_gpu),
                    rebind[Tensor[f32, Device.gpu]](rms_gpu),
                    rebind[Tensor[f32, Device.gpu]](grad_gpu),
                    self.lr,
                    self.beta1,
                    self.beta2,
                    self.epsilon,
                    one_minus_beta1_pow_t,
                    one_minus_beta2_pow_t,
                )

    def allocate_rms_and_momentum(mut self):
        # They are initialized to zero
        # Loop over all trainable parameters
        var tr = materialize[Self.trainable_parameters]()
        for i in range(len(tr)):
            var param = tr[i]
            self.rms_grads.append(Tensor[f32, Self.device](param.shape), param)
            self.momentum_grads.append(
                Tensor[f32, Self.device](param.shape), param
            )


# ===----------------------------------------------------------------------===#
# AdamW
# ===----------------------------------------------------------------------===#


struct AdamW[
    g: Graph,
    trainable_parameters: List[Symbol] = get_trainable_parameters(g),
    device: Device = Device.cpu,
](Optimizer):
    """
    Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

    Unlike plain Adam + L2 regularization (which folds `weight_decay * w`
    into the gradient, so it gets divided by the RMS term), AdamW applies
    decay directly to the parameter: `param -= lr * weight_decay * param`,
    decoupled from the gradient-based update. This is the standard
    optimizer for training Transformers.
    """

    var parameters: Pointer[Parameters[Self.device], MutUntrackedOrigin]

    var lr: Scalar[f32]
    var beta1: Scalar[f32]
    var beta2: Scalar[f32]
    var epsilon: Scalar[f32]
    var weight_decay: Scalar[f32]
    var iter: Int

    var rms_grads: Collection[Self.device]
    var momentum_grads: Collection[Self.device]

    def __init__(
        out self,
        ref[MutAnyOrigin] parameters: Parameters[Self.device],
        lr: Scalar[f32] = 0.001,
        beta1: Scalar[f32] = 0.9,
        beta2: Scalar[f32] = 0.999,
        epsilon: Scalar[f32] = 1e-8,
        weight_decay: Scalar[f32] = 0.01,
    ):
        self.parameters = Pointer(to=parameters).unsafe_origin_cast[
            MutUntrackedOrigin
        ]()

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.iter = 0

        var tr = materialize[Self.trainable_parameters]()
        self.rms_grads = Collection[Self.device](capacity=len(tr))
        self.momentum_grads = Collection[Self.device](capacity=len(tr))

        self.allocate_rms_and_momentum()

    def zero_grad(mut self) raises:
        """Set all gradients to zero."""
        self.parameters[].grads.set_zero()

    def step(mut self) raises:
        """Update model parameters."""
        self.iter += 1
        var tr = materialize[Self.trainable_parameters]()

        var one_minus_beta1_pow_t = 1 - self.beta1**self.iter
        var one_minus_beta2_pow_t = 1 - self.beta2**self.iter

        comptime if Self.device.id == Device.cpu.id:
            comptime assert Self.device.id == Device.cpu.id

            # See Adam.step for why this chunks each parameter into
            # independent pieces instead of parallelizing over whole
            # parameters — same worker-imbalance problem, same fix.
            comptime chunk_elements = 32768
            for i in range(len(tr)):
                var param = tr[i]
                var momentum_t = self.momentum_grads[param]
                var rms_t = self.rms_grads[param]
                var grad_t = self.parameters[].grads[param]
                var param_t = self.parameters[].tensors[param]
                var n = param.shape.num_elements()

                def chunk_step(
                    chunk: Int,
                ) {
                    imm self,
                    mut momentum_t,
                    mut rms_t,
                    imm grad_t,
                    mut param_t,
                    imm one_minus_beta1_pow_t,
                    imm one_minus_beta2_pow_t,
                    imm n,
                }:
                    var offset = chunk * chunk_elements

                    def v_step[
                        nelts: Int
                    ](j: Int) {
                        imm self,
                        mut momentum_t,
                        mut rms_t,
                        imm grad_t,
                        mut param_t,
                        imm one_minus_beta1_pow_t,
                        imm one_minus_beta2_pow_t,
                        imm offset,
                    }:
                        var index = offset + j
                        var momentum_grads = momentum_t.load[nelts](index)
                        var rms_grads = rms_t.load[nelts](index)
                        var grads = grad_t.load[nelts](index)
                        var params = param_t.load[nelts](index)

                        # Momentum beta 1
                        momentum_grads = (
                            self.beta1 * momentum_grads
                            + (1 - self.beta1) * grads
                        )
                        momentum_t.store[nelts](index, momentum_grads)
                        momentum_grads = momentum_grads / one_minus_beta1_pow_t

                        # RMS beta 2
                        rms_grads = (
                            self.beta2 * rms_grads
                            + (1 - self.beta2) * grads * grads
                        )
                        rms_t.store[nelts](index, rms_grads)
                        rms_grads = rms_grads / one_minus_beta2_pow_t

                        # Decoupled weight decay, applied directly to the
                        # param (not folded into the gradient like Adam + L2
                        # would).
                        if self.weight_decay != 0.0:
                            params = (
                                params - self.lr * self.weight_decay * params
                            )

                        params = params - self.lr * (
                            momentum_grads / (sqrt(rms_grads) + self.epsilon)
                        )
                        param_t.store[nelts](index, params)

                    vectorize[simd_width_of[f32]()](
                        min(chunk_elements, n - offset), v_step
                    )

                var num_chunks = (n + chunk_elements - 1) // chunk_elements
                if num_chunks == 1:
                    # See Adam.step: avoid thread-pool dispatch overhead for
                    # the common single-chunk (small parameter) case.
                    chunk_step(0)
                else:
                    parallelize(chunk_step, num_chunks)
        else:
            for i in range(len(tr)):
                var param = tr[i]
                var momentum_gpu = self.momentum_grads[param]
                var rms_gpu = self.rms_grads[param]
                var params_gpu = self.parameters[].tensors[param]
                var grad_gpu = self.parameters[].grads[param]
                gpu_adamw_step(
                    rebind[Tensor[f32, Device.gpu]](params_gpu),
                    rebind[Tensor[f32, Device.gpu]](momentum_gpu),
                    rebind[Tensor[f32, Device.gpu]](rms_gpu),
                    rebind[Tensor[f32, Device.gpu]](grad_gpu),
                    self.lr,
                    self.beta1,
                    self.beta2,
                    self.epsilon,
                    self.weight_decay,
                    one_minus_beta1_pow_t,
                    one_minus_beta2_pow_t,
                )

    def allocate_rms_and_momentum(mut self):
        var tr = materialize[Self.trainable_parameters]()
        for i in range(len(tr)):
            var param = tr[i]
            self.rms_grads.append(Tensor[f32, Self.device](param.shape), param)
            self.momentum_grads.append(
                Tensor[f32, Self.device](param.shape), param
            )


# ===----------------------------------------------------------------------===#
# SGD
# ===----------------------------------------------------------------------===#


struct SGD[
    g: Graph,
    trainable_parameters: List[Symbol] = get_trainable_parameters(g),
](Optimizer):
    """
    Stochastic Gradient Descent optimizer with optional momentum.

    Update rule (momentum=0):
        param = param - lr * grad

    Update rule (momentum > 0):
        velocity = momentum * velocity - lr * grad
        param    = param + velocity
    """

    var parameters: Pointer[Parameters[Device.cpu], MutUntrackedOrigin]

    var lr: Scalar[f32]
    var momentum: Scalar[f32]
    var weight_decay: Scalar[f32]

    var velocities: Collection[Device.cpu]

    def __init__(
        out self,
        ref[MutAnyOrigin] parameters: Parameters[Device.cpu],
        lr: Scalar[f32] = 0.01,
        momentum: Scalar[f32] = 0.0,
        weight_decay: Scalar[f32] = 0.0,
    ):
        self.parameters = Pointer(to=parameters).unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        var tr = materialize[Self.trainable_parameters]()
        self.velocities = Collection[Device.cpu](capacity=len(tr))
        self.allocate_velocities()

    def zero_grad(mut self) raises:
        """Set all gradients to zero."""
        self.parameters[].grads.set_zero()

    def step(mut self):
        """Update model parameters."""
        var tr = materialize[Self.trainable_parameters]()

        def p_step(i: Int) {mut self, imm tr}:
            var param = tr[i]

            # See Adam.step for why these are hoisted out of `v_step`.
            var grad_t = self.parameters[].grads[param]
            var param_t = self.parameters[].tensors[param]
            var vel_t = self.velocities[param]

            def v_step[
                nelts: Int
            ](j: Int) {imm self, imm grad_t, mut param_t, mut vel_t}:
                var grad = grad_t.load[nelts](j)
                var w = param_t.load[nelts](j)

                # Optional weight decay (L2 regularization)
                if self.weight_decay != 0.0:
                    grad = grad + self.weight_decay * w

                if self.momentum != 0.0:
                    var vel = vel_t.load[nelts](j)
                    vel = self.momentum * vel - self.lr * grad
                    vel_t.store[nelts](j, vel)
                    param_t.store[nelts](j, w + vel)
                else:
                    param_t.store[nelts](j, w - self.lr * grad)

            vectorize[simd_width_of[f32]()](param.shape.num_elements(), v_step)

        parallelize(p_step, len(tr))

    def allocate_velocities(mut self):
        var tr = materialize[Self.trainable_parameters]()
        for i in range(len(tr)):
            var param = tr[i]
            self.velocities.append(Tensor[f32](param.shape), param)


# ===----------------------------------------------------------------------===#
# LR Schedulers
# ===----------------------------------------------------------------------===#


struct WarmupCosineSchedule(Copyable, Movable):
    """
    Linear warmup for `warmup_steps`, then cosine decay from `base_lr` down
    to `min_lr` over the remaining `total_steps - warmup_steps`. Standard
    schedule for training Transformers.

    Usage:
        var sched = WarmupCosineSchedule(base_lr=3e-4, warmup_steps=100, total_steps=3000)
        for step in range(num_steps):
            optim.lr = sched.get_lr(step)
            ...
    """

    var base_lr: Scalar[f32]
    var min_lr: Scalar[f32]
    var warmup_steps: Int
    var total_steps: Int

    def __init__(
        out self,
        base_lr: Scalar[f32],
        warmup_steps: Int,
        total_steps: Int,
        min_lr: Scalar[f32] = 0.0,
    ):
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_lr(self, step: Int) -> Scalar[f32]:
        """Learning rate for `step` (0-indexed)."""
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return (
                self.base_lr
                * Scalar[f32](step + 1)
                / Scalar[f32](self.warmup_steps)
            )

        var decay_steps = self.total_steps - self.warmup_steps
        if decay_steps <= 0:
            return self.base_lr

        var progress = Scalar[f32](step - self.warmup_steps) / Scalar[f32](
            decay_steps
        )
        if progress > 1.0:
            progress = 1.0

        var cosine_factor = Scalar[f32](
            0.5 * (1.0 + cos(PI * Float64(progress)))
        )
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_factor


struct StepDecaySchedule(Copyable, Movable):
    """
    Multiplies `base_lr` by `gamma` every `step_size` steps. Standard
    schedule for vision models (e.g. ResNet).
    """

    var base_lr: Scalar[f32]
    var step_size: Int
    var gamma: Scalar[f32]

    def __init__(
        out self, base_lr: Scalar[f32], step_size: Int, gamma: Scalar[f32] = 0.1
    ):
        self.base_lr = base_lr
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, step: Int) -> Scalar[f32]:
        """Learning rate for `step` (0-indexed)."""
        var num_decays = step // self.step_size
        var lr = self.base_lr
        for _ in range(num_decays):
            lr *= self.gamma
        return lr

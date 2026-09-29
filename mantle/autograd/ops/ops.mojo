# ===----------------------------------------------------------------------=== #
# Mantle: Ops Registry
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Ops (mantle.autograd.ops.ops)
------------------------------------------------
OP enum and dispatcher for shape inference, forward, and backward passes.
"""
from std.algorithm import vectorize
from max.algorithm import parallelize
from std.sys.info import simd_width_of
from std.utils.index import IndexList

from .basics import (
    ADD,
    SUB,
    MUL,
    DIV,
    EXP,
    LOG,
    POW,
    DOT,
    dot_batch_broadcast_shape,
    SUM,
    MEAN,
    MAX,
    MIN,
    ARGMAX,
    FLATTEN,
    RESHAPE,
    TRANSPOSE,
    FMA,
)
from .mlops import (
    SIGMOID,
    RELU,
    LEAKYRELU,
    TANH,
    GELU,
    NEG,
    ABS,
    SQRT,
    CLIP,
    SQUEEZE,
    UNSQUEEZE,
    SLICE,
    DROPOUT,
    BATCHNORM2D,
    GATHER,
    PAD,
)
from .dynamics import CONCAT, SPLIT
from .conv import CONV2D
from .conv_transpose import CONVTRANSPOSE2D
from .pool import MAXPOOL2D, AVGPOOL2D
from .matmul import dot_transpose_t1, dot_transpose_t2

from std.os import abort
from std.math import sqrt

from mantle import f32
from mantle.autograd.symbol import Symbol
from mantle.core.tensor import Tensor, TensorShape, MAX_RANK
from mantle.core.device import Device
from mantle.nn.parameters import Parameters
from mantle.core.bytes import Bytes
from mantle.core.tensorutils import broadcast_shapes, accumulate_grad
from mantle.autograd.attributes import AttributeVector
from .gpu_elementwise import (
    gpu_add_forward,
    gpu_add_bias_forward,
    gpu_bias_grad,
    gpu_bias_grad_into,
    gpu_sub_forward,
    gpu_mul_forward,
    gpu_div_forward,
    gpu_broadcast_binary_forward,
    gpu_broadcast_binary_backward_t1,
    gpu_broadcast_binary_backward_t2,
    gpu_relu_forward,
    gpu_relu_backward,
    gpu_relu_backward_into,
    gpu_sub_backward_t2,
    gpu_sub_backward_t2_into,
    gpu_mul_backward,
    gpu_mul_backward_into,
    gpu_div_backward_t1,
    gpu_div_backward_t1_into,
    gpu_div_backward_t2,
    gpu_div_backward_t2_into,
    gpu_pow_forward,
    gpu_pow_backward,
    gpu_pow_backward_into,
    gpu_exp_forward,
    gpu_exp_backward,
    gpu_exp_backward_into,
    gpu_log_forward,
    gpu_log_backward,
    gpu_log_backward_into,
    gpu_sqrt_forward,
    gpu_sqrt_backward,
    gpu_sqrt_backward_into,
    gpu_mean_forward,
    gpu_mean_backward,
    gpu_mean_backward_into,
    gpu_sum_forward,
    gpu_sum_backward,
    gpu_sum_backward_into,
    gpu_reduce_last_forward,
    gpu_reduce_last_backward,
    gpu_reduce_last_backward_into,
    gpu_gather_forward,
    gpu_gather_backward,
    gpu_gelu_forward,
    gpu_gelu_backward,
    gpu_gelu_backward_into,
    gpu_dropout_forward,
    gpu_dropout_backward,
    gpu_dropout_backward_into,
    gpu_accumulate_grad,
    gpu_write_from_host,
    gpu_layernorm_forward,
    gpu_layernorm_input_backward,
    gpu_layernorm_affine_backward,
    gpu_channel_bias_add_forward,
)
from .gpu_matmul import (
    gpu_matmul,
    gpu_matmul_bt,
    gpu_matmul_at,
    gpu_matmul_bias,
    gpu_batched_matmul,
    gpu_batched_matmul_bt,
    gpu_batched_matmul_at,
    gpu_transpose_4d,
)
from .gpu_conv import (
    gpu_conv2d_forward,
    gpu_conv2d_input_backward,
    gpu_conv2d_kernel_backward,
    gpu_conv2d_bias_backward,
    gpu_conv2d_parameter_backward_direct,
)
from .gpu_pool import gpu_maxpool2d_forward, gpu_maxpool2d_backward


# Define operators as named parameter expression
struct OP(TrivialRegisterPassable, Writable):
    """
    Compile time Operators list.
    """

    comptime ADD = OP(0, "ADD")
    comptime SUB = OP(1, "SUB")
    comptime MUL = OP(2, "MUL")
    comptime DIV = OP(3, "DIV")
    comptime EXP = OP(4, "EXP")
    comptime LOG = OP(5, "LOG")
    comptime POW = OP(6, "POW")
    comptime DOT = OP(7, "DOT")
    comptime SUM = OP(8, "SUM")
    comptime MEAN = OP(9, "MEAN")
    comptime MAX = OP(10, "MAX")
    comptime FLATTEN = OP(11, "FLATTEN")
    comptime RESHAPE = OP(12, "RESHAPE")
    comptime SIGMOID = OP(13, "SIGMOID")
    comptime RELU = OP(14, "RELU")
    comptime TANH = OP(15, "TANH")
    comptime CONV2D = OP(16, "CONV2D")
    comptime TRANSPOSE = OP(17, "TRANSPOSE")
    comptime MAXPOOL2D = OP(18, "MAXPOOL2D")
    comptime FMA = OP(19, "FMA")
    comptime CLIP = OP(20, "CLIP")
    comptime SQUEEZE = OP(21, "SQUEEZE")
    comptime UNSQUEEZE = OP(22, "UNSQUEEZE")
    comptime CONCAT = OP(23, "CONCAT", dynamic=True)
    comptime SPLIT = OP(24, "SPLIT", dynamic=True)
    comptime SLICE = OP(25, "SLICE")
    comptime DROPOUT = OP(26, "DROPOUT")
    comptime LEAKYRELU = OP(28, "LEAKYRELU")
    comptime BATCHNORM2D = OP(29, "BATCHNORM2D")
    comptime NEG = OP(30, "NEG")
    comptime ABS = OP(31, "ABS")
    comptime SQRT = OP(32, "SQRT")
    comptime GATHER = OP(33, "GATHER")
    comptime GELU = OP(34, "GELU")
    comptime PAD = OP(35, "PAD")
    comptime AVGPOOL2D = OP(36, "AVGPOOL2D")
    comptime LINEAR = OP(37, "LINEAR")
    comptime MIN = OP(38, "MIN")
    comptime ARGMAX = OP(39, "ARGMAX")
    comptime LAYERNORM = OP(40, "LAYERNORM")
    comptime CONVTRANSPOSE2D = OP(41, "CONVTRANSPOSE2D")

    var id: UInt8
    var name: Bytes[16]
    var dynamic: Bool

    def __init__(out self, id: UInt8, name: String, dynamic: Bool = False):
        self.id = id
        self.name = Bytes[16](name)
        self.dynamic = dynamic

    def __eq__(self, other: OP) -> Bool:
        return self.id == other.id

    def write_to[W: Writer](self, mut writer: W):
        writer.write(String(self.name))

    def __str__(self) -> String:
        return String(self)


@always_inline
def inverse_transpose_axes(axes: TensorShape) -> TensorShape:
    """Return the inverse permutation used by TRANSPOSE backward."""
    var inverse = IndexList[MAX_RANK]()
    for i in range(axes.rank()):
        inverse[axes[i]] = i
    return TensorShape(rank=axes.rank(), shape=inverse)


def cpu_add_bias_forward[
    outer: Int, n: Int
](mut res: Tensor[f32], a: Tensor[f32], bias: Tensor[f32]):
    """Res[i,j] = a[i,j] + bias[j] — the CPU counterpart of
    `gpu_add_bias_forward`: `elwise_op`'s generic broadcast path
    (`broadcast_elwise_op`) is `vectorize[1]` (scalar; broadcast indexing
    generally doesn't vectorize), so `LINEAR`'s bias-add gets its own
    fast path here, vectorizing per row where the bias indices are
    contiguous."""
    comptime nelts = simd_width_of[f32]()

    def row(i: Int) {mut res, imm a, imm bias}:
        def vec[w: Int](j: Int) {mut res, imm a, imm bias, imm i}:
            res.store[w](i * n + j, a.load[w](i * n + j) + bias.load[w](j))

        vectorize[nelts](n, vec)

    parallelize(row, outer)


def cpu_bias_grad_accumulate[
    outer: Int, n: Int
](mut grad: Tensor[f32], ug: Tensor[f32]):
    """Grad[j] += sum_i ug[i,j] — the CPU counterpart of `gpu_bias_grad`
    (see `cpu_add_bias_forward` for why this needs its own fast path
    instead of the generic broadcast `accumulate_grad`)."""
    comptime nelts = simd_width_of[f32]()

    def vec[w: Int](j: Int) {mut grad, imm ug}:
        var acc = grad.load[w](j)
        for i in range(outer):
            acc += ug.load[w](i * n + j)
        grad.store[w](j, acc)

    vectorize[nelts](n, vec)


def cpu_bias_grad_overwrite[
    outer: Int, n: Int
](mut grad: Tensor[f32], ug: Tensor[f32]):
    """Grad[j] = sum_i ug[i,j] — the `overwrite_grad` counterpart of
    `cpu_bias_grad_accumulate`, for when `grad` is a single-consumer buffer
    being written directly rather than accumulated into."""
    comptime nelts = simd_width_of[f32]()

    def vec[w: Int](j: Int) {mut grad, imm ug}:
        var acc: SIMD[f32, w] = 0
        for i in range(outer):
            acc += ug.load[w](i * n + j)
        grad.store[w](j, acc)

    vectorize[nelts](n, vec)


def cpu_layernorm_forward[
    outer: Int, width: Int
](
    mut res: Tensor[f32],
    src: Tensor[f32],
    gamma: Tensor[f32],
    beta: Tensor[f32],
    epsilon: Scalar[f32],
):
    def row(i: Int) {mut res, imm src, imm gamma, imm beta, imm epsilon}:
        var mean: Scalar[f32] = 0.0
        for j in range(width):
            mean += src.load[1](i * width + j)
        mean /= Scalar[f32](width)
        var variance: Scalar[f32] = 0.0
        for j in range(width):
            var delta = src.load[1](i * width + j) - mean
            variance += delta * delta
        var inv_std = 1.0 / sqrt(variance / Scalar[f32](width) + epsilon)
        for j in range(width):
            var xhat = (src.load[1](i * width + j) - mean) * inv_std
            res.store[1](
                i * width + j, xhat * gamma.load[1](j) + beta.load[1](j)
            )

    parallelize(row, outer)


def cpu_layernorm_input_backward[
    outer: Int, width: Int
](
    ug: Tensor[f32],
    src: Tensor[f32],
    gamma: Tensor[f32],
    mut grad: Tensor[f32],
    epsilon: Scalar[f32],
):
    def row(i: Int) {mut grad, imm ug, imm src, imm gamma, imm epsilon}:
        var mean: Scalar[f32] = 0.0
        for j in range(width):
            mean += src.load[1](i * width + j)
        mean /= Scalar[f32](width)
        var variance: Scalar[f32] = 0.0
        for j in range(width):
            var delta = src.load[1](i * width + j) - mean
            variance += delta * delta
        var inv_std = 1.0 / sqrt(variance / Scalar[f32](width) + epsilon)
        var sum_dy: Scalar[f32] = 0.0
        var sum_dy_xhat: Scalar[f32] = 0.0
        for j in range(width):
            var dy = ug.load[1](i * width + j) * gamma.load[1](j)
            var xhat = (src.load[1](i * width + j) - mean) * inv_std
            sum_dy += dy
            sum_dy_xhat += dy * xhat
        for j in range(width):
            var dy = ug.load[1](i * width + j) * gamma.load[1](j)
            var xhat = (src.load[1](i * width + j) - mean) * inv_std
            var index = i * width + j
            grad.store[1](
                index,
                grad.load[1](index)
                + inv_std
                * (
                    dy
                    - sum_dy / Scalar[f32](width)
                    - xhat * sum_dy_xhat / Scalar[f32](width)
                ),
            )

    parallelize(row, outer)


def cpu_layernorm_affine_backward[
    outer: Int, width: Int, affine_id: Int
](
    ug: Tensor[f32],
    src: Tensor[f32],
    mut grad: Tensor[f32],
    epsilon: Scalar[f32],
):
    def feature(j: Int) {mut grad, imm ug, imm src, imm epsilon}:
        var total: Scalar[f32] = 0.0
        for i in range(outer):
            var mean: Scalar[f32] = 0.0
            for k in range(width):
                mean += src.load[1](i * width + k)
            mean /= Scalar[f32](width)
            var variance: Scalar[f32] = 0.0
            for k in range(width):
                var delta = src.load[1](i * width + k) - mean
                variance += delta * delta
            var upper = ug.load[1](i * width + j)
            comptime if affine_id == 0:
                total += (
                    upper
                    * (src.load[1](i * width + j) - mean)
                    / sqrt(variance / Scalar[f32](width) + epsilon)
                )
            else:
                total += upper
        grad.store[1](j, grad.load[1](j) + total)

    parallelize(feature, width)


def static_result_shape(
    op: OP, operands: VariadicList[Symbol, _], attributes: AttributeVector
) -> TensorShape:
    """
    Static result shape for operators.
    """
    if len(operands) == 1:
        return static_result_shape(op, operands[0].shape, attributes)
    elif len(operands) == 2:
        return static_result_shape(
            op, operands[0].shape, operands[1].shape, attributes
        )
    elif len(operands) == 3:
        return static_result_shape(
            op,
            operands[0].shape,
            operands[1].shape,
            operands[2].shape,
            attributes,
        )
    else:
        print("Error: Invalid number of operands")
        return TensorShape()


def static_result_shape(
    op: OP, t1_shape: TensorShape, attributes: AttributeVector
) -> TensorShape:
    """
    Static result shape for unary operators.
    """
    if op == OP.EXP:
        return EXP.result_shape(t1_shape)
    elif op == OP.LOG:
        return LOG.result_shape(t1_shape)
    elif op == OP.SUM:
        return SUM.result_shape(t1_shape, attributes)
    elif op == OP.MEAN:
        return MEAN.result_shape(t1_shape, attributes)
    elif op == OP.MAX:
        return MAX.result_shape(t1_shape, attributes)
    elif op == OP.MIN:
        return MIN.result_shape(t1_shape, attributes)
    elif op == OP.ARGMAX:
        return ARGMAX.result_shape(t1_shape, attributes)
    elif op == OP.FLATTEN:
        return FLATTEN.result_shape(t1_shape)
    elif op == OP.RESHAPE:
        return RESHAPE.result_shape(t1_shape, attributes)
    elif op == OP.SIGMOID:
        return SIGMOID.result_shape(t1_shape)
    elif op == OP.RELU:
        return RELU.result_shape(t1_shape)
    elif op == OP.LEAKYRELU:
        return LEAKYRELU.result_shape(t1_shape)
    elif op == OP.TANH:
        return TANH.result_shape(t1_shape)
    elif op == OP.GELU:
        return GELU.result_shape(t1_shape)
    elif op == OP.TRANSPOSE:
        return TRANSPOSE.result_shape(t1_shape, attributes)
    elif op == OP.MAXPOOL2D:
        return MAXPOOL2D.result_shape(t1_shape, attributes)
    elif op == OP.AVGPOOL2D:
        return AVGPOOL2D.result_shape(t1_shape, attributes)
    elif op == OP.CLIP:
        return CLIP.result_shape(t1_shape)
    elif op == OP.SQUEEZE:
        return SQUEEZE.result_shape(t1_shape, attributes)
    elif op == OP.UNSQUEEZE:
        return UNSQUEEZE.result_shape(t1_shape, attributes)
    elif op == OP.SLICE:
        return SLICE.result_shape(t1_shape, attributes)
    elif op == OP.PAD:
        return PAD.result_shape(t1_shape, attributes)
    elif op == OP.DROPOUT:
        return DROPOUT.result_shape(t1_shape)
    elif op == OP.NEG:
        return NEG.result_shape(t1_shape)
    elif op == OP.ABS:
        return ABS.result_shape(t1_shape)
    elif op == OP.SQRT:
        return SQRT.result_shape(t1_shape)
    else:
        print("[ERROR] Operator not found.")
    return TensorShape(-1)


def gpu_binary_op_code(op: OP) -> Int:
    if op == OP.ADD:
        return 0
    elif op == OP.SUB:
        return 1
    elif op == OP.MUL:
        return 2
    else:
        return 3


def gpu_reduce_op_code(op: OP) -> Int:
    if op == OP.SUM:
        return 0
    elif op == OP.MEAN:
        return 1
    else:
        return 2


def matching_batch_dims(t1_shape: TensorShape, t2_shape: TensorShape) -> Bool:
    if t1_shape.rank() != t2_shape.rank():
        return False
    for i in range(t1_shape.rank() - 2):
        if t1_shape[i] != t2_shape[i]:
            return False
    return True


def leading_broadcast_compatible(
    output_shape: TensorShape, input_shape: TensorShape
) -> Bool:
    if input_shape.rank() >= output_shape.rank():
        return False
    for i in range(input_shape.rank()):
        if (
            input_shape[i]
            != output_shape[output_shape.rank() - input_shape.rank() + i]
        ):
            return False
    return True


def static_result_shape(
    op: OP,
    t1_shape: TensorShape,
    t2_shape: TensorShape,
    attributes: AttributeVector,
) -> TensorShape:
    """
    Static result shape for binary operators.
    """
    if op == OP.ADD:
        return ADD.result_shape(t1_shape, t2_shape)
    elif op == OP.SUB:
        return SUB.result_shape(t1_shape, t2_shape)
    elif op == OP.MUL:
        return MUL.result_shape(t1_shape, t2_shape)
    elif op == OP.DIV:
        return DIV.result_shape(t1_shape, t2_shape)
    elif op == OP.POW:
        return POW.result_shape(t1_shape, t2_shape)
    elif op == OP.DOT:
        return DOT.result_shape(t1_shape, t2_shape)
    elif op == OP.GATHER:
        return GATHER.result_shape(t1_shape, t2_shape)
    else:
        # We can't print at compile time (at least for now it crashes at comp time with an error)
        print("[ERROR] Operator not found.")
        return TensorShape(-1, -1)


def static_result_shape(
    op: OP,
    t1_shape: TensorShape,
    t2_shape: TensorShape,
    t3_shape: TensorShape,
    attributes: AttributeVector,
) -> TensorShape:
    """
    Static result shape for ternary operators.
    """

    if op == OP.CONV2D:
        return CONV2D.result_shape(t1_shape, t2_shape, t3_shape, attributes)
    elif op == OP.CONVTRANSPOSE2D:
        return CONVTRANSPOSE2D.result_shape(
            t1_shape, t2_shape, t3_shape, attributes
        )
    elif op == OP.FMA:
        return FMA.result_shape(t1_shape, t2_shape, t3_shape)
    elif op == OP.BATCHNORM2D:
        return BATCHNORM2D.result_shape(
            t1_shape, t2_shape, t3_shape, attributes
        )
    elif op == OP.LINEAR:
        return DOT.result_shape(t1_shape, t2_shape)
    elif op == OP.LAYERNORM:
        return t1_shape
    else:
        print("[ERROR] Operator not found.")
        return TensorShape(-1, -1)


def dynamic_result_shape(
    op: OP,
    operands: VariadicList[Symbol, _],
    attributes: AttributeVector,
) -> List[TensorShape]:
    """
    Static result shape for dynamic operators.
    """
    # Unknown number of inputs and outputs.
    var input_shapes = List[TensorShape]()
    for operand in operands:
        input_shapes.append(operand.shape)

    if op == OP.CONCAT:
        return CONCAT.result_shape(input_shapes, attributes)
    elif op == OP.SPLIT:
        return SPLIT.result_shape(input_shapes, attributes)
    else:
        print("[ERROR] Operator not found.")
        return [TensorShape(-1)]


def forward_op[
    op: OP,
    t1_shape: TensorShape,
    attributes: AttributeVector,
    device: Device = Device.cpu,
](
    mut res: Tensor[f32, device],
    t1: Tensor[f32, device],
    runtime_seed: UInt64 = 0,
    training: Bool = True,
) raises:
    """
    Forward pass for unary operators, dispatching by device.
    """
    comptime if device.id == Device.cpu.id:
        comptime assert device.id == Device.cpu.id
        _forward_op_cpu[op, t1_shape, attributes](
            rebind[Tensor[f32, Device.cpu]](res),
            rebind[Tensor[f32, Device.cpu]](t1),
            runtime_seed,
            training,
        )
    elif op == OP.RELU:
        gpu_relu_forward(
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
        )
    elif op == OP.EXP:
        gpu_exp_forward(
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
        )
    elif op == OP.LOG:
        gpu_log_forward(
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
        )
    elif op == OP.GELU:
        gpu_gelu_forward(
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
        )
    elif op == OP.DROPOUT:
        comptime p = attributes["p"].value().to_scalar[f32]()
        comptime base_seed = UInt64(
            attributes["seed"].value().to_scalar[DType.int64]()
        )
        gpu_dropout_forward(
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
            p,
            base_seed ^ runtime_seed,
            training,
        )
    elif op == OP.MEAN and not attributes["axis"]:
        gpu_mean_forward(
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
        )
    elif op == OP.SUM and not attributes["axis"]:
        gpu_sum_forward(
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
        )
    elif (
        (op == OP.SUM or op == OP.MEAN or op == OP.MAX)
        and attributes["axis"]
        and attributes["axis"].value().to_int() == t1_shape.rank() - 1
    ):
        gpu_reduce_last_forward(
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
            t1_shape[-1],
            gpu_reduce_op_code(op),
        )
    elif op == OP.SQRT:
        gpu_sqrt_forward(
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
        )
    elif op == OP.MAXPOOL2D:
        comptime kernel_size = attributes["kernel_size"].value().to_static[2]()
        comptime padding = attributes["padding"].value().to_static[2]()
        comptime stride = attributes["stride"].value().to_static[2]()
        comptime dilation = attributes["dilation"].value().to_static[2]()
        comptime out_shape = MAXPOOL2D.result_shape(t1_shape, attributes)
        gpu_maxpool2d_forward[
            t1_shape[0],
            t1_shape[1],
            t1_shape[2],
            t1_shape[3],
            out_shape[2],
            out_shape[3],
            kernel_size[0],
            kernel_size[1],
            padding[0],
            padding[1],
            stride[0],
            stride[1],
            dilation[0],
            dilation[1],
        ](
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
        )
    elif op == OP.RESHAPE or op == OP.FLATTEN:
        # These operators only change metadata. Graph tensors own distinct
        # buffers today, so preserve that representation with a queued
        # device-to-device copy instead of falling back through host memory.
        rebind[Tensor[f32, Device.gpu]](res).copy_from(
            rebind[Tensor[f32, Device.gpu]](t1)
        )
    elif op == OP.TRANSPOSE and t1_shape.rank() == 4:
        # Multi-head attention uses only rank-4 layout permutations. Keep
        # them device-resident instead of sending Q/K/V through the CPU.
        comptime axes_attribute = attributes["axes"]
        comptime if axes_attribute:
            comptime axes = axes_attribute.value().to_shape()
            gpu_transpose_4d(
                rebind[Tensor[f32, Device.gpu]](res),
                rebind[Tensor[f32, Device.gpu]](t1),
                axes,
            )
        else:
            gpu_transpose_4d(
                rebind[Tensor[f32, Device.gpu]](res),
                rebind[Tensor[f32, Device.gpu]](t1),
                TensorShape(3, 2, 1, 0),
            )
    else:
        # Host round-trip fallback for unary ops without a native GPU
        # kernel yet (e.g. SIGMOID, TANH, axis-reduced MEAN, ...): run the
        # existing, unmodified CPU implementation against host copies, then
        # write the result back into `res`'s existing GPU buffer.
        var res_cpu = res.to_host()
        var t1_cpu = t1.to_host()
        _forward_op_cpu[op, t1_shape, attributes](
            res_cpu, t1_cpu, runtime_seed, training
        )
        gpu_write_from_host(rebind[Tensor[f32, Device.gpu]](res), res_cpu)


def _forward_op_cpu[
    op: OP, t1_shape: TensorShape, attributes: AttributeVector
](
    mut res: Tensor[f32],
    t1: Tensor[f32],
    runtime_seed: UInt64 = 0,
    training: Bool = True,
):
    """
    Forward pass for unary operators (CPU implementations).
    """

    comptime if op == OP.EXP:
        EXP.forward[t1_shape](res, t1)
    elif op == OP.LOG:
        LOG.forward[t1_shape](res, t1)
    elif op == OP.SUM:
        SUM.forward[t1_shape, attributes](res, t1)
    elif op == OP.MEAN:
        MEAN.forward[t1_shape, attributes](res, t1)
    elif op == OP.MAX:
        MAX.forward[t1_shape, attributes](res, t1)
    elif op == OP.MIN:
        MIN.forward[t1_shape, attributes](res, t1)
    elif op == OP.ARGMAX:
        ARGMAX.forward[t1_shape, attributes](res, t1)
    elif op == OP.FLATTEN:
        FLATTEN.forward[t1_shape](res, t1)
    elif op == OP.RESHAPE:
        RESHAPE.forward[t1_shape](res, t1)
    elif op == OP.SIGMOID:
        SIGMOID.forward[t1_shape](res, t1)
    elif op == OP.RELU:
        RELU.forward[t1_shape](res, t1)
    elif op == OP.LEAKYRELU:
        LEAKYRELU.forward[t1_shape, attributes](res, t1)
    elif op == OP.TANH:
        TANH.forward[t1_shape](res, t1)
    elif op == OP.GELU:
        GELU.forward[t1_shape](res, t1)
    elif op == OP.TRANSPOSE:
        TRANSPOSE.forward[t1_shape, attributes](res, t1)
    elif op == OP.MAXPOOL2D:
        MAXPOOL2D.forward[t1_shape, attributes](res, t1)
    elif op == OP.CLIP:
        CLIP.forward[t1_shape, attributes](res, t1)
    elif op == OP.SQUEEZE:
        SQUEEZE.forward[t1_shape, attributes](res, t1)
    elif op == OP.UNSQUEEZE:
        UNSQUEEZE.forward[t1_shape, attributes](res, t1)
    elif op == OP.SLICE:
        SLICE.forward[t1_shape, attributes](res, t1)
    elif op == OP.PAD:
        PAD.forward[t1_shape, attributes](res, t1)
    elif op == OP.DROPOUT:
        DROPOUT.forward[t1_shape, attributes](res, t1, runtime_seed, training)
    elif op == OP.NEG:
        NEG.forward[t1_shape](res, t1)
    elif op == OP.ABS:
        ABS.forward[t1_shape](res, t1)
    elif op == OP.SQRT:
        SQRT.forward[t1_shape](res, t1)
    else:
        print("[ERROR] Operator not found.")


def forward_op[
    op: OP,
    t1_shape: TensorShape,
    t2_shape: TensorShape,
    attributes: AttributeVector,
    device: Device = Device.cpu,
](
    mut res: Tensor[f32, device],
    t1: Tensor[f32, device],
    t2: Tensor[f32, device],
) raises:
    """
    Forward pass for binary operators, dispatching by device.
    """
    comptime if device.id == Device.cpu.id:
        comptime assert device.id == Device.cpu.id
        _forward_op_cpu[op, t1_shape, t2_shape, attributes](
            rebind[Tensor[f32, Device.cpu]](res),
            rebind[Tensor[f32, Device.cpu]](t1),
            rebind[Tensor[f32, Device.cpu]](t2),
        )
    elif t1_shape == t2_shape and op == OP.ADD:
        gpu_add_forward(
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
            rebind[Tensor[f32, Device.gpu]](t2),
        )
    elif t1_shape == t2_shape and op == OP.SUB:
        gpu_sub_forward(
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
            rebind[Tensor[f32, Device.gpu]](t2),
        )
    elif t1_shape == t2_shape and op == OP.MUL:
        gpu_mul_forward(
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
            rebind[Tensor[f32, Device.gpu]](t2),
        )
    elif t1_shape == t2_shape and op == OP.DIV:
        gpu_div_forward(
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
            rebind[Tensor[f32, Device.gpu]](t2),
        )
    elif (
        op == OP.ADD
        and t1_shape.rank() >= 2
        and t2_shape.rank() == 1
        and t2_shape[0] == t1_shape[-1]
    ):
        gpu_add_bias_forward(
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
            rebind[Tensor[f32, Device.gpu]](t2),
        )
    elif (
        (op == OP.ADD or op == OP.SUB or op == OP.MUL or op == OP.DIV)
        and t1_shape.num_elements() > t2_shape.num_elements()
        and (
            t2_shape.num_elements() == 1
            or (t2_shape.rank() == 1 and t2_shape[0] == t1_shape[-1])
            or (
                t2_shape.rank() == t1_shape.rank()
                and t2_shape[-1] == 1
                and t2_shape.num_elements() * t1_shape[-1]
                == t1_shape.num_elements()
            )
            or leading_broadcast_compatible(t1_shape, t2_shape)
        )
    ):
        comptime vector_mode = (
            t2_shape.rank() == 1 and t2_shape[0] == t1_shape[-1]
        )
        comptime leading_broadcast = leading_broadcast_compatible(
            t1_shape, t2_shape
        )
        comptime repeat = 1 if vector_mode or leading_broadcast else (
            t1_shape.num_elements() // t2_shape.num_elements()
        )
        gpu_broadcast_binary_forward(
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
            rebind[Tensor[f32, Device.gpu]](t2),
            repeat,
            0 if vector_mode else (2 if leading_broadcast else 1),
            gpu_binary_op_code(op),
        )
    elif t1_shape.rank() == 2 and t2_shape.rank() == 2 and op == OP.DOT:
        gpu_matmul[t1_shape[0], t1_shape[1], t2_shape[1]](
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
            rebind[Tensor[f32, Device.gpu]](t2),
        )
    elif (
        op == OP.DOT
        and t1_shape.rank() >= 3
        and matching_batch_dims(t1_shape, t2_shape)
    ):
        comptime batches = t1_shape.num_elements() // (
            t1_shape[-2] * t1_shape[-1]
        )
        gpu_batched_matmul[batches, t1_shape[-2], t1_shape[-1], t2_shape[-1]](
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
            rebind[Tensor[f32, Device.gpu]](t2),
        )
    elif op == OP.GATHER and t1_shape.rank() == 2:
        gpu_gather_forward(
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
            rebind[Tensor[f32, Device.gpu]](t2),
            t1_shape[-1],
        )
    elif op == OP.POW:
        gpu_pow_forward(
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
            rebind[Tensor[f32, Device.gpu]](t2),
        )
    else:
        # Host round-trip fallback for GPU ops without a native kernel yet.
        # Runs the existing, unmodified CPU implementation
        # against host copies, then writes the result back into `res`'s
        # existing GPU buffer.
        var res_cpu = res.to_host()
        var t1_cpu = t1.to_host()
        var t2_cpu = t2.to_host()
        _forward_op_cpu[op, t1_shape, t2_shape, attributes](
            res_cpu, t1_cpu, t2_cpu
        )
        gpu_write_from_host(rebind[Tensor[f32, Device.gpu]](res), res_cpu)


def _forward_op_cpu[
    op: OP,
    t1_shape: TensorShape,
    t2_shape: TensorShape,
    attributes: AttributeVector,
](mut res: Tensor[f32], t1: Tensor[f32], t2: Tensor[f32]) raises:
    """
    Forward pass for binary operators (CPU implementations).
    """

    comptime if op == OP.ADD:
        ADD.forward[t1_shape, t2_shape](res, t1, t2)
    elif op == OP.SUB:
        SUB.forward[t1_shape, t2_shape](res, t1, t2)
    elif op == OP.MUL:
        MUL.forward[t1_shape, t2_shape](res, t1, t2)
    elif op == OP.DIV:
        DIV.forward[t1_shape, t2_shape](res, t1, t2)
    elif op == OP.POW:
        POW.forward[t1_shape, t2_shape](res, t1, t2)
    elif op == OP.DOT:
        DOT.forward[t1_shape, t2_shape](res, t1, t2)
    elif op == OP.GATHER:
        GATHER.forward[t1_shape, t2_shape](res, t1, t2)
    else:
        print("[ERROR] Operator not found.")


def forward_op[
    op: OP,
    t1_shape: TensorShape,
    t2_shape: TensorShape,
    t3_shape: TensorShape,
    attributes: AttributeVector,
    device: Device = Device.cpu,
](
    mut res: Tensor[f32, device],
    t1: Tensor[f32, device],
    t2: Tensor[f32, device],
    t3: Tensor[f32, device],
) raises:
    """
    Forward pass for ternary operators, dispatching by device. LINEAR,
    LAYERNORM, CONV2D, and CONVTRANSPOSE2D have GPU kernels; FMA and
    BATCHNORM2D are still CPU-only.
    """
    comptime if device.id == Device.cpu.id:
        comptime assert device.id == Device.cpu.id
        _forward_op_cpu[op, t1_shape, t2_shape, t3_shape, attributes](
            rebind[Tensor[f32, Device.cpu]](res),
            rebind[Tensor[f32, Device.cpu]](t1),
            rebind[Tensor[f32, Device.cpu]](t2),
            rebind[Tensor[f32, Device.cpu]](t3),
        )
    elif op == OP.LINEAR and t1_shape.rank() >= 2 and t2_shape.rank() == 2:
        # A Linear layer acts on the final dimension.  The leading dimensions
        # are contiguous, so `(B, T, K) @ (K, N)` can use the same rank-2
        # kernel as `(B*T, K) @ (K, N)` without materializing a reshape.
        comptime outer = t1_shape.num_elements() // t1_shape[-1]
        gpu_matmul_bias[outer, t1_shape[-1], t2_shape[-1]](
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
            rebind[Tensor[f32, Device.gpu]](t2),
            rebind[Tensor[f32, Device.gpu]](t3),
        )
    elif op == OP.LAYERNORM and t2_shape.rank() == 1 and t3_shape == t2_shape:
        comptime epsilon = attributes["epsilon"].value().to_scalar[f32]()
        gpu_layernorm_forward(
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
            rebind[Tensor[f32, Device.gpu]](t2),
            rebind[Tensor[f32, Device.gpu]](t3),
            epsilon,
        )
    elif op == OP.CONV2D:
        comptime padding = attributes["padding"].value().to_static[2]()
        comptime stride = attributes["stride"].value().to_static[2]()
        comptime dilation = attributes["dilation"].value().to_static[2]()
        comptime out_shape = CONV2D.result_shape(
            t1_shape, t2_shape, t3_shape, attributes
        )
        gpu_conv2d_forward[
            t1_shape[0],
            t1_shape[1],
            t1_shape[2],
            t1_shape[3],
            t2_shape[0],
            t2_shape[2],
            t2_shape[3],
            out_shape[2],
            out_shape[3],
            padding[0],
            padding[1],
            stride[0],
            stride[1],
            dilation[0],
            dilation[1],
        ](
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
            rebind[Tensor[f32, Device.gpu]](t2),
            rebind[Tensor[f32, Device.gpu]](t3),
        )
    elif op == OP.CONVTRANSPOSE2D:
        # Forward is CONV2D's col2im input-gradient kernel with input/output
        # roles swapped (see conv_transpose.mojo), plus a channel-broadcast
        # bias add since that kernel has no bias epilogue of its own.
        comptime out_shape = CONVTRANSPOSE2D.result_shape(
            t1_shape, t2_shape, t3_shape, attributes
        )
        comptime padding = attributes["padding"].value().to_static[2]()
        comptime stride = attributes["stride"].value().to_static[2]()
        comptime dilation = attributes["dilation"].value().to_static[2]()
        var unbiased = Tensor[f32, Device.gpu](out_shape, uninitialized=True)
        gpu_conv2d_input_backward[
            out_shape[0],
            out_shape[1],
            out_shape[2],
            out_shape[3],
            t2_shape[0],
            t2_shape[2],
            t2_shape[3],
            t1_shape[2],
            t1_shape[3],
            padding[0],
            padding[1],
            stride[0],
            stride[1],
            dilation[0],
            dilation[1],
        ](
            unbiased,
            rebind[Tensor[f32, Device.gpu]](t1),
            rebind[Tensor[f32, Device.gpu]](t2),
        )
        gpu_channel_bias_add_forward[out_shape[1], out_shape[2] * out_shape[3]](
            rebind[Tensor[f32, Device.gpu]](res),
            unbiased,
            rebind[Tensor[f32, Device.gpu]](t3),
        )
    else:
        comptime assert False, "forward_op: unsupported ternary GPU operator"


def _forward_op_cpu[
    op: OP,
    t1_shape: TensorShape,
    t2_shape: TensorShape,
    t3_shape: TensorShape,
    attributes: AttributeVector,
](
    mut res: Tensor[f32],
    t1: Tensor[f32],
    t2: Tensor[f32],
    t3: Tensor[f32],
) raises:
    """
    Forward pass for ternary operators (CPU implementations).
    """

    comptime if op == OP.CONV2D:
        CONV2D.forward[t1_shape, t2_shape, t3_shape, attributes](
            res, t1, t2, t3
        )
    elif op == OP.CONVTRANSPOSE2D:
        CONVTRANSPOSE2D.forward[t1_shape, t2_shape, t3_shape, attributes](
            res, t1, t2, t3
        )
    elif op == OP.FMA:
        FMA.forward[t1_shape, t2_shape, t3_shape](res, t1, t2, t3)
    elif op == OP.BATCHNORM2D:
        BATCHNORM2D.forward[t1_shape, t2_shape, t3_shape, attributes](
            res, t1, t2, t3
        )
    elif op == OP.LINEAR:
        comptime dot_shape = DOT.result_shape(t1_shape, t2_shape)
        comptime n = dot_shape[-1]
        comptime outer = dot_shape.num_elements() // n
        # Matmul overwrites every element, so avoid a full zero-fill of this
        # short-lived intermediate on every Linear forward pass.
        var dot_res = Tensor[f32](dot_shape, uninitialized=True)
        DOT.forward[t1_shape, t2_shape](dot_res, t1, t2)
        cpu_add_bias_forward[outer, n](res, dot_res, t3)
    elif op == OP.LAYERNORM:
        comptime width = t1_shape[-1]
        comptime outer = t1_shape.num_elements() // width
        comptime epsilon = attributes["epsilon"].value().to_scalar[f32]()
        cpu_layernorm_forward[outer, width](res, t1, t2, t3, epsilon)
    else:
        print("[ERROR] Operator not found.")


def forward_op[
    op: OP,
    attributes: AttributeVector,
    device: Device = Device.cpu,
](
    inputs: List[Symbol],
    outputs: List[Symbol],
    mut parameters: Parameters[device],
):
    """
    Forward pass for dynamic operators (CPU-only; CONCAT/SPLIT are out of
    scope for the GPU port).
    """
    comptime assert (
        device.id == Device.cpu.id
    ), "forward_op: dynamic operators (CONCAT/SPLIT) are CPU-only"
    _forward_op_cpu[op, attributes](
        inputs, outputs, rebind[Parameters[Device.cpu]](parameters)
    )


def _forward_op_cpu[
    op: OP,
    attributes: AttributeVector,
](
    inputs: List[Symbol],
    outputs: List[Symbol],
    mut parameters: Parameters[Device.cpu],
):
    """
    Forward pass for dynamic operators (CPU implementation).
    """
    if op == OP.CONCAT:
        CONCAT.forward[attributes](inputs, outputs, parameters)
    elif op == OP.SPLIT:
        SPLIT.forward[attributes](inputs, outputs, parameters)
    else:
        print("[ERROR] Operator not found.")


def backward_conv2d_parameters[
    t1_shape: TensorShape,
    t2_shape: TensorShape,
    t3_shape: TensorShape,
    attributes: AttributeVector,
](
    ug: Tensor[f32, Device.gpu],
    t1: Tensor[f32, Device.gpu],
    mut kernel_grad: Tensor[f32, Device.gpu],
    mut bias_grad: Tensor[f32, Device.gpu],
) raises:
    """Write Conv2d filter and bias gradients using shared temporaries."""
    comptime padding = attributes["padding"].value().to_static[2]()
    comptime stride = attributes["stride"].value().to_static[2]()
    comptime dilation = attributes["dilation"].value().to_static[2]()
    comptime out_shape = CONV2D.result_shape(
        t1_shape, t2_shape, t3_shape, attributes
    )
    gpu_conv2d_parameter_backward_direct[
        t1_shape[0],
        t1_shape[1],
        t1_shape[2],
        t1_shape[3],
        t2_shape[0],
        t2_shape[2],
        t2_shape[3],
        out_shape[2],
        out_shape[3],
        padding[0],
        padding[1],
        stride[0],
        stride[1],
        dilation[0],
        dilation[1],
    ](kernel_grad, bias_grad, t1, ug)


def backward_op[
    tensor_id: Int,
    op: OP,
    ug_shape: TensorShape,
    t1_shape: TensorShape,
    attributes: AttributeVector,
    device: Device = Device.cpu,
    overwrite_grad: Bool = False,
](
    ug: Tensor[f32, device],
    t1: Tensor[f32, device],
    mut grad: Tensor[f32, device],
    runtime_seed: UInt64 = 0,
    training: Bool = True,
) raises:
    """
    Backward pass for unary operators, dispatching by device.
    """
    comptime if device.id == Device.cpu.id:
        comptime assert device.id == Device.cpu.id
        _backward_op_cpu[tensor_id, op, ug_shape, t1_shape, attributes](
            rebind[Tensor[f32, Device.cpu]](ug),
            rebind[Tensor[f32, Device.cpu]](t1),
            rebind[Tensor[f32, Device.cpu]](grad),
            runtime_seed,
            training,
        )
    elif op == OP.RELU:
        comptime if overwrite_grad:
            gpu_relu_backward_into(
                rebind[Tensor[f32, Device.gpu]](grad),
                rebind[Tensor[f32, Device.gpu]](ug),
                rebind[Tensor[f32, Device.gpu]](t1),
            )
        else:
            var res_grad = gpu_relu_backward(
                rebind[Tensor[f32, Device.gpu]](ug),
                rebind[Tensor[f32, Device.gpu]](t1),
            )
            gpu_accumulate_grad(rebind[Tensor[f32, Device.gpu]](grad), res_grad)
    elif op == OP.MAXPOOL2D:
        comptime kernel_size = attributes["kernel_size"].value().to_static[2]()
        comptime padding = attributes["padding"].value().to_static[2]()
        comptime stride = attributes["stride"].value().to_static[2]()
        comptime dilation = attributes["dilation"].value().to_static[2]()
        comptime if overwrite_grad:
            gpu_maxpool2d_backward[
                t1_shape[0],
                t1_shape[1],
                t1_shape[2],
                t1_shape[3],
                ug_shape[2],
                ug_shape[3],
                kernel_size[0],
                kernel_size[1],
                padding[0],
                padding[1],
                stride[0],
                stride[1],
                dilation[0],
                dilation[1],
            ](
                rebind[Tensor[f32, Device.gpu]](grad),
                rebind[Tensor[f32, Device.gpu]](ug),
                rebind[Tensor[f32, Device.gpu]](t1),
            )
        else:
            var res_grad = Tensor[f32, Device.gpu](t1_shape, uninitialized=True)
            gpu_maxpool2d_backward[
                t1_shape[0],
                t1_shape[1],
                t1_shape[2],
                t1_shape[3],
                ug_shape[2],
                ug_shape[3],
                kernel_size[0],
                kernel_size[1],
                padding[0],
                padding[1],
                stride[0],
                stride[1],
                dilation[0],
                dilation[1],
            ](
                res_grad,
                rebind[Tensor[f32, Device.gpu]](ug),
                rebind[Tensor[f32, Device.gpu]](t1),
            )
            gpu_accumulate_grad(rebind[Tensor[f32, Device.gpu]](grad), res_grad)
    elif op == OP.EXP:
        comptime if overwrite_grad:
            gpu_exp_backward_into(
                rebind[Tensor[f32, Device.gpu]](grad),
                rebind[Tensor[f32, Device.gpu]](ug),
                rebind[Tensor[f32, Device.gpu]](t1),
            )
        else:
            var res_grad = gpu_exp_backward(
                rebind[Tensor[f32, Device.gpu]](ug),
                rebind[Tensor[f32, Device.gpu]](t1),
            )
            gpu_accumulate_grad(rebind[Tensor[f32, Device.gpu]](grad), res_grad)
    elif op == OP.LOG:
        comptime if overwrite_grad:
            gpu_log_backward_into(
                rebind[Tensor[f32, Device.gpu]](grad),
                rebind[Tensor[f32, Device.gpu]](ug),
                rebind[Tensor[f32, Device.gpu]](t1),
            )
        else:
            var res_grad = gpu_log_backward(
                rebind[Tensor[f32, Device.gpu]](ug),
                rebind[Tensor[f32, Device.gpu]](t1),
            )
            gpu_accumulate_grad(rebind[Tensor[f32, Device.gpu]](grad), res_grad)
    elif op == OP.GELU:
        comptime if overwrite_grad:
            gpu_gelu_backward_into(
                rebind[Tensor[f32, Device.gpu]](grad),
                rebind[Tensor[f32, Device.gpu]](ug),
                rebind[Tensor[f32, Device.gpu]](t1),
            )
        else:
            var res_grad = gpu_gelu_backward(
                rebind[Tensor[f32, Device.gpu]](ug),
                rebind[Tensor[f32, Device.gpu]](t1),
            )
            gpu_accumulate_grad(rebind[Tensor[f32, Device.gpu]](grad), res_grad)
    elif op == OP.DROPOUT:
        comptime p = attributes["p"].value().to_scalar[f32]()
        comptime base_seed = UInt64(
            attributes["seed"].value().to_scalar[DType.int64]()
        )
        comptime if overwrite_grad:
            gpu_dropout_backward_into(
                rebind[Tensor[f32, Device.gpu]](grad),
                rebind[Tensor[f32, Device.gpu]](ug),
                p,
                base_seed ^ runtime_seed,
                training,
            )
        else:
            var res_grad = gpu_dropout_backward(
                rebind[Tensor[f32, Device.gpu]](ug),
                p,
                base_seed ^ runtime_seed,
                training,
            )
            gpu_accumulate_grad(rebind[Tensor[f32, Device.gpu]](grad), res_grad)
    elif op == OP.MEAN and not attributes["axis"]:
        comptime if overwrite_grad:
            gpu_mean_backward_into(
                rebind[Tensor[f32, Device.gpu]](grad),
                rebind[Tensor[f32, Device.gpu]](ug),
            )
        else:
            var res_grad = gpu_mean_backward(
                rebind[Tensor[f32, Device.gpu]](ug), t1_shape
            )
            gpu_accumulate_grad(rebind[Tensor[f32, Device.gpu]](grad), res_grad)
    elif op == OP.SUM and not attributes["axis"]:
        comptime if overwrite_grad:
            gpu_sum_backward_into(
                rebind[Tensor[f32, Device.gpu]](grad),
                rebind[Tensor[f32, Device.gpu]](ug),
            )
        else:
            var res_grad = gpu_sum_backward(
                rebind[Tensor[f32, Device.gpu]](ug), t1_shape
            )
            gpu_accumulate_grad(rebind[Tensor[f32, Device.gpu]](grad), res_grad)
    elif (
        (op == OP.SUM or op == OP.MEAN or op == OP.MAX)
        and attributes["axis"]
        and attributes["axis"].value().to_int() == t1_shape.rank() - 1
    ):
        comptime if overwrite_grad:
            gpu_reduce_last_backward_into(
                rebind[Tensor[f32, Device.gpu]](grad),
                rebind[Tensor[f32, Device.gpu]](ug),
                rebind[Tensor[f32, Device.gpu]](t1),
                t1_shape[-1],
                gpu_reduce_op_code(op),
            )
        else:
            var res_grad = gpu_reduce_last_backward(
                rebind[Tensor[f32, Device.gpu]](ug),
                rebind[Tensor[f32, Device.gpu]](t1),
                t1_shape[-1],
                gpu_reduce_op_code(op),
            )
            gpu_accumulate_grad(rebind[Tensor[f32, Device.gpu]](grad), res_grad)
    elif op == OP.SQRT:
        comptime if overwrite_grad:
            gpu_sqrt_backward_into(
                rebind[Tensor[f32, Device.gpu]](grad),
                rebind[Tensor[f32, Device.gpu]](ug),
                rebind[Tensor[f32, Device.gpu]](t1),
            )
        else:
            var res_grad = gpu_sqrt_backward(
                rebind[Tensor[f32, Device.gpu]](ug),
                rebind[Tensor[f32, Device.gpu]](t1),
            )
            gpu_accumulate_grad(rebind[Tensor[f32, Device.gpu]](grad), res_grad)
    elif op == OP.RESHAPE or op == OP.FLATTEN:
        # Shape-only operations preserve element order. A value with one
        # backward contributor can receive the upper gradient directly;
        # shared values still need the normal accumulation path.
        comptime if overwrite_grad:
            rebind[Tensor[f32, Device.gpu]](grad).copy_from(
                rebind[Tensor[f32, Device.gpu]](ug)
            )
        else:
            gpu_accumulate_grad(
                rebind[Tensor[f32, Device.gpu]](grad),
                rebind[Tensor[f32, Device.gpu]](ug),
            )
    elif op == OP.TRANSPOSE and t1_shape.rank() == 4:
        comptime axes_attribute = attributes["axes"]
        comptime if overwrite_grad:
            comptime if axes_attribute:
                comptime axes = axes_attribute.value().to_shape()
                comptime inverse_axes = inverse_transpose_axes(axes)
                gpu_transpose_4d(
                    rebind[Tensor[f32, Device.gpu]](grad),
                    rebind[Tensor[f32, Device.gpu]](ug),
                    inverse_axes,
                )
            else:
                gpu_transpose_4d(
                    rebind[Tensor[f32, Device.gpu]](grad),
                    rebind[Tensor[f32, Device.gpu]](ug),
                    TensorShape(3, 2, 1, 0),
                )
        else:
            var res_grad = Tensor[f32, Device.gpu](t1_shape, uninitialized=True)
            comptime if axes_attribute:
                comptime axes = axes_attribute.value().to_shape()
                comptime inverse_axes = inverse_transpose_axes(axes)
                gpu_transpose_4d(
                    res_grad,
                    rebind[Tensor[f32, Device.gpu]](ug),
                    inverse_axes,
                )
            else:
                gpu_transpose_4d(
                    res_grad,
                    rebind[Tensor[f32, Device.gpu]](ug),
                    TensorShape(3, 2, 1, 0),
                )
            gpu_accumulate_grad(rebind[Tensor[f32, Device.gpu]](grad), res_grad)
    else:
        # Host round-trip fallback (mirrors forward_op's).
        var ug_cpu = ug.to_host()
        var t1_cpu = t1.to_host()
        var grad_cpu = grad.to_host()
        _backward_op_cpu[tensor_id, op, ug_shape, t1_shape, attributes](
            ug_cpu, t1_cpu, grad_cpu, runtime_seed, training
        )
        gpu_write_from_host(rebind[Tensor[f32, Device.gpu]](grad), grad_cpu)


def _backward_op_cpu[
    tensor_id: Int,
    op: OP,
    ug_shape: TensorShape,
    t1_shape: TensorShape,
    attributes: AttributeVector,
](
    ug: Tensor[f32],
    t1: Tensor[f32],
    mut grad: Tensor[f32],
    runtime_seed: UInt64 = 0,
    training: Bool = True,
):
    """
    Backward pass for unary operators (CPU implementations).
    """
    var res_grad: Tensor[f32]

    comptime if op == OP.EXP:
        res_grad = EXP.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.LOG:
        res_grad = LOG.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.SUM:
        res_grad = SUM.backward[ug_shape, t1_shape, attributes](ug, t1)
    elif op == OP.MEAN:
        res_grad = MEAN.backward[ug_shape, t1_shape, attributes](ug, t1)
    elif op == OP.MAX:
        res_grad = MAX.backward[ug_shape, t1_shape, attributes](ug, t1)
    elif op == OP.MIN:
        res_grad = MIN.backward[ug_shape, t1_shape, attributes](ug, t1)
    elif op == OP.ARGMAX:
        res_grad = ARGMAX.backward[ug_shape, t1_shape, attributes](ug, t1)
    elif op == OP.FLATTEN:
        res_grad = FLATTEN.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.RESHAPE:
        res_grad = RESHAPE.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.SIGMOID:
        res_grad = SIGMOID.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.RELU:
        res_grad = RELU.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.LEAKYRELU:
        res_grad = LEAKYRELU.backward[ug_shape, t1_shape, attributes](ug, t1)
    elif op == OP.TANH:
        res_grad = TANH.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.GELU:
        res_grad = GELU.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.TRANSPOSE:
        res_grad = TRANSPOSE.backward[ug_shape, t1_shape, attributes](ug, t1)
    elif op == OP.MAXPOOL2D:
        res_grad = MAXPOOL2D.backward[ug_shape, t1_shape, attributes](ug, t1)
    elif op == OP.CLIP:
        res_grad = CLIP.backward[ug_shape, t1_shape, attributes](ug, t1)
    elif op == OP.SQUEEZE:
        res_grad = SQUEEZE.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.UNSQUEEZE:
        res_grad = UNSQUEEZE.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.SLICE:
        res_grad = SLICE.backward[ug_shape, t1_shape, attributes](ug, t1)
    elif op == OP.PAD:
        res_grad = PAD.backward[ug_shape, t1_shape, attributes](ug, t1)
    elif op == OP.DROPOUT:
        res_grad = DROPOUT.backward[ug_shape, t1_shape, attributes](
            ug, t1, runtime_seed, training
        )
    elif op == OP.NEG:
        res_grad = NEG.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.ABS:
        res_grad = ABS.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.SQRT:
        res_grad = SQRT.backward[ug_shape, t1_shape](ug, t1)
    else:
        print("[ERROR] Operator not found.")
        res_grad = Tensor[f32](-1)

    accumulate_grad(grad, res_grad)


def backward_op[
    tensor_id: Int,
    op: OP,
    ug_shape: TensorShape,
    t1_shape: TensorShape,
    t2_shape: TensorShape,
    attributes: AttributeVector,
    device: Device = Device.cpu,
    overwrite_grad: Bool = False,
](
    ug: Tensor[f32, device],
    t1: Tensor[f32, device],
    t2: Tensor[f32, device],
    mut grad: Tensor[f32, device],
) raises:
    """
    Backward pass for binary operators, dispatching by device.
    """
    comptime if device.id == Device.cpu.id:
        comptime assert device.id == Device.cpu.id
        _backward_op_cpu[
            tensor_id, op, ug_shape, t1_shape, t2_shape, attributes
        ](
            rebind[Tensor[f32, Device.cpu]](ug),
            rebind[Tensor[f32, Device.cpu]](t1),
            rebind[Tensor[f32, Device.cpu]](t2),
            rebind[Tensor[f32, Device.cpu]](grad),
        )
    elif t1_shape == t2_shape and (
        op == OP.ADD or op == OP.SUB or op == OP.MUL or op == OP.DIV
    ):
        comptime if overwrite_grad and (
            op == OP.ADD or (op == OP.SUB and tensor_id == 0)
        ):
            rebind[Tensor[f32, Device.gpu]](grad).copy_from(
                rebind[Tensor[f32, Device.gpu]](ug)
            )
        elif overwrite_grad and op == OP.SUB:
            gpu_sub_backward_t2_into(
                rebind[Tensor[f32, Device.gpu]](grad),
                rebind[Tensor[f32, Device.gpu]](ug),
            )
        elif overwrite_grad and op == OP.MUL:
            comptime if tensor_id == 0:
                gpu_mul_backward_into(
                    rebind[Tensor[f32, Device.gpu]](grad),
                    rebind[Tensor[f32, Device.gpu]](ug),
                    rebind[Tensor[f32, Device.gpu]](t2),
                )
            else:
                gpu_mul_backward_into(
                    rebind[Tensor[f32, Device.gpu]](grad),
                    rebind[Tensor[f32, Device.gpu]](ug),
                    rebind[Tensor[f32, Device.gpu]](t1),
                )
        elif overwrite_grad and op == OP.DIV:
            comptime if tensor_id == 0:
                gpu_div_backward_t1_into(
                    rebind[Tensor[f32, Device.gpu]](grad),
                    rebind[Tensor[f32, Device.gpu]](ug),
                    rebind[Tensor[f32, Device.gpu]](t2),
                )
            else:
                gpu_div_backward_t2_into(
                    rebind[Tensor[f32, Device.gpu]](grad),
                    rebind[Tensor[f32, Device.gpu]](ug),
                    rebind[Tensor[f32, Device.gpu]](t1),
                    rebind[Tensor[f32, Device.gpu]](t2),
                )
        else:
            var res_grad: Tensor[f32, Device.gpu]
            comptime if op == OP.ADD:
                res_grad = rebind[Tensor[f32, Device.gpu]](ug).copy()
            elif op == OP.SUB:
                comptime if tensor_id == 0:
                    res_grad = rebind[Tensor[f32, Device.gpu]](ug).copy()
                else:
                    res_grad = gpu_sub_backward_t2(
                        rebind[Tensor[f32, Device.gpu]](ug)
                    )
            elif op == OP.MUL:
                comptime if tensor_id == 0:
                    res_grad = gpu_mul_backward(
                        rebind[Tensor[f32, Device.gpu]](ug),
                        rebind[Tensor[f32, Device.gpu]](t2),
                    )
                else:
                    res_grad = gpu_mul_backward(
                        rebind[Tensor[f32, Device.gpu]](ug),
                        rebind[Tensor[f32, Device.gpu]](t1),
                    )
            else:
                comptime assert op == OP.DIV
                comptime if tensor_id == 0:
                    res_grad = gpu_div_backward_t1(
                        rebind[Tensor[f32, Device.gpu]](ug),
                        rebind[Tensor[f32, Device.gpu]](t2),
                    )
                else:
                    res_grad = gpu_div_backward_t2(
                        rebind[Tensor[f32, Device.gpu]](ug),
                        rebind[Tensor[f32, Device.gpu]](t1),
                        rebind[Tensor[f32, Device.gpu]](t2),
                    )
            gpu_accumulate_grad(rebind[Tensor[f32, Device.gpu]](grad), res_grad)
    elif (
        op == OP.ADD
        and t1_shape.rank() >= 2
        and t2_shape.rank() == 1
        and t2_shape[0] == t1_shape[-1]
    ):
        comptime if tensor_id == 0:
            comptime if overwrite_grad:
                rebind[Tensor[f32, Device.gpu]](grad).copy_from(
                    rebind[Tensor[f32, Device.gpu]](ug)
                )
            else:
                var res_grad = rebind[Tensor[f32, Device.gpu]](ug).copy()
                gpu_accumulate_grad(
                    rebind[Tensor[f32, Device.gpu]](grad), res_grad
                )
        else:
            comptime if overwrite_grad:
                gpu_bias_grad_into(
                    rebind[Tensor[f32, Device.gpu]](grad),
                    rebind[Tensor[f32, Device.gpu]](ug),
                )
            else:
                var res_grad = gpu_bias_grad(
                    rebind[Tensor[f32, Device.gpu]](ug), t2_shape[0]
                )
                gpu_accumulate_grad(
                    rebind[Tensor[f32, Device.gpu]](grad), res_grad
                )
    elif (
        (op == OP.ADD or op == OP.SUB or op == OP.MUL or op == OP.DIV)
        and t1_shape.num_elements() > t2_shape.num_elements()
        and (
            t2_shape.num_elements() == 1
            or (t2_shape.rank() == 1 and t2_shape[0] == t1_shape[-1])
            or (
                t2_shape.rank() == t1_shape.rank()
                and t2_shape[-1] == 1
                and t2_shape.num_elements() * t1_shape[-1]
                == t1_shape.num_elements()
            )
            or leading_broadcast_compatible(t1_shape, t2_shape)
        )
    ):
        comptime vector_mode = (
            t2_shape.rank() == 1 and t2_shape[0] == t1_shape[-1]
        )
        comptime leading_broadcast = leading_broadcast_compatible(
            t1_shape, t2_shape
        )
        comptime repeat = 1 if vector_mode or leading_broadcast else (
            t1_shape.num_elements() // t2_shape.num_elements()
        )
        comptime if tensor_id == 0:
            comptime if overwrite_grad and op == OP.ADD:
                rebind[Tensor[f32, Device.gpu]](grad).copy_from(
                    rebind[Tensor[f32, Device.gpu]](ug)
                )
            else:
                var res_grad = gpu_broadcast_binary_backward_t1(
                    rebind[Tensor[f32, Device.gpu]](ug),
                    rebind[Tensor[f32, Device.gpu]](t1),
                    rebind[Tensor[f32, Device.gpu]](t2),
                    repeat,
                    0 if vector_mode else (2 if leading_broadcast else 1),
                    gpu_binary_op_code(op),
                )
                gpu_accumulate_grad(
                    rebind[Tensor[f32, Device.gpu]](grad), res_grad
                )
        else:
            var res_grad = gpu_broadcast_binary_backward_t2(
                rebind[Tensor[f32, Device.gpu]](ug),
                rebind[Tensor[f32, Device.gpu]](t1),
                rebind[Tensor[f32, Device.gpu]](t2),
                repeat,
                0 if vector_mode else (2 if leading_broadcast else 1),
                gpu_binary_op_code(op),
            )
            gpu_accumulate_grad(rebind[Tensor[f32, Device.gpu]](grad), res_grad)
    elif t1_shape.rank() == 2 and t2_shape.rank() == 2 and op == OP.DOT:
        comptime if tensor_id == 0:
            comptime if overwrite_grad:
                gpu_matmul_bt[ug_shape[0], ug_shape[1], t2_shape[0]](
                    rebind[Tensor[f32, Device.gpu]](grad),
                    rebind[Tensor[f32, Device.gpu]](ug),
                    rebind[Tensor[f32, Device.gpu]](t2),
                )
            else:
                var res_grad = Tensor[f32, Device.gpu](
                    t1_shape, uninitialized=True
                )
                gpu_matmul_bt[ug_shape[0], ug_shape[1], t2_shape[0]](
                    res_grad,
                    rebind[Tensor[f32, Device.gpu]](ug),
                    rebind[Tensor[f32, Device.gpu]](t2),
                )
                gpu_accumulate_grad(
                    rebind[Tensor[f32, Device.gpu]](grad), res_grad
                )
        else:
            comptime if overwrite_grad:
                gpu_matmul_at[t1_shape[0], t1_shape[1], ug_shape[1]](
                    rebind[Tensor[f32, Device.gpu]](grad),
                    rebind[Tensor[f32, Device.gpu]](t1),
                    rebind[Tensor[f32, Device.gpu]](ug),
                )
            else:
                var res_grad = Tensor[f32, Device.gpu](
                    t2_shape, uninitialized=True
                )
                gpu_matmul_at[t1_shape[0], t1_shape[1], ug_shape[1]](
                    res_grad,
                    rebind[Tensor[f32, Device.gpu]](t1),
                    rebind[Tensor[f32, Device.gpu]](ug),
                )
                gpu_accumulate_grad(
                    rebind[Tensor[f32, Device.gpu]](grad), res_grad
                )
    elif (
        op == OP.DOT
        and t1_shape.rank() >= 3
        and matching_batch_dims(t1_shape, t2_shape)
    ):
        comptime batches = t1_shape.num_elements() // (
            t1_shape[-2] * t1_shape[-1]
        )
        comptime if tensor_id == 0:
            comptime if overwrite_grad:
                gpu_batched_matmul_bt[
                    batches, ug_shape[-2], ug_shape[-1], t2_shape[-2]
                ](
                    rebind[Tensor[f32, Device.gpu]](grad),
                    rebind[Tensor[f32, Device.gpu]](ug),
                    rebind[Tensor[f32, Device.gpu]](t2),
                )
            else:
                var res_grad = Tensor[f32, Device.gpu](
                    t1_shape, uninitialized=True
                )
                gpu_batched_matmul_bt[
                    batches, ug_shape[-2], ug_shape[-1], t2_shape[-2]
                ](
                    res_grad,
                    rebind[Tensor[f32, Device.gpu]](ug),
                    rebind[Tensor[f32, Device.gpu]](t2),
                )
                gpu_accumulate_grad(
                    rebind[Tensor[f32, Device.gpu]](grad), res_grad
                )
        else:
            comptime if overwrite_grad:
                gpu_batched_matmul_at[
                    batches, t1_shape[-2], t1_shape[-1], ug_shape[-1]
                ](
                    rebind[Tensor[f32, Device.gpu]](grad),
                    rebind[Tensor[f32, Device.gpu]](t1),
                    rebind[Tensor[f32, Device.gpu]](ug),
                )
            else:
                var res_grad = Tensor[f32, Device.gpu](
                    t2_shape, uninitialized=True
                )
                gpu_batched_matmul_at[
                    batches, t1_shape[-2], t1_shape[-1], ug_shape[-1]
                ](
                    res_grad,
                    rebind[Tensor[f32, Device.gpu]](t1),
                    rebind[Tensor[f32, Device.gpu]](ug),
                )
                gpu_accumulate_grad(
                    rebind[Tensor[f32, Device.gpu]](grad), res_grad
                )
    elif op == OP.GATHER and t1_shape.rank() == 2:
        comptime if tensor_id == 0:
            var res_grad = gpu_gather_backward(
                rebind[Tensor[f32, Device.gpu]](ug),
                rebind[Tensor[f32, Device.gpu]](t1),
                rebind[Tensor[f32, Device.gpu]](t2),
                t1_shape[-1],
            )
            gpu_accumulate_grad(rebind[Tensor[f32, Device.gpu]](grad), res_grad)
        else:
            comptime assert False, "GATHER does not differentiate indices"
    elif op == OP.POW:
        # The exponent (t2) is never trainable, so the caller
        # (Model.backward) never invokes tensor_id == 1 here.
        comptime assert tensor_id == 0
        comptime if overwrite_grad:
            gpu_pow_backward_into(
                rebind[Tensor[f32, Device.gpu]](grad),
                rebind[Tensor[f32, Device.gpu]](ug),
                rebind[Tensor[f32, Device.gpu]](t1),
                rebind[Tensor[f32, Device.gpu]](t2),
            )
        else:
            var res_grad = gpu_pow_backward(
                rebind[Tensor[f32, Device.gpu]](ug),
                rebind[Tensor[f32, Device.gpu]](t1),
                rebind[Tensor[f32, Device.gpu]](t2),
            )
            gpu_accumulate_grad(rebind[Tensor[f32, Device.gpu]](grad), res_grad)
    else:
        # Host round-trip fallback for GPU ops without a native kernel yet.
        var ug_cpu = ug.to_host()
        var t1_cpu = t1.to_host()
        var t2_cpu = t2.to_host()
        var grad_cpu = grad.to_host()
        _backward_op_cpu[
            tensor_id, op, ug_shape, t1_shape, t2_shape, attributes
        ](ug_cpu, t1_cpu, t2_cpu, grad_cpu)
        gpu_write_from_host(rebind[Tensor[f32, Device.gpu]](grad), grad_cpu)


def _backward_op_cpu[
    tensor_id: Int,
    op: OP,
    ug_shape: TensorShape,
    t1_shape: TensorShape,
    t2_shape: TensorShape,
    attributes: AttributeVector,
](
    ug: Tensor[f32],
    t1: Tensor[f32],
    t2: Tensor[f32],
    mut grad: Tensor[f32],
) raises:
    """
    Backward pass for binary operators (CPU implementations).
    """
    var res_grad: Tensor[f32]

    comptime if op == OP.ADD:
        res_grad = ADD.backward[tensor_id, ug_shape, t1_shape, t2_shape](
            ug, t1, t2
        )
    elif op == OP.SUB:
        res_grad = SUB.backward[tensor_id, ug_shape, t1_shape, t2_shape](
            ug, t1, t2
        )
    elif op == OP.MUL:
        res_grad = MUL.backward[tensor_id, ug_shape, t1_shape, t2_shape](
            ug, t1, t2
        )
    elif op == OP.DIV:
        res_grad = DIV.backward[tensor_id, ug_shape, t1_shape, t2_shape](
            ug, t1, t2
        )
    elif op == OP.POW:
        res_grad = POW.backward[tensor_id, ug_shape, t1_shape, t2_shape](
            ug, t1, t2
        )
    elif op == OP.DOT:
        res_grad = DOT.backward[tensor_id, ug_shape, t1_shape, t2_shape](
            ug, t1, t2
        )
    elif op == OP.GATHER:
        res_grad = GATHER.backward[tensor_id, ug_shape, t1_shape, t2_shape](
            ug, t1, t2
        )
    else:
        print("[ERROR] Operator not found.")
        res_grad = Tensor[f32](-1, -1)

    def broadcastable(op: OP) -> Bool:
        return op == OP.ADD or op == OP.SUB or op == OP.MUL or op == OP.DIV

    comptime if broadcastable(op):
        accumulate_grad[
            grad_shape=t1_shape if tensor_id == 0 else t2_shape,
            res_grad_shape=broadcast_shapes(t1_shape, t2_shape),
        ](grad, res_grad)
    elif op == OP.DOT:
        comptime if tensor_id == 0:
            accumulate_grad[
                grad_shape=t1_shape,
                res_grad_shape=dot_batch_broadcast_shape(
                    ug_shape, TensorShape(t2_shape[-1], t2_shape[-2])
                ),
            ](grad, res_grad)
        else:
            accumulate_grad[
                grad_shape=t2_shape,
                res_grad_shape=dot_batch_broadcast_shape(
                    TensorShape(t1_shape[-1], t1_shape[-2]), ug_shape
                ),
            ](grad, res_grad)
    else:
        accumulate_grad(grad, res_grad)


def backward_op[
    tensor_id: Int,
    op: OP,
    ug_shape: TensorShape,
    t1_shape: TensorShape,
    t2_shape: TensorShape,
    t3_shape: TensorShape,
    attributes: AttributeVector,
    device: Device = Device.cpu,
    overwrite_grad: Bool = False,
](
    ug: Tensor[f32, device],
    t1: Tensor[f32, device],
    t2: Tensor[f32, device],
    t3: Tensor[f32, device],
    mut grad: Tensor[f32, device],
) raises:
    """
    Backward pass for ternary operators, dispatching by device (see
    `forward_op`'s ternary overload for which ops have GPU kernels).
    """
    comptime if device.id == Device.cpu.id:
        comptime assert device.id == Device.cpu.id
        comptime if overwrite_grad and op == OP.LINEAR and t1_shape.rank() == 2 and t2_shape.rank() == 2:
            # A single-consumer gradient was zeroed before this backward pass,
            # so write the matmul result directly into its final buffer.
            # This avoids both a large temporary tensor and a second full
            # accumulation pass; shared graph values retain the path below.
            comptime if tensor_id == 0:
                dot_transpose_t2[ug_shape, t2_shape](
                    rebind[Tensor[f32, Device.cpu]](grad),
                    rebind[Tensor[f32, Device.cpu]](ug),
                    rebind[Tensor[f32, Device.cpu]](t2),
                )
            elif tensor_id == 1:
                dot_transpose_t1[t1_shape, ug_shape](
                    rebind[Tensor[f32, Device.cpu]](grad),
                    rebind[Tensor[f32, Device.cpu]](t1),
                    rebind[Tensor[f32, Device.cpu]](ug),
                )
            else:
                comptime assert tensor_id == 2
                comptime n = ug_shape[-1]
                comptime outer = ug_shape.num_elements() // n
                cpu_bias_grad_overwrite[outer, n](
                    rebind[Tensor[f32, Device.cpu]](grad),
                    rebind[Tensor[f32, Device.cpu]](ug),
                )
        else:
            _backward_op_cpu[
                tensor_id,
                op,
                ug_shape,
                t1_shape,
                t2_shape,
                t3_shape,
                attributes,
            ](
                rebind[Tensor[f32, Device.cpu]](ug),
                rebind[Tensor[f32, Device.cpu]](t1),
                rebind[Tensor[f32, Device.cpu]](t2),
                rebind[Tensor[f32, Device.cpu]](t3),
                rebind[Tensor[f32, Device.cpu]](grad),
            )
    elif op == OP.LINEAR and t1_shape.rank() >= 2 and t2_shape.rank() == 2:
        # See the forward path: all leading dimensions form one contiguous
        # matrix row dimension for Linear's input-gradient product.
        comptime outer = t1_shape.num_elements() // t1_shape[-1]
        comptime if tensor_id == 0:
            comptime if overwrite_grad:
                gpu_matmul_bt[outer, ug_shape[-1], t2_shape[0]](
                    rebind[Tensor[f32, Device.gpu]](grad),
                    rebind[Tensor[f32, Device.gpu]](ug),
                    rebind[Tensor[f32, Device.gpu]](t2),
                )
            else:
                var res_grad = Tensor[f32, Device.gpu](
                    t1_shape, uninitialized=True
                )
                gpu_matmul_bt[outer, ug_shape[-1], t2_shape[0]](
                    res_grad,
                    rebind[Tensor[f32, Device.gpu]](ug),
                    rebind[Tensor[f32, Device.gpu]](t2),
                )
                gpu_accumulate_grad(
                    rebind[Tensor[f32, Device.gpu]](grad), res_grad
                )
        elif tensor_id == 1:
            comptime if overwrite_grad:
                gpu_matmul_at[outer, t1_shape[-1], ug_shape[-1]](
                    rebind[Tensor[f32, Device.gpu]](grad),
                    rebind[Tensor[f32, Device.gpu]](t1),
                    rebind[Tensor[f32, Device.gpu]](ug),
                )
            else:
                var res_grad = Tensor[f32, Device.gpu](
                    t2_shape, uninitialized=True
                )
                gpu_matmul_at[outer, t1_shape[-1], ug_shape[-1]](
                    res_grad,
                    rebind[Tensor[f32, Device.gpu]](t1),
                    rebind[Tensor[f32, Device.gpu]](ug),
                )
                gpu_accumulate_grad(
                    rebind[Tensor[f32, Device.gpu]](grad), res_grad
                )
        else:
            comptime assert tensor_id == 2
            comptime if overwrite_grad:
                gpu_bias_grad_into(
                    rebind[Tensor[f32, Device.gpu]](grad),
                    rebind[Tensor[f32, Device.gpu]](ug),
                )
            else:
                var res_grad = gpu_bias_grad(
                    rebind[Tensor[f32, Device.gpu]](ug), t3_shape[0]
                )
                gpu_accumulate_grad(
                    rebind[Tensor[f32, Device.gpu]](grad), res_grad
                )
    elif op == OP.CONV2D:
        comptime padding = attributes["padding"].value().to_static[2]()
        comptime stride = attributes["stride"].value().to_static[2]()
        comptime dilation = attributes["dilation"].value().to_static[2]()
        comptime out_shape = CONV2D.result_shape(
            t1_shape, t2_shape, t3_shape, attributes
        )
        comptime if tensor_id == 0:
            comptime if overwrite_grad:
                gpu_conv2d_input_backward[
                    t1_shape[0],
                    t1_shape[1],
                    t1_shape[2],
                    t1_shape[3],
                    t2_shape[0],
                    t2_shape[2],
                    t2_shape[3],
                    out_shape[2],
                    out_shape[3],
                    padding[0],
                    padding[1],
                    stride[0],
                    stride[1],
                    dilation[0],
                    dilation[1],
                ](
                    rebind[Tensor[f32, Device.gpu]](grad),
                    rebind[Tensor[f32, Device.gpu]](ug),
                    rebind[Tensor[f32, Device.gpu]](t2),
                )
            else:
                var res_grad = Tensor[f32, Device.gpu](
                    t1_shape, uninitialized=True
                )
                gpu_conv2d_input_backward[
                    t1_shape[0],
                    t1_shape[1],
                    t1_shape[2],
                    t1_shape[3],
                    t2_shape[0],
                    t2_shape[2],
                    t2_shape[3],
                    out_shape[2],
                    out_shape[3],
                    padding[0],
                    padding[1],
                    stride[0],
                    stride[1],
                    dilation[0],
                    dilation[1],
                ](
                    res_grad,
                    rebind[Tensor[f32, Device.gpu]](ug),
                    rebind[Tensor[f32, Device.gpu]](t2),
                )
                gpu_accumulate_grad(
                    rebind[Tensor[f32, Device.gpu]](grad), res_grad
                )
        elif tensor_id == 1:
            comptime if overwrite_grad:
                gpu_conv2d_kernel_backward[
                    t1_shape[0],
                    t1_shape[1],
                    t1_shape[2],
                    t1_shape[3],
                    t2_shape[0],
                    t2_shape[2],
                    t2_shape[3],
                    out_shape[2],
                    out_shape[3],
                    padding[0],
                    padding[1],
                    stride[0],
                    stride[1],
                    dilation[0],
                    dilation[1],
                ](
                    rebind[Tensor[f32, Device.gpu]](grad),
                    rebind[Tensor[f32, Device.gpu]](t1),
                    rebind[Tensor[f32, Device.gpu]](ug),
                )
            else:
                var res_grad = Tensor[f32, Device.gpu](
                    t2_shape, uninitialized=True
                )
                gpu_conv2d_kernel_backward[
                    t1_shape[0],
                    t1_shape[1],
                    t1_shape[2],
                    t1_shape[3],
                    t2_shape[0],
                    t2_shape[2],
                    t2_shape[3],
                    out_shape[2],
                    out_shape[3],
                    padding[0],
                    padding[1],
                    stride[0],
                    stride[1],
                    dilation[0],
                    dilation[1],
                ](
                    res_grad,
                    rebind[Tensor[f32, Device.gpu]](t1),
                    rebind[Tensor[f32, Device.gpu]](ug),
                )
                gpu_accumulate_grad(
                    rebind[Tensor[f32, Device.gpu]](grad), res_grad
                )
        else:
            comptime assert tensor_id == 2
            comptime if overwrite_grad:
                gpu_conv2d_bias_backward[
                    t1_shape[0], t2_shape[0], out_shape[2], out_shape[3]
                ](
                    rebind[Tensor[f32, Device.gpu]](grad),
                    rebind[Tensor[f32, Device.gpu]](ug),
                )
            else:
                var res_grad = Tensor[f32, Device.gpu](
                    t3_shape, uninitialized=True
                )
                gpu_conv2d_bias_backward[
                    t1_shape[0], t2_shape[0], out_shape[2], out_shape[3]
                ](res_grad, rebind[Tensor[f32, Device.gpu]](ug))
                gpu_accumulate_grad(
                    rebind[Tensor[f32, Device.gpu]](grad), res_grad
                )
    elif op == OP.LAYERNORM:
        comptime epsilon = attributes["epsilon"].value().to_scalar[f32]()
        comptime if tensor_id == 0:
            comptime if overwrite_grad:
                gpu_layernorm_input_backward(
                    rebind[Tensor[f32, Device.gpu]](grad),
                    rebind[Tensor[f32, Device.gpu]](ug),
                    rebind[Tensor[f32, Device.gpu]](t1),
                    rebind[Tensor[f32, Device.gpu]](t2),
                    epsilon,
                )
            else:
                var res_grad = Tensor[f32, Device.gpu](
                    t1_shape, uninitialized=True
                )
                gpu_layernorm_input_backward(
                    res_grad,
                    rebind[Tensor[f32, Device.gpu]](ug),
                    rebind[Tensor[f32, Device.gpu]](t1),
                    rebind[Tensor[f32, Device.gpu]](t2),
                    epsilon,
                )
                gpu_accumulate_grad(
                    rebind[Tensor[f32, Device.gpu]](grad), res_grad
                )
        else:
            comptime affine_id = tensor_id - 1
            comptime target_shape = t2_shape if tensor_id == 1 else t3_shape
            comptime if overwrite_grad:
                gpu_layernorm_affine_backward(
                    rebind[Tensor[f32, Device.gpu]](grad),
                    rebind[Tensor[f32, Device.gpu]](ug),
                    rebind[Tensor[f32, Device.gpu]](t1),
                    epsilon,
                    Int64(affine_id),
                )
            else:
                var res_grad = Tensor[f32, Device.gpu](
                    target_shape, uninitialized=True
                )
                gpu_layernorm_affine_backward(
                    res_grad,
                    rebind[Tensor[f32, Device.gpu]](ug),
                    rebind[Tensor[f32, Device.gpu]](t1),
                    epsilon,
                    Int64(affine_id),
                )
                gpu_accumulate_grad(
                    rebind[Tensor[f32, Device.gpu]](grad), res_grad
                )
    elif op == OP.CONVTRANSPOSE2D:
        # Mirrors conv_transpose.mojo's CPU backward: each tensor_id
        # delegates to the matching CONV2D GPU kernel with input/output
        # roles swapped.
        comptime padding = attributes["padding"].value().to_static[2]()
        comptime stride = attributes["stride"].value().to_static[2]()
        comptime dilation = attributes["dilation"].value().to_static[2]()
        comptime if tensor_id == 0:
            var zero_bias = Tensor[f32, Device.gpu](
                TensorShape(t2_shape[0])
            )
            comptime if overwrite_grad:
                gpu_conv2d_forward[
                    ug_shape[0], ug_shape[1], ug_shape[2], ug_shape[3],
                    t2_shape[0], t2_shape[2], t2_shape[3],
                    t1_shape[2], t1_shape[3],
                    padding[0], padding[1], stride[0], stride[1],
                    dilation[0], dilation[1],
                ](
                    rebind[Tensor[f32, Device.gpu]](grad),
                    rebind[Tensor[f32, Device.gpu]](ug),
                    rebind[Tensor[f32, Device.gpu]](t2),
                    zero_bias,
                )
            else:
                var res_grad = Tensor[f32, Device.gpu](
                    t1_shape, uninitialized=True
                )
                gpu_conv2d_forward[
                    ug_shape[0], ug_shape[1], ug_shape[2], ug_shape[3],
                    t2_shape[0], t2_shape[2], t2_shape[3],
                    t1_shape[2], t1_shape[3],
                    padding[0], padding[1], stride[0], stride[1],
                    dilation[0], dilation[1],
                ](
                    res_grad,
                    rebind[Tensor[f32, Device.gpu]](ug),
                    rebind[Tensor[f32, Device.gpu]](t2),
                    zero_bias,
                )
                gpu_accumulate_grad(
                    rebind[Tensor[f32, Device.gpu]](grad), res_grad
                )
        elif tensor_id == 1:
            comptime if overwrite_grad:
                gpu_conv2d_kernel_backward[
                    ug_shape[0], ug_shape[1], ug_shape[2], ug_shape[3],
                    t2_shape[0], t2_shape[2], t2_shape[3],
                    t1_shape[2], t1_shape[3],
                    padding[0], padding[1], stride[0], stride[1],
                    dilation[0], dilation[1],
                ](
                    rebind[Tensor[f32, Device.gpu]](grad),
                    rebind[Tensor[f32, Device.gpu]](ug),
                    rebind[Tensor[f32, Device.gpu]](t1),
                )
            else:
                var res_grad = Tensor[f32, Device.gpu](
                    t2_shape, uninitialized=True
                )
                gpu_conv2d_kernel_backward[
                    ug_shape[0], ug_shape[1], ug_shape[2], ug_shape[3],
                    t2_shape[0], t2_shape[2], t2_shape[3],
                    t1_shape[2], t1_shape[3],
                    padding[0], padding[1], stride[0], stride[1],
                    dilation[0], dilation[1],
                ](
                    res_grad,
                    rebind[Tensor[f32, Device.gpu]](ug),
                    rebind[Tensor[f32, Device.gpu]](t1),
                )
                gpu_accumulate_grad(
                    rebind[Tensor[f32, Device.gpu]](grad), res_grad
                )
        else:
            comptime assert tensor_id == 2
            comptime if overwrite_grad:
                gpu_conv2d_bias_backward[
                    ug_shape[0], ug_shape[1], ug_shape[2], ug_shape[3]
                ](
                    rebind[Tensor[f32, Device.gpu]](grad),
                    rebind[Tensor[f32, Device.gpu]](ug),
                )
            else:
                var res_grad = Tensor[f32, Device.gpu](
                    t3_shape, uninitialized=True
                )
                gpu_conv2d_bias_backward[
                    ug_shape[0], ug_shape[1], ug_shape[2], ug_shape[3]
                ](res_grad, rebind[Tensor[f32, Device.gpu]](ug))
                gpu_accumulate_grad(
                    rebind[Tensor[f32, Device.gpu]](grad), res_grad
                )
    else:
        comptime assert False, "backward_op: unsupported ternary GPU operator"


def _backward_op_cpu[
    tensor_id: Int,
    op: OP,
    ug_shape: TensorShape,
    t1_shape: TensorShape,
    t2_shape: TensorShape,
    t3_shape: TensorShape,
    attributes: AttributeVector,
](
    ug: Tensor[f32],
    t1: Tensor[f32],
    t2: Tensor[f32],
    t3: Tensor[f32],
    mut grad: Tensor[f32],
) raises:
    """
    Backward pass for ternary operators (CPU implementations).
    """
    comptime if op == OP.LINEAR:
        comptime if tensor_id == 0:
            var res_grad = DOT.backward[0, ug_shape, t1_shape, t2_shape](
                ug, t1, t2
            )
            accumulate_grad[
                grad_shape=t1_shape,
                res_grad_shape=dot_batch_broadcast_shape(
                    ug_shape, TensorShape(t2_shape[-1], t2_shape[-2])
                ),
            ](grad, res_grad)
        elif tensor_id == 1:
            var res_grad = DOT.backward[1, ug_shape, t1_shape, t2_shape](
                ug, t1, t2
            )
            accumulate_grad[
                grad_shape=t2_shape,
                res_grad_shape=dot_batch_broadcast_shape(
                    TensorShape(t1_shape[-1], t1_shape[-2]), ug_shape
                ),
            ](grad, res_grad)
        else:
            comptime assert tensor_id == 2
            comptime n = ug_shape[-1]
            comptime outer = ug_shape.num_elements() // n
            cpu_bias_grad_accumulate[outer, n](grad, ug)
    elif op == OP.LAYERNORM:
        comptime width = t1_shape[-1]
        comptime outer = t1_shape.num_elements() // width
        comptime epsilon = attributes["epsilon"].value().to_scalar[f32]()
        comptime if tensor_id == 0:
            cpu_layernorm_input_backward[outer, width](
                ug, t1, t2, grad, epsilon
            )
        elif tensor_id == 1:
            cpu_layernorm_affine_backward[outer, width, 0](
                ug, t1, grad, epsilon
            )
        else:
            comptime assert tensor_id == 2
            cpu_layernorm_affine_backward[outer, width, 1](
                ug, t1, grad, epsilon
            )
    else:
        var res_grad: Tensor[f32]

        comptime if op == OP.CONV2D:
            res_grad = CONV2D.backward[
                tensor_id, ug_shape, t1_shape, t2_shape, t3_shape, attributes
            ](ug, t1, t2, t3)
        elif op == OP.CONVTRANSPOSE2D:
            res_grad = CONVTRANSPOSE2D.backward[
                tensor_id, ug_shape, t1_shape, t2_shape, t3_shape, attributes
            ](ug, t1, t2, t3)
        elif op == OP.FMA:
            res_grad = FMA.backward[
                tensor_id, ug_shape, t1_shape, t2_shape, t3_shape
            ](ug, t1, t2, t3)
        elif op == OP.BATCHNORM2D:
            res_grad = BATCHNORM2D.backward[
                tensor_id, ug_shape, t1_shape, t2_shape, t3_shape, attributes
            ](ug, t1, t2, t3)
        else:
            print("[ERROR] Operator not found.")
            res_grad = Tensor[f32](-1, -1)

        accumulate_grad(grad, res_grad)


def backward_op[
    input_id: Int,
    op: OP,
    attributes: AttributeVector,
    device: Device = Device.cpu,
](
    inputs: List[Symbol],
    outputs: List[Symbol],
    mut grad: Tensor[f32, device],
    mut parameters: Parameters[device],
):
    """
    Backward pass for dynamic operators (CPU-only; CONCAT/SPLIT are out of
    scope for the GPU port).
    """
    comptime assert (
        device.id == Device.cpu.id
    ), "backward_op: dynamic operators (CONCAT/SPLIT) are CPU-only"
    _backward_op_cpu[input_id, op, attributes](
        inputs,
        outputs,
        rebind[Tensor[f32, Device.cpu]](grad),
        rebind[Parameters[Device.cpu]](parameters),
    )


def _backward_op_cpu[
    input_id: Int,
    op: OP,
    attributes: AttributeVector,
](
    inputs: List[Symbol],
    outputs: List[Symbol],
    mut grad: Tensor[f32],
    mut parameters: Parameters[Device.cpu],
):
    """
    Backward pass for dynamic operators (CPU implementation).
    """
    var res_grad: Tensor[f32]

    if op == OP.CONCAT:
        res_grad = CONCAT.backward[input_id, attributes](
            inputs, outputs, parameters
        )
    elif op == OP.SPLIT:
        res_grad = SPLIT.backward[input_id, attributes](
            inputs, outputs, parameters
        )
    else:
        print("[ERROR] Operator not found.")
        res_grad = Tensor[f32](-1, -1)

    accumulate_grad(grad, res_grad)

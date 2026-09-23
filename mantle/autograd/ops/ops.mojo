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
from .pool import MAXPOOL2D, AVGPOOL2D

from std.os import abort

from mantle import f32
from mantle.autograd.symbol import Symbol
from mantle.core.tensor import Tensor, TensorShape
from mantle.core.device import Device
from mantle.nn.parameters import Parameters
from mantle.core.bytes import Bytes
from mantle.core.tensorutils import broadcast_shapes, accumulate_grad
from mantle.autograd.attributes import AttributeVector
from .gpu_elementwise import (
    gpu_add_forward,
    gpu_sub_forward,
    gpu_mul_forward,
    gpu_div_forward,
    gpu_relu_forward,
    gpu_relu_backward,
    gpu_sub_backward_t2,
    gpu_mul_backward,
    gpu_div_backward_t1,
    gpu_div_backward_t2,
    gpu_accumulate_grad,
)


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
    elif op == OP.FMA:
        return FMA.result_shape(t1_shape, t2_shape, t3_shape)
    elif op == OP.BATCHNORM2D:
        return BATCHNORM2D.result_shape(
            t1_shape, t2_shape, t3_shape, attributes
        )
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
    else:
        abort("forward_op: unary operator " + String(op) + " is not supported on GPU")


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
    mut res: Tensor[f32, device], t1: Tensor[f32, device], t2: Tensor[f32, device]
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
    elif t1_shape != t2_shape:
        abort("forward_op: GPU binary operators do not support broadcasting")
    elif op == OP.ADD:
        gpu_add_forward(
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
            rebind[Tensor[f32, Device.gpu]](t2),
        )
    elif op == OP.SUB:
        gpu_sub_forward(
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
            rebind[Tensor[f32, Device.gpu]](t2),
        )
    elif op == OP.MUL:
        gpu_mul_forward(
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
            rebind[Tensor[f32, Device.gpu]](t2),
        )
    elif op == OP.DIV:
        gpu_div_forward(
            rebind[Tensor[f32, Device.gpu]](res),
            rebind[Tensor[f32, Device.gpu]](t1),
            rebind[Tensor[f32, Device.gpu]](t2),
        )
    else:
        abort("forward_op: binary operator " + String(op) + " is not supported on GPU")


def _forward_op_cpu[
    op: OP,
    t1_shape: TensorShape,
    t2_shape: TensorShape,
    attributes: AttributeVector,
](mut res: Tensor[f32], t1: Tensor[f32], t2: Tensor[f32]):
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
):
    """
    Forward pass for ternary operators (CPU-only; every ternary op —
    CONV2D/FMA/BATCHNORM2D — is out of scope for the GPU port).
    """
    comptime assert device.id == Device.cpu.id, (
        "forward_op: ternary operators are CPU-only"
    )
    _forward_op_cpu[op, t1_shape, t2_shape, t3_shape, attributes](
        rebind[Tensor[f32, Device.cpu]](res),
        rebind[Tensor[f32, Device.cpu]](t1),
        rebind[Tensor[f32, Device.cpu]](t2),
        rebind[Tensor[f32, Device.cpu]](t3),
    )


def _forward_op_cpu[
    op: OP,
    t1_shape: TensorShape,
    t2_shape: TensorShape,
    t3_shape: TensorShape,
    attributes: AttributeVector,
](mut res: Tensor[f32], t1: Tensor[f32], t2: Tensor[f32], t3: Tensor[f32],):
    """
    Forward pass for ternary operators (CPU implementations).
    """

    comptime if op == OP.CONV2D:
        CONV2D.forward[t1_shape, t2_shape, t3_shape, attributes](
            res, t1, t2, t3
        )
    elif op == OP.FMA:
        FMA.forward[t1_shape, t2_shape, t3_shape](res, t1, t2, t3)
    elif op == OP.BATCHNORM2D:
        BATCHNORM2D.forward[t1_shape, t2_shape, t3_shape, attributes](
            res, t1, t2, t3
        )
    else:
        print("[ERROR] Operator not found.")


def forward_op[
    op: OP,
    attributes: AttributeVector,
    device: Device = Device.cpu,
](inputs: List[Symbol], outputs: List[Symbol], mut parameters: Parameters[device]):
    """
    Forward pass for dynamic operators (CPU-only; CONCAT/SPLIT are out of
    scope for the GPU port).
    """
    comptime assert device.id == Device.cpu.id, (
        "forward_op: dynamic operators (CONCAT/SPLIT) are CPU-only"
    )
    _forward_op_cpu[op, attributes](
        inputs, outputs, rebind[Parameters[Device.cpu]](parameters)
    )


def _forward_op_cpu[
    op: OP,
    attributes: AttributeVector,
](inputs: List[Symbol], outputs: List[Symbol], mut parameters: Parameters[Device.cpu],):
    """
    Forward pass for dynamic operators (CPU implementation).
    """
    if op == OP.CONCAT:
        CONCAT.forward[attributes](inputs, outputs, parameters)
    elif op == OP.SPLIT:
        SPLIT.forward[attributes](inputs, outputs, parameters)
    else:
        print("[ERROR] Operator not found.")


def backward_op[
    tensor_id: Int,
    op: OP,
    ug_shape: TensorShape,
    t1_shape: TensorShape,
    attributes: AttributeVector,
    device: Device = Device.cpu,
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
        var res_grad = gpu_relu_backward(
            rebind[Tensor[f32, Device.gpu]](ug),
            rebind[Tensor[f32, Device.gpu]](t1),
        )
        gpu_accumulate_grad(rebind[Tensor[f32, Device.gpu]](grad), res_grad)
    else:
        abort("backward_op: unary operator " + String(op) + " is not supported on GPU")


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
        _backward_op_cpu[tensor_id, op, ug_shape, t1_shape, t2_shape, attributes](
            rebind[Tensor[f32, Device.cpu]](ug),
            rebind[Tensor[f32, Device.cpu]](t1),
            rebind[Tensor[f32, Device.cpu]](t2),
            rebind[Tensor[f32, Device.cpu]](grad),
        )
    elif t1_shape != t2_shape:
        abort("backward_op: GPU binary operators do not support broadcasting")
    else:
        var res_grad: Tensor[f32, Device.gpu]

        comptime if op == OP.ADD:
            res_grad = rebind[Tensor[f32, Device.gpu]](ug).copy()
        elif op == OP.SUB:
            comptime if tensor_id == 0:
                res_grad = rebind[Tensor[f32, Device.gpu]](ug).copy()
            else:
                res_grad = gpu_sub_backward_t2(rebind[Tensor[f32, Device.gpu]](ug))
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
        elif op == OP.DIV:
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
        else:
            abort("backward_op: binary operator " + String(op) + " is not supported on GPU")

        gpu_accumulate_grad(rebind[Tensor[f32, Device.gpu]](grad), res_grad)


def _backward_op_cpu[
    tensor_id: Int,
    op: OP,
    ug_shape: TensorShape,
    t1_shape: TensorShape,
    t2_shape: TensorShape,
    attributes: AttributeVector,
](ug: Tensor[f32], t1: Tensor[f32], t2: Tensor[f32], mut grad: Tensor[f32],):
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
](
    ug: Tensor[f32, device],
    t1: Tensor[f32, device],
    t2: Tensor[f32, device],
    t3: Tensor[f32, device],
    mut grad: Tensor[f32, device],
):
    """
    Backward pass for ternary operators (CPU-only; every ternary op —
    CONV2D/FMA/BATCHNORM2D — is out of scope for the GPU port).
    """
    comptime assert device.id == Device.cpu.id, (
        "backward_op: ternary operators are CPU-only"
    )
    _backward_op_cpu[
        tensor_id, op, ug_shape, t1_shape, t2_shape, t3_shape, attributes
    ](
        rebind[Tensor[f32, Device.cpu]](ug),
        rebind[Tensor[f32, Device.cpu]](t1),
        rebind[Tensor[f32, Device.cpu]](t2),
        rebind[Tensor[f32, Device.cpu]](t3),
        rebind[Tensor[f32, Device.cpu]](grad),
    )


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
):
    """
    Backward pass for ternary operators (CPU implementations).
    """
    var res_grad: Tensor[f32]

    comptime if op == OP.CONV2D:
        res_grad = CONV2D.backward[
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
    comptime assert device.id == Device.cpu.id, (
        "backward_op: dynamic operators (CONCAT/SPLIT) are CPU-only"
    )
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

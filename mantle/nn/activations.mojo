# ===----------------------------------------------------------------------=== #
# Mantle: Activations
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Activations (mantle.nn.activations)
------------------------------------------------
Activation functions (ReLU, LeakyReLU, Sigmoid, Tanh, Softmax) with and without
`Layer` wrappers.
"""
from mantle import f32
from mantle.core.tensor import Tensor, TensorShape
from mantle.autograd.graph import Graph
from mantle.autograd.symbol import Symbol
from mantle.autograd.ops import OP
from mantle.autograd.attributes import Attribute, AttributeVector
from mantle.nn.module import Expr, Layer, Module


# ===----------------------------------------------------------------------===#
# ReLU
# ===----------------------------------------------------------------------===#


def ReLU(mut g: Graph, input: Symbol) -> Symbol:
    return g.op(OP.RELU, input)


def ReLU() -> ReLULayer:
    """Create a ReLU layer for ``Sequential`` or a reflected module."""
    return ReLULayer()


@fieldwise_init
struct ReLULayer(Copyable, Layer, Module, Movable):
    """
    `Layer`-conforming wrapper around `ReLU`, for use in a reflection-based
    Module struct.
    """

    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        return ReLU(g, input)

    def forward(mut self, input: Expr) -> Expr:
        return input.relu()

    def __call__(mut self, input: Expr) -> Expr:
        return self.forward(input)


# ===----------------------------------------------------------------------===#
# LeakyReLU
# ===----------------------------------------------------------------------===#


def LeakyReLU(
    mut g: Graph, input: Symbol, negative_slope: Scalar[f32]
) -> Symbol:
    return g.op(
        OP.LEAKYRELU,
        input,
        attributes=AttributeVector(Attribute("negative_slope", negative_slope)),
    )


def LeakyReLU(negative_slope: Scalar[f32] = 0.01) -> LeakyReLULayer:
    """Create a leaky-ReLU layer for ``Sequential`` or a custom module."""
    return LeakyReLULayer(negative_slope)


struct LeakyReLULayer(Copyable, Layer, Module, Movable):
    var negative_slope: Scalar[f32]

    def __init__(out self, negative_slope: Scalar[f32] = 0.01):
        self.negative_slope = negative_slope

    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        return LeakyReLU(g, input, self.negative_slope)

    def forward(mut self, input: Expr) -> Expr:
        return Expr(
            input.graph,
            LeakyReLU(input.graph[], input.symbol, self.negative_slope),
        )

    def __call__(mut self, input: Expr) -> Expr:
        return self.forward(input)


# ===----------------------------------------------------------------------===#
# Sigmoid
# ===----------------------------------------------------------------------===#


def Sigmoid(mut g: Graph, input: Symbol) -> Symbol:
    return g.op(OP.SIGMOID, input)


def Sigmoid() -> SigmoidLayer:
    """Create a sigmoid layer for ``Sequential`` or a custom module."""
    return SigmoidLayer()


@fieldwise_init
struct SigmoidLayer(Copyable, Layer, Module, Movable):
    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        return Sigmoid(g, input)

    def forward(mut self, input: Expr) -> Expr:
        return Expr(input.graph, Sigmoid(input.graph[], input.symbol))

    def __call__(mut self, input: Expr) -> Expr:
        return self.forward(input)


# ===----------------------------------------------------------------------===#
# Tanh
# ===----------------------------------------------------------------------===#


def Tanh(mut g: Graph, input: Symbol) -> Symbol:
    return g.op(OP.TANH, input)


def Tanh() -> TanhLayer:
    """Create a tanh layer for ``Sequential`` or a custom module."""
    return TanhLayer()


@fieldwise_init
struct TanhLayer(Copyable, Layer, Module, Movable):
    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        return Tanh(g, input)

    def forward(mut self, input: Expr) -> Expr:
        return Expr(input.graph, Tanh(input.graph[], input.symbol))

    def __call__(mut self, input: Expr) -> Expr:
        return self.forward(input)


# ===----------------------------------------------------------------------===#
# GELU
# ===----------------------------------------------------------------------===#


def GELU(mut g: Graph, input: Symbol) -> Symbol:
    return g.op(OP.GELU, input)


def GELU() -> GELULayer:
    """Create a GELU layer for ``Sequential`` or a custom module."""
    return GELULayer()


@fieldwise_init
struct GELULayer(Copyable, Layer, Module, Movable):
    """
    `Layer`-conforming wrapper around `GELU`, for use in a reflection-based
    Module struct.
    """

    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        return GELU(g, input)

    def forward(mut self, input: Expr) -> Expr:
        return Expr(input.graph, GELU(input.graph[], input.symbol))

    def __call__(mut self, input: Expr) -> Expr:
        return self.forward(input)


# ===----------------------------------------------------------------------===#
# Softmax
# ===----------------------------------------------------------------------===#


def Softmax(mut g: Graph, input: Symbol, axis: Int) -> Symbol:
    # softmax: exp(x_i) / sum(exp(x_j))
    # stable softmax: exp(x_i - max(x_j)) / sum(exp(x_j - max(x_j)))

    var max_values = g.op(
        OP.MAX, input, attributes=AttributeVector(Attribute("axis", axis))
    )
    var input_minus_max = g.op(OP.SUB, input, max_values)
    var exp_values = g.op(OP.EXP, input_minus_max)
    var sum_values = g.op(
        OP.SUM, exp_values, attributes=AttributeVector(Attribute("axis", axis))
    )

    return g.op(OP.DIV, exp_values, sum_values)


def Softmax(axis: Int = 1) -> SoftmaxLayer:
    """Create a softmax layer for ``Sequential`` or a custom module."""
    return SoftmaxLayer(axis)


@fieldwise_init
struct SoftmaxLayer(Copyable, Layer, Module, Movable):
    """
    `Layer`-conforming wrapper around `Softmax`, for use in a reflection-based
    Module struct or `Sequential`.
    """

    var axis: Int

    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        return Softmax(g, input, self.axis)

    def forward(mut self, input: Expr) -> Expr:
        return Expr(input.graph, Softmax(input.graph[], input.symbol, self.axis))

    def __call__(mut self, input: Expr) -> Expr:
        return self.forward(input)


# ===----------------------------------------------------------------------===#
# LogSoftmax
# ===----------------------------------------------------------------------===#


def LogSoftmax(mut g: Graph, input: Symbol, axis: Int) -> Symbol:
    # stable logsoftmax: log(exp(x_i - max(x_j)) / sum(exp(x_j - max(x_j))))
    # stable logsoftmax: x_i - max(x_j) - log(sum(exp(x_j - max(x_j))))

    var max_values = g.op(
        OP.MAX, input, attributes=AttributeVector(Attribute("axis", axis))
    )
    var input_minus_max = g.op(OP.SUB, input, max_values)
    var exp_values = g.op(OP.EXP, input_minus_max)
    var sum_values = g.op(
        OP.SUM, exp_values, attributes=AttributeVector(Attribute("axis", axis))
    )
    var log_values = g.op(OP.LOG, sum_values)

    return g.op(OP.SUB, input_minus_max, log_values)


def LogSoftmax(axis: Int = 1) -> LogSoftmaxLayer:
    """Create a log-softmax layer for ``Sequential`` or a custom module."""
    return LogSoftmaxLayer(axis)


struct LogSoftmaxLayer(Copyable, Layer, Module, Movable):
    var axis: Int

    def __init__(out self, axis: Int = 1):
        self.axis = axis

    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        return LogSoftmax(g, input, self.axis)

    def forward(mut self, input: Expr) -> Expr:
        return Expr(
            input.graph, LogSoftmax(input.graph[], input.symbol, self.axis)
        )

    def __call__(mut self, input: Expr) -> Expr:
        return self.forward(input)

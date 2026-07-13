# ===----------------------------------------------------------------------=== #
# Mantle: Layer Normalization
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""LayerNorm (mantle.nn.layers.layernorm)
------------------------------------------------
Layer normalization over the last axis, composed from existing ops
(MEAN, SUB, POW, SQRT, DIV, MUL, ADD). Learnable gamma/beta of shape
`(normalized_shape,)`.
"""
from std.reflection import reflect_fn

from mantle import f32
from mantle.autograd.graph import Graph
from mantle.autograd.symbol import Symbol
from mantle.autograd.ops import OP
from mantle.autograd.attributes import Attribute, AttributeVector
from mantle.autograd.params import Param
from mantle.core.tensor import Tensor, TensorShape
from mantle.nn.module import Layer


# ===----------------------------------------------------------------------===#
# LayerNorm (functional)
# ===----------------------------------------------------------------------===#


def LayerNorm(
    mut g: Graph,
    inputs: Symbol,
    normalized_shape: Int,
    epsilon: Float32 = 1e-5,
) -> Symbol:
    """
    Normalizes over the last axis of `inputs` (size `normalized_shape`) to
    zero mean / unit variance, then applies a learnable affine transform:
    `gamma * (x - mean) / sqrt(var + eps) + beta`.

    Learnable gamma (scale) and beta (shift) are `(normalized_shape,)`
    parameters initialized to 1 and 0 respectively.
    """
    var before = len(g.nodes)
    var axis = inputs.shape.rank() - 1

    var gamma = g.param(
        TensorShape(normalized_shape),
        init=Param("constant", Scalar[f32](1.0), Scalar[f32](0.0)),
    )
    var beta = g.param(
        TensorShape(normalized_shape),
        init=Param("constant", Scalar[f32](0.0), Scalar[f32](0.0)),
    )

    var mean = g.op(
        OP.MEAN, inputs, attributes=AttributeVector(Attribute("axis", axis))
    )
    var diff = g.op(OP.SUB, inputs, mean)
    var sq_diff = g.op(OP.POW, diff, 2.0)
    var var_ = g.op(
        OP.MEAN, sq_diff, attributes=AttributeVector(Attribute("axis", axis))
    )
    var std = g.op(OP.SQRT, g.op(OP.ADD, var_, Float64(epsilon)))
    var normed = g.op(OP.DIV, diff, std)
    var scaled = g.op(OP.MUL, normed, gamma)
    var res = g.op(OP.ADD, scaled, beta)

    g.set_scope_from(before, reflect_fn[LayerNorm].display_name())
    return res


# ===----------------------------------------------------------------------===#
# LayerNormLayer
# ===----------------------------------------------------------------------===#


struct LayerNormLayer(Copyable, Layer, Movable):
    """
    `Layer`-conforming wrapper around `LayerNorm`, for use in a
    reflection-based Module struct.
    """

    var normalized_shape: Int
    var epsilon: Float32

    def __init__(out self, normalized_shape: Int, epsilon: Float32 = 1e-5):
        self.normalized_shape = normalized_shape
        self.epsilon = epsilon

    def __init__(out self, *, copy: Self):
        self.normalized_shape = copy.normalized_shape
        self.epsilon = copy.epsilon

    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        return LayerNorm(g, input, self.normalized_shape, self.epsilon)

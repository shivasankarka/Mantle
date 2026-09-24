# ===----------------------------------------------------------------------=== #
# Mantle: Layer Normalization
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""LayerNorm (mantle.nn.layers.layernorm)
------------------------------------------------
Fused layer normalization over the last axis with learnable gamma/beta of
shape `(normalized_shape,)`.
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
    var gamma = g.param(
        TensorShape(normalized_shape),
        init=Param("constant", Scalar[f32](1.0), Scalar[f32](0.0)),
    )
    var beta = g.param(
        TensorShape(normalized_shape),
        init=Param("constant", Scalar[f32](0.0), Scalar[f32](0.0)),
    )

    # A composite LayerNorm previously emitted eight graph nodes and kept all
    # their activation/gradient buffers alive for backward. The fused op uses
    # a row-wise reduction and stores only the original input plus output.
    var res = g.op(
        OP.LAYERNORM,
        inputs,
        gamma,
        beta,
        attributes=AttributeVector(Attribute("epsilon", Scalar[f32](epsilon))),
    )

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

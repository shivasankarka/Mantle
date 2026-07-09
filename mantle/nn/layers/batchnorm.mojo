# ===----------------------------------------------------------------------=== #
# Mantle: Batch Normalization Layer
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""BatchNorm (mantle.nn.layers.batchnorm)
------------------------------------------------
Batch Normalization layer for 2D [N, C] and 4D [N, C, H, W] inputs.
Trainable scale (gamma) and shift (beta) parameters per channel.
"""
from mantle import f32
from mantle.autograd.graph import Graph
from mantle.autograd.symbol import Symbol
from mantle.autograd.ops import OP
from mantle.autograd.attributes import Attribute, AttributeVector
from mantle.autograd.params import Param
from mantle.core.tensor import Tensor, TensorShape
from mantle.nn.module import Layer


# ===----------------------------------------------------------------------===#
# BatchNorm2d (functional)
# ===----------------------------------------------------------------------===#

def BatchNorm2d(
    mut g: Graph,
    inputs: Symbol,
    epsilon: Float32 = 1e-5,
) -> Symbol:
    """
    Batch Normalization over [N, C] or [N, C, H, W] inputs.

    Learnable gamma (scale) and beta (shift) are [C]-shaped parameters
    initialized to 1 and 0 respectively.

    Args:
        g: The computation graph.
        inputs: Input symbol of shape [N, C] or [N, C, H, W].
        epsilon: Small value added to variance for numerical stability.
    """
    var C = inputs.shape[1]
    var gamma = g.param(
        TensorShape(C), init=Param("constant", Scalar[f32](1.0), Scalar[f32](0.0))
    )
    var beta = g.param(
        TensorShape(C), init=Param("constant", Scalar[f32](0.0), Scalar[f32](0.0))
    )

    return g.op(
        OP.BATCHNORM2D,
        inputs,
        gamma,
        beta,
        attributes=AttributeVector(
            Attribute("epsilon", epsilon),
        ),
    )


# ===----------------------------------------------------------------------===#
# BatchNorm2dLayer
# ===----------------------------------------------------------------------===#

struct BatchNorm2dLayer(Layer, Copyable, Movable):
    """
    `Layer`-conforming wrapper around `BatchNorm2d`, for use in a
    reflection-based Module struct.
    """

    var epsilon: Float32

    def __init__(out self, epsilon: Float32 = 1e-5):
        self.epsilon = epsilon

    def __init__(out self, *, copy: Self):
        self.epsilon = copy.epsilon

    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        return BatchNorm2d(g, input, self.epsilon)

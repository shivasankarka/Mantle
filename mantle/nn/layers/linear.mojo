# ===----------------------------------------------------------------------=== #
# Mantle: Linear Layer
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Linear (mantle.nn.layers.linear)
------------------------------------------------
Fully connected (dense) layer with uniform initialization.
"""
from mantle import f32
from mantle.core.tensor import Tensor, TensorShape
from mantle.autograd.graph import Graph
from mantle.autograd.symbol import Symbol
from mantle.autograd.ops import OP
from mantle.core.math_util import q_sqrt
from mantle.autograd.params import Param
from mantle.nn.module import Layer


# ===----------------------------------------------------------------------===#
# Linear (functional)
# ===----------------------------------------------------------------------===#


def Linear(
    mut g: Graph,
    inputs: Symbol,
    n_outputs: Int,
    output_scale: Float32 = 1.0,
) -> Symbol:
    """
    A fully connected layer. Accepts input of any rank >= 2; the matmul
    contracts the last dimension (e.g. `(B, D)` or `(B, T, D)`).

    `output_scale` shrinks the weight init bound below the usual
    `1/sqrt(fan_in)`. Pass e.g. `1/sqrt(2*num_blocks)` for a residual
    block's *output* projection (attention/FFN's last Linear) in a deep
    pre-norm Transformer -- otherwise the residual stream's variance grows
    with depth and, combined with enough layers/width, blows up to NaN or
    collapses to 0 (this is the same reason nanoGPT scales its `c_proj`
    init by `1/sqrt(2*n_layer)`).
    """

    var fan_in: Scalar[f32] = Scalar[f32](inputs.shape[-1])
    var bound = q_sqrt(fan_in) * output_scale
    var weights = g.param(
        TensorShape(inputs.shape[-1], n_outputs),
        init=Param("random_uniform", -bound, bound)
        # init=Param("random_uniform", 1) # NOTE: mode: fan_out required as weight are defined transposed
    )
    var b = g.param(
        TensorShape(n_outputs), init=Param("random_uniform", -bound, bound)
    )

    return g.op(OP.LINEAR, inputs, weights, b)


# ===----------------------------------------------------------------------===#
# LinearLayer
# ===----------------------------------------------------------------------===#


@fieldwise_init
struct LinearLayer(Copyable, Layer, Movable):
    """
    `Layer`-conforming wrapper around `Linear`, for use in a reflection-based
    Module struct.
    """

    var n_outputs: Int

    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        return Linear(g, input, self.n_outputs)

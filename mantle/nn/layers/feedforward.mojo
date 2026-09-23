# ===----------------------------------------------------------------------=== #
# Mantle: Feed-Forward Block
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""FeedForward (mantle.nn.layers.feedforward)
------------------------------------------------
Position-wise feed-forward block used inside a Transformer block:
Linear(d_ff) -> GELU -> Linear(d_model) -> Dropout.
"""
from std.reflection import reflect_fn

from mantle import f32
from mantle.autograd.graph import Graph
from mantle.autograd.symbol import Symbol
from mantle.nn.module import Layer
from mantle.nn.layers.linear import Linear
from mantle.nn.layers.dropout import Dropout
from mantle.nn.activations import GELU


# ===----------------------------------------------------------------------===#
# FeedForward (functional)
# ===----------------------------------------------------------------------===#


def FeedForward(
    mut g: Graph,
    inputs: Symbol,
    d_ff: Int,
    dropout_p: Float32 = 0.0,
) -> Symbol:
    """
    Position-wise feed-forward block: expands to `d_ff`, applies GELU,
    projects back to `inputs`' last dim, then dropout.
    """
    var before = len(g.nodes)
    var d_model = inputs.shape[-1]

    var hidden = Linear(g, inputs, d_ff)
    var activated = GELU(g, hidden)
    var res = Linear(g, activated, d_model)
    if dropout_p > 0.0:
        res = Dropout(g, res, dropout_p)

    g.set_scope_from(before, reflect_fn[FeedForward].display_name())
    return res


# ===----------------------------------------------------------------------===#
# FeedForwardLayer
# ===----------------------------------------------------------------------===#


struct FeedForwardLayer(Copyable, Layer, Movable):
    """
    `Layer`-conforming wrapper around `FeedForward`, for use in a
    reflection-based Module struct.
    """

    var d_ff: Int
    var dropout_p: Float32

    def __init__(out self, d_ff: Int, dropout_p: Float32 = 0.0):
        self.d_ff = d_ff
        self.dropout_p = dropout_p

    def __init__(out self, *, copy: Self):
        self.d_ff = copy.d_ff
        self.dropout_p = copy.dropout_p

    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        return FeedForward(g, input, self.d_ff, self.dropout_p)

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
from std.math import sqrt

from mantle import f32
from mantle.autograd.graph import Graph
from mantle.autograd.symbol import Symbol
from mantle.nn.module import Expr, Layer, Module
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
    num_blocks: Int = 1,
) -> Symbol:
    """
    Position-wise feed-forward block: expands to `d_ff`, applies GELU,
    projects back to `inputs`' last dim, then dropout.

    `num_blocks` scales down the down-projection's init (see `Linear`'s
    `output_scale`) to keep deep pre-norm stacks numerically stable. Leave
    at 1 for a standalone feed-forward block.
    """
    var before = len(g.nodes)
    var d_model = inputs.shape[-1]

    var hidden = Linear(g, inputs, d_ff)
    var activated = GELU(g, hidden)
    var output_scale = 1.0 / sqrt(2.0 * Float64(num_blocks))
    var res = Linear(g, activated, d_model, Float32(output_scale))
    if dropout_p > 0.0:
        res = Dropout(g, res, dropout_p)

    g.set_scope_from(before, "FeedForward")
    return res


def FeedForward(
    d_ff: Int, dropout_p: Float32 = 0.0, num_blocks: Int = 1
) -> FeedForwardLayer:
    """Create a feed-forward layer for ``Sequential`` or a custom module."""
    return FeedForwardLayer(d_ff, dropout_p, num_blocks)


# ===----------------------------------------------------------------------===#
# FeedForwardLayer
# ===----------------------------------------------------------------------===#


struct FeedForwardLayer(Copyable, Layer, Module, Movable):
    """
    `Layer`-conforming wrapper around `FeedForward`, for use in a
    reflection-based Module struct.
    """

    var d_ff: Int
    var dropout_p: Float32
    var num_blocks: Int

    def __init__(
        out self, d_ff: Int, dropout_p: Float32 = 0.0, num_blocks: Int = 1
    ):
        self.d_ff = d_ff
        self.dropout_p = dropout_p
        self.num_blocks = num_blocks

    def __init__(out self, *, copy: Self):
        self.d_ff = copy.d_ff
        self.dropout_p = copy.dropout_p
        self.num_blocks = copy.num_blocks

    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        return FeedForward(
            g, input, self.d_ff, self.dropout_p, self.num_blocks
        )

    def forward(mut self, input: Expr) -> Expr:
        return Expr(
            input.graph,
            FeedForward(
                input.graph[],
                input.symbol,
                self.d_ff,
                self.dropout_p,
                self.num_blocks,
            ),
        )

    def __call__(mut self, input: Expr) -> Expr:
        return self.forward(input)

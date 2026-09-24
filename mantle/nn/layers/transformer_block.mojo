# ===----------------------------------------------------------------------=== #
# Mantle: Transformer Block
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""TransformerBlock (mantle.nn.layers.transformer_block)
------------------------------------------------
Pre-norm Transformer block: x + MHA(LN(x)), then x + FF(LN(x)).
"""
from std.reflection import reflect_fn

from mantle import f32
from mantle.autograd.graph import Graph
from mantle.autograd.symbol import Symbol
from mantle.autograd.ops import OP
from mantle.nn.module import Layer
from mantle.nn.layers.layernorm import LayerNorm
from mantle.nn.layers.attention import MultiHeadAttention
from mantle.nn.layers.feedforward import FeedForward


# ===----------------------------------------------------------------------===#
# TransformerBlock (functional)
# ===----------------------------------------------------------------------===#


def TransformerBlock(
    mut g: Graph,
    inputs: Symbol,
    num_heads: Int,
    d_ff: Int,
    dropout_p: Float32 = 0.0,
    causal: Bool = True,
    num_blocks: Int = 1,
) -> Symbol:
    """
    Pre-norm Transformer block over `inputs` of shape `(B, T, D)`:
    `x = x + MHA(LN(x))`, then `x = x + FF(LN(x))`.

    `num_blocks` is the total depth of the stack this block belongs to --
    pass the real block count when stacking several of these (see
    `MultiHeadAttention`/`FeedForward`'s `num_blocks` for why: without it,
    deep stacks' residual stream variance grows unboundedly and blows up
    to NaN).
    """
    var before = len(g.nodes)
    var d_model = inputs.shape[-1]

    var normed1 = LayerNorm(g, inputs, d_model)
    var attn = MultiHeadAttention(
        g, normed1, num_heads, dropout_p, causal, num_blocks
    )
    var res1 = g.op(OP.ADD, inputs, attn)

    var normed2 = LayerNorm(g, res1, d_model)
    var ff = FeedForward(g, normed2, d_ff, dropout_p, num_blocks)
    var res2 = g.op(OP.ADD, res1, ff)

    g.set_scope_from(before, reflect_fn[TransformerBlock].display_name())
    return res2


# ===----------------------------------------------------------------------===#
# TransformerBlockLayer
# ===----------------------------------------------------------------------===#


struct TransformerBlockLayer(Copyable, Layer, Movable):
    """
    `Layer`-conforming wrapper around `TransformerBlock`, for use in a
    reflection-based Module struct.
    """

    var num_heads: Int
    var d_ff: Int
    var dropout_p: Float32
    var causal: Bool
    var num_blocks: Int

    def __init__(
        out self,
        num_heads: Int,
        d_ff: Int,
        dropout_p: Float32 = 0.0,
        causal: Bool = True,
        num_blocks: Int = 1,
    ):
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_p = dropout_p
        self.causal = causal
        self.num_blocks = num_blocks

    def __init__(out self, *, copy: Self):
        self.num_heads = copy.num_heads
        self.d_ff = copy.d_ff
        self.dropout_p = copy.dropout_p
        self.causal = copy.causal
        self.num_blocks = copy.num_blocks

    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        return TransformerBlock(
            g,
            input,
            self.num_heads,
            self.d_ff,
            self.dropout_p,
            self.causal,
            self.num_blocks,
        )

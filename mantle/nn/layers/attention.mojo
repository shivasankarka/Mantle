# ===----------------------------------------------------------------------=== #
# Mantle: Multi-Head Attention
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Attention (mantle.nn.layers.attention)
------------------------------------------------
Scaled dot-product multi-head self-attention, composed from existing ops
(DOT, TRANSPOSE, RESHAPE, Softmax, Dropout). Optional causal masking for
autoregressive (GPT-style) models.
"""
from std.math import sqrt
from std.reflection import reflect_fn

from mantle import f32
from mantle.autograd.graph import Graph
from mantle.autograd.symbol import Symbol
from mantle.autograd.ops import OP
from mantle.autograd.attributes import Attribute, AttributeVector
from mantle.core.tensor import Tensor, TensorShape
from mantle.nn.module import Layer
from mantle.nn.layers.linear import Linear
from mantle.nn.layers.dropout import Dropout
from mantle.nn.activations import Softmax


# ===----------------------------------------------------------------------===#
# Causal mask
# ===----------------------------------------------------------------------===#


def causal_mask(mut g: Graph, seq_len: Int) -> Symbol:
    """
    Builds a `(seq_len, seq_len)` additive mask: 0 on/below the diagonal,
    -1e9 above it. Add to attention scores before Softmax to prevent
    attending to future positions.
    """
    var data = List[Scalar[f32]]()
    data.reserve(seq_len * seq_len)
    for i in range(seq_len):
        for j in range(seq_len):
            data.append(Float32(0.0) if j <= i else -1e9)
    return g.constant(TensorShape(seq_len, seq_len), data)


# ===----------------------------------------------------------------------===#
# MultiHeadAttention (functional)
# ===----------------------------------------------------------------------===#


def MultiHeadAttention(
    mut g: Graph,
    inputs: Symbol,
    num_heads: Int,
    dropout_p: Float32 = 0.0,
    causal: Bool = False,
) -> Symbol:
    """
    Scaled dot-product multi-head self-attention over `inputs` of shape
    `(B, T, D)`. `D` must be divisible by `num_heads`.
    """
    var before = len(g.nodes)

    var batch = inputs.shape[0]
    var seq_len = inputs.shape[1]
    var d_model = inputs.shape[2]
    var head_dim = d_model // num_heads

    var q = Linear(g, inputs, d_model)
    var k = Linear(g, inputs, d_model)
    var v = Linear(g, inputs, d_model)

    # (B, T, D) -> (B, T, H, d) -> (B, H, T, d)
    def split_heads(
        t: Symbol,
    ) {mut g, imm batch, imm seq_len, imm num_heads, imm head_dim} -> Symbol:
        var reshaped = g.op(
            OP.RESHAPE,
            t,
            attributes=AttributeVector(
                Attribute(
                    "shape", TensorShape(batch, seq_len, num_heads, head_dim)
                )
            ),
        )
        return g.op(
            OP.TRANSPOSE,
            reshaped,
            attributes=AttributeVector(
                Attribute("axes", TensorShape(0, 2, 1, 3))
            ),
        )

    var qh = split_heads(q)
    var kh = split_heads(k)
    var vh = split_heads(v)

    # scores = q @ k^T / sqrt(d): (B, H, T, d) @ (B, H, d, T) -> (B, H, T, T)
    var kh_t = g.op(
        OP.TRANSPOSE,
        kh,
        attributes=AttributeVector(Attribute("axes", TensorShape(0, 1, 3, 2))),
    )
    var scale = 1.0 / sqrt(Float64(head_dim))
    var scores = g.op(OP.MUL, g.op(OP.DOT, qh, kh_t), scale)

    if causal:
        var mask = causal_mask(g, seq_len)
        scores = g.op(OP.ADD, scores, mask)

    var probs = Softmax(g, scores, axis=3)
    if dropout_p > 0.0:
        probs = Dropout(g, probs, dropout_p)

    # (B, H, T, T) @ (B, H, T, d) -> (B, H, T, d)
    var attn_out = g.op(OP.DOT, probs, vh)

    # (B, H, T, d) -> (B, T, H, d) -> (B, T, D)
    var merged = g.op(
        OP.TRANSPOSE,
        attn_out,
        attributes=AttributeVector(Attribute("axes", TensorShape(0, 2, 1, 3))),
    )
    var merged_flat = g.op(
        OP.RESHAPE,
        merged,
        attributes=AttributeVector(
            Attribute("shape", TensorShape(batch, seq_len, d_model))
        ),
    )

    var res = Linear(g, merged_flat, d_model)

    g.set_scope_from(before, reflect_fn[MultiHeadAttention].display_name())
    return res


# ===----------------------------------------------------------------------===#
# MultiHeadAttentionLayer
# ===----------------------------------------------------------------------===#


struct MultiHeadAttentionLayer(Copyable, Layer, Movable):
    """
    `Layer`-conforming wrapper around `MultiHeadAttention`, for use in a
    reflection-based Module struct.
    """

    var num_heads: Int
    var dropout_p: Float32
    var causal: Bool

    def __init__(
        out self,
        num_heads: Int,
        dropout_p: Float32 = 0.0,
        causal: Bool = False,
    ):
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.causal = causal

    def __init__(out self, *, copy: Self):
        self.num_heads = copy.num_heads
        self.dropout_p = copy.dropout_p
        self.causal = copy.causal

    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        return MultiHeadAttention(
            g, input, self.num_heads, self.dropout_p, self.causal
        )

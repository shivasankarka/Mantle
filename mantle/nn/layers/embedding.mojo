# ===----------------------------------------------------------------------=== #
# Mantle: Embedding Layer
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Embedding (mantle.nn.layers.embedding)
------------------------------------------------
Token embedding lookup table, backed by OP.GATHER.
"""
from mantle import f32
from mantle.core.tensor import TensorShape
from mantle.autograd.graph import Graph
from mantle.autograd.symbol import Symbol
from mantle.autograd.ops import OP
from mantle.autograd.attributes import Attribute, AttributeVector
from mantle.autograd.params import Param
from mantle.nn.module import Layer


# ===----------------------------------------------------------------------===#
# Embedding (functional)
# ===----------------------------------------------------------------------===#


def Embedding(
    mut g: Graph, indices: Symbol, vocab_size: Int, embed_dim: Int
) -> Symbol:
    """
    Looks up rows of a `(vocab_size, embed_dim)` table by `indices` (any
    shape, holding row ids as f32). Output shape is `indices.shape +
    (embed_dim,)`.
    """
    var table = g.param(
        TensorShape(vocab_size, embed_dim),
        init=Param("random_normal", 0.0, 0.02),
    )
    return g.op(OP.GATHER, table, indices)


# ===----------------------------------------------------------------------===#
# EmbeddingLayer
# ===----------------------------------------------------------------------===#


@fieldwise_init
struct EmbeddingLayer(Copyable, Layer, Movable):
    """
    `Layer`-conforming wrapper around `Embedding`, for use in a
    reflection-based Module struct.
    """

    var vocab_size: Int
    var embed_dim: Int

    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        return Embedding(g, input, self.vocab_size, self.embed_dim)


# ===----------------------------------------------------------------------===#
# PositionalEmbedding
# ===----------------------------------------------------------------------===#


def PositionalEmbedding(
    mut g: Graph, inputs: Symbol, max_seq_len: Int
) -> Symbol:
    """
    Adds a learned `(seq_len, embed_dim)` positional embedding (sliced from
    a `(max_seq_len, embed_dim)` table) to `inputs` of shape
    `(B, seq_len, embed_dim)`. Broadcasts over the batch dim.
    """
    var seq_len = inputs.shape[1]
    var embed_dim = inputs.shape[2]

    var table = g.param(
        TensorShape(max_seq_len, embed_dim),
        init=Param("random_normal", 0.0, 0.02),
    )
    var pos = table if seq_len == max_seq_len else g.op(
        OP.SLICE,
        table,
        attributes=AttributeVector(
            Attribute("starts", TensorShape(0)),
            Attribute("ends", TensorShape(seq_len)),
            Attribute("axes", TensorShape(0)),
        ),
    )
    return g.op(OP.ADD, inputs, pos)

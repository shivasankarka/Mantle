# ===----------------------------------------------------------------------=== #
# Mantle: Dropout Layer
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Dropout (mantle.nn.layers.dropout)
-----------------------------------------------
Dropout regularization layer: randomly zeroes elements with probability p.
Uses a seed-based deterministic mask for consistent forward/backward.
"""
from mantle.autograd.graph import Graph
from mantle.autograd.symbol import Symbol
from mantle.autograd.ops import OP
from mantle.autograd.attributes import Attribute, AttributeVector
from mantle.nn.module import Expr, Layer, Module
from mantle.core.tensor import Tensor, TensorShape


# ===----------------------------------------------------------------------===#
# Dropout (functional)
# ===----------------------------------------------------------------------===#


def Dropout(mut g: Graph, inputs: Symbol, p: Float32, seed: Int = 42) -> Symbol:
    """
    Apply dropout with probability `p`. Uses a deterministic seed for mask
    reproducibility between forward and backward passes.
    """
    return g.op(
        OP.DROPOUT,
        inputs,
        attributes=AttributeVector(
            Attribute("p", p),
            Attribute("seed", seed),
        ),
    )


def Dropout(p: Float32 = 0.5) -> DropoutLayer:
    """Create a dropout layer for ``Sequential`` or a custom module."""
    return DropoutLayer(p)


# ===----------------------------------------------------------------------===#
# DropoutLayer
# ===----------------------------------------------------------------------===#


struct DropoutLayer(Copyable, Layer, Module, Movable):
    """
    `Layer`-conforming wrapper around `Dropout`, for use in a reflection-based
    Module struct.
    """

    var p: Float32
    var seed: Int

    def __init__(out self, p: Float32, seed: Int = 42):
        self.p = p
        self.seed = seed

    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        return Dropout(g, input, self.p, self.seed)

    def forward(mut self, input: Expr) -> Expr:
        return Expr(input.graph, Dropout(input.graph[], input.symbol, self.p, self.seed))

    def __call__(mut self, input: Expr) -> Expr:
        return self.forward(input)

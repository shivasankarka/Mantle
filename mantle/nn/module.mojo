# ===----------------------------------------------------------------------=== #
# Mantle: Module
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Module (mantle.nn.module)
------------------------------------------------
Layer trait, FlattenLayer, graph-building reflection, and Sequential container.
"""
from std.reflection import reflect

from mantle.autograd.graph import Graph
from mantle.autograd.symbol import Symbol
from mantle.autograd.ops import OP
from mantle.core.tensor import TensorShape
from mantle.autograd.attributes import Attribute, AttributeVector


# ===----------------------------------------------------------------------===#
# Layer Trait
# ===----------------------------------------------------------------------===#

trait Layer:
    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        ...


# ===----------------------------------------------------------------------===#
# FlattenLayer
# ===----------------------------------------------------------------------===#

@fieldwise_init
struct FlattenLayer(Layer, Copyable, Movable):
    """
    Flattens every dim except the batch dim (dim 0). Equivalent to PyTorch's
    `x.view(x.size(0), -1)`, e.g. after Conv2d/MaxPool2d before a Linear layer.
    """

    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        var batch = input.shape[0]
        var rest = input.shape.num_elements() // batch
        return g.op(
            OP.RESHAPE,
            input,
            attributes=AttributeVector(
                Attribute("shape", TensorShape(batch, rest))
            ),
        )


# ===----------------------------------------------------------------------===#
# build_graph
# ===----------------------------------------------------------------------===#

def build_graph[T: AnyType](mut layers: T, mut g: Graph, input: Symbol) -> Symbol:
    """
    Reflects over `layers`' fields in declaration order, chaining every
    `Layer`-conforming field's `forward(g, x) -> x` to build up a Graph.

    Non-Layer fields (e.g. plain config values) are skipped.

    Each Layer's ops are tagged with the layer type name as the scope,
    enabling architectural visualization of the model.
    """
    comptime r = reflect[T]
    comptime field_types = r.field_types()

    var x = input
    comptime for idx in range(r.field_count()):
        comptime field_type = field_types[idx]
        comptime if conforms_to(field_type, Layer):
            ref field_val = r.field_ref[idx](layers)
            var before = len(g.nodes)
            comptime type_name = reflect[field_type].base_name()
            x = trait_downcast[Layer](field_val).forward(g, x)
            g.set_scope_from(before, type_name)
    return x


# ===----------------------------------------------------------------------===#
# Sequential
# ===----------------------------------------------------------------------===#

struct Sequential[*Ts: Layer & Movable](Layer, Movable):
    """
    A plain ordered list of heterogeneous `Layer`s, chained in the order
    given to the constructor: `Sequential(LinearLayer(32), ReLULayer())`.
    """

    var layers: Tuple[*Self.Ts]

    def __init__(out self, var *layers: *Self.Ts):
        self.layers = Tuple(*layers^)

    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        var x = input
        comptime for i in range(Self.Ts.__len__()):
            x = self.layers[i].forward(g, x)
        return x

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
from std.memory.unsafe_pointer import Pointer

from mantle.autograd.graph import Graph
from mantle.autograd.symbol import Symbol
from mantle.autograd.ops import OP
from mantle.core.tensor import TensorShape
from mantle.autograd.attributes import Attribute, AttributeVector


# ===----------------------------------------------------------------------===#
# Layer Trait
# ===----------------------------------------------------------------------===#


struct Expr(Copyable, Movable):
    """An opaque symbolic value bound to one graph-building context.

    ``Expr`` exists only while Mantle is constructing a static graph.  It
    carries the builder internally, letting user-defined modules compose
    symbolic values without accepting a ``Graph`` argument themselves.
    """

    var graph: Pointer[Graph, MutUntrackedOrigin]
    var symbol: Symbol

    def __init__(
        out self,
        ref[MutAnyOrigin] graph: Graph,
        symbol: Symbol,
    ):
        self.graph = Pointer(to=graph).unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        self.symbol = symbol

    def __init__(
        out self,
        graph: Pointer[Graph, MutUntrackedOrigin],
        symbol: Symbol,
    ):
        self.graph = graph
        self.symbol = symbol

    def relu(self) -> Self:
        return Expr(self.graph, self.graph[].op(OP.RELU, self.symbol))

    def __add__(self, other: Self) -> Self:
        return Expr(
            self.graph, self.graph[].op(OP.ADD, self.symbol, other.symbol)
        )


trait Module:
    """A custom model block built from opaque graph expressions."""

    def forward(mut self, input: Expr) -> Expr:
        ...


trait Layer:
    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        ...


# ===----------------------------------------------------------------------===#
# FlattenLayer
# ===----------------------------------------------------------------------===#


@fieldwise_init
struct FlattenLayer(Copyable, Layer, Module, Movable):
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

    def forward(mut self, input: Expr) -> Expr:
        return Expr(
            input.graph,
            self.forward(input.graph[], input.symbol),
        )

    def __call__(mut self, input: Expr) -> Expr:
        return self.forward(input)


def Flatten() -> FlattenLayer:
    """Create a flatten layer for ``Sequential`` or a reflected module."""
    return FlattenLayer()


# ===----------------------------------------------------------------------===#
# build_graph
# ===----------------------------------------------------------------------===#


def build_graph[
    T: AnyType
](mut layers: T, mut g: Graph, input: Symbol) -> Symbol:
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
            # x = trait_downcast[Layer](field_val).forward(g, x)
            comptime if conforms_to(type_of(field_val), Layer):
                x = field_val.forward(g, x)
            else:
                continue
            g.set_scope_from(before, type_name)
    return x


def build_module_graph[T: Module](
    mut module: T, input_shape: TensorShape
) -> Graph:
    """Build an inference graph from a custom ``Module`` definition."""
    var g = Graph()
    var input = Expr(g, g.input(input_shape))
    var output = module.forward(input)
    g.out(output.symbol)
    return g^


# ===----------------------------------------------------------------------===#
# Sequential
# ===----------------------------------------------------------------------===#


struct Sequential[*Ts: Layer & Movable & Deinitable](Layer, Movable):
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

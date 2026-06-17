from std.reflection import reflect

from src.autograd.graph import Graph
from src.autograd.symbol import Symbol
from src.autograd.ops import OP
from src.nn.tensor import TensorShape
from src.autograd.attributes import Attribute, AttributeVector


trait Layer:
    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        ...


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


def build_graph[T: AnyType](mut layers: T, mut g: Graph, input: Symbol) -> Symbol:
    """
    Reflects over `layers`' fields in declaration order, chaining every
    `Layer`-conforming field's `forward(g, x) -> x` to build up a Graph.

    Non-Layer fields (e.g. plain config values) are skipped.
    """
    comptime r = reflect[T]
    comptime field_types = r.field_types()

    var x = input
    comptime for idx in range(r.field_count()):
        comptime field_type = field_types[idx]
        comptime if conforms_to(field_type, Layer):
            ref field_val = r.field_ref[idx](layers)
            x = trait_downcast[Layer](field_val).forward(g, x)
    return x


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

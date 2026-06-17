from mantle.nn.tensor import Tensor, TensorShape
from mantle.autograd.graph import Graph
from mantle.autograd.symbol import Symbol
from mantle.autograd.ops import OP
from mantle.utils import q_sqrt
from mantle.autograd.params import Param
from mantle.nn.module import Layer


def Linear(
    mut g: Graph,
    inputs: Symbol,
    n_outputs: Int,
) -> Symbol:
    """
    A fully connected layer.
    """

    var fan_in: Scalar[f32] = Scalar[f32](inputs.shape[1])
    var bound = q_sqrt(fan_in)
    var weights = g.param(
        TensorShape(inputs.shape[1], n_outputs),
        init=Param("random_uniform", -bound, bound)
        # init=Param("random_uniform", 1) # NOTE: mode: fan_out required as weight are defined transposed
    )
    var b = g.param(
        TensorShape(n_outputs), init=Param("random_uniform", -bound, bound)
    )

    var res = g.op(OP.DOT, inputs, weights)
    return g.op(OP.ADD, res, b)


@fieldwise_init
struct LinearLayer(Layer, Copyable, Movable):
    """
    `Layer`-conforming wrapper around `Linear`, for use in a reflection-based
    Module struct.
    """

    var n_outputs: Int

    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        return Linear(g, input, self.n_outputs)

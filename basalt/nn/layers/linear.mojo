from basalt import Tensor, TensorShape
from basalt import Graph, Symbol, OP
from basalt.utils import q_sqrt
from basalt.autograd.params import Param
from basalt.nn.module import Layer


def Linear(
    mut g: Graph,
    inputs: Symbol,
    n_outputs: Int,
) -> Symbol:
    """
    A fully connected layer.
    """

    var fan_in: Scalar[dtype] = Scalar[dtype](inputs.shape[1])
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

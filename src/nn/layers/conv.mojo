from src.autograd.graph import Graph
from src.autograd.symbol import Symbol
from src.autograd.ops import OP
from src.nn.tensor import Tensor, TensorShape
from src.utils import q_sqrt
from src.autograd.params import Param
from src.autograd.attributes import AttributeVector, Attribute
from src.nn.module import Layer

from std.utils.index import IndexList


def Conv2d(
    mut g: Graph,
    inputs: Symbol,
    out_channels: Int,
    kernel_size: IndexList[2],
    padding: IndexList[2] = IndexList[2](0, 0),
    stride: IndexList[2] = IndexList[2](1, 1),
    dilation: IndexList[2] = IndexList[2](1, 1),
) -> Symbol:
    """
    A 2D Convolution Layer.

    Parameters
        inputs.shape     [batch, in_channels, iX, iY]
        kernel.shape     [out_channels, in_channels, kX, kY] (or weights)
        bias.shape       [out_channels].
        output.shape     [batch, out_channels, oX, oY].
    """

    var in_channels: Int = inputs.shape[1]
    var fan_in: Scalar[f32] = Scalar[f32](
        in_channels * kernel_size[0] * kernel_size[1]
    )
    var bound = q_sqrt(fan_in)
    var weights = g.param(
        TensorShape(out_channels, in_channels, kernel_size[0], kernel_size[1]),
        init=Param("random_uniform", -bound, bound)
        # init=Param("kaiming_uniform", 0)
    )
    var bias = g.param(
        TensorShape(out_channels), init=Param("random_uniform", -bound, bound)
    )

    return g.op(
        OP.CONV2D,
        inputs,
        weights,
        bias,
        attributes=AttributeVector(
            Attribute("padding", padding),
            Attribute("stride", stride),
            Attribute("dilation", dilation),
        ),
    )


struct Conv2dLayer(Layer, Copyable, Movable):
    """
    `Layer`-conforming wrapper around `Conv2d`, for use in a reflection-based
    Module struct.
    """

    var out_channels: Int
    var kernel_size: IndexList[2]
    var padding: IndexList[2]
    var stride: IndexList[2]
    var dilation: IndexList[2]

    def __init__(
        out self,
        out_channels: Int,
        kernel_size: IndexList[2],
        padding: IndexList[2] = IndexList[2](0, 0),
        stride: IndexList[2] = IndexList[2](1, 1),
        dilation: IndexList[2] = IndexList[2](1, 1),
    ):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        return Conv2d(
            g,
            input,
            self.out_channels,
            self.kernel_size,
            self.padding,
            self.stride,
            self.dilation,
        )

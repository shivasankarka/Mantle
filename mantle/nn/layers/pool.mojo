from mantle.core.tensor import Tensor, TensorShape
from std.collections.optional import Optional
from std.utils.index import IndexList

from mantle.autograd.graph import Graph
from mantle.autograd.symbol import Symbol
from mantle.autograd.ops import OP
from mantle.autograd.attributes import AttributeVector, Attribute
from mantle.nn.module import Layer


def set_static_stride(
    kernel_size: IndexList[2], stride: Optional[Int] = None
) -> IndexList[2]:
    if stride:
        return IndexList[2](stride.value(), stride.value())
    else:
        return kernel_size


def MaxPool2d(
    mut g: Graph,
    inputs: Symbol,
    kernel_size: IndexList[2],
    stride: Optional[Int] = None,
    padding: IndexList[2] = IndexList[2](0, 0),
    dilation: IndexList[2] = IndexList[2](1, 1),
) -> Symbol:
    """
    A 2D Max Pooling Layer.

    Kernel is unaware of the in_channels and out_channels of the input tensor.
    kernel.size     (kX, kY)
    """

    # TODO: assert padding <= kernel_size / 2 (at compile time)

    var stride_temp = set_static_stride(kernel_size, stride)

    return MaxPool2d(g, inputs, kernel_size, stride_temp, padding, dilation)


def MaxPool2d(
    mut g: Graph,
    inputs: Symbol,
    kernel_size: IndexList[2],
    stride: IndexList[2],  # stride should be 1 or more
    padding: IndexList[2] = IndexList[2](0, 0),
    dilation: IndexList[2] = IndexList[2](1, 1),
) -> Symbol:
    """
    A 2D Max Pooling Layer.

    Kernel is unaware of the in_channels and out_channels of the input tensor.
    kernel.size     (kX, kY)
    """
    # TODO: assert padding <= kernel_size / 2 (at compile time)

    return g.op(
        OP.MAXPOOL2D,
        inputs,
        attributes=AttributeVector(
            Attribute("kernel_size", kernel_size),
            Attribute("padding", padding),
            Attribute("stride", stride),
            Attribute("dilation", dilation),
        ),
    )


struct MaxPool2dLayer(Layer, Copyable, Movable):
    """
    `Layer`-conforming wrapper around `MaxPool2d`, for use in a
    reflection-based Module struct.
    """

    var kernel_size: IndexList[2]
    var stride: IndexList[2]
    var padding: IndexList[2]
    var dilation: IndexList[2]

    def __init__(
        out self,
        kernel_size: IndexList[2],
        stride: Optional[Int] = None,
        padding: IndexList[2] = IndexList[2](0, 0),
        dilation: IndexList[2] = IndexList[2](1, 1),
    ):
        self.kernel_size = kernel_size
        self.stride = set_static_stride(kernel_size, stride)
        self.padding = padding
        self.dilation = dilation

    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        return MaxPool2d(
            g, input, self.kernel_size, self.stride, self.padding, self.dilation
        )


# # TODO

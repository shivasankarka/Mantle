# ===----------------------------------------------------------------------=== #
# Mantle: Convolution Layers
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Conv (mantle.nn.layers.conv)
------------------------------------------------
2D Convolution layer with im2col-based implementation and layer wrapper.
"""
from mantle import f32
from mantle.autograd.graph import Graph
from mantle.autograd.symbol import Symbol
from mantle.autograd.ops import OP
from mantle.core.tensor import Tensor, TensorShape
from mantle.core.math_util import q_sqrt
from mantle.autograd.params import Param
from mantle.autograd.attributes import AttributeVector, Attribute
from mantle.nn.module import Expr, Layer, Module

from std.utils.index import IndexList


# ===----------------------------------------------------------------------===#
# Conv2d (functional)
# ===----------------------------------------------------------------------===#


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


def Conv2d(
    out_channels: Int,
    kernel_size: Int,
    padding: Int = 0,
    stride: Int = 1,
    dilation: Int = 1,
) -> Conv2dLayer:
    """Create a square Conv2d layer for ``Sequential`` or a module.

    The graph-building overload remains available for custom graphs.  Square
    integer arguments cover the common case without exposing ``IndexList``.
    """
    return Conv2dLayer(
        out_channels,
        IndexList[2](kernel_size, kernel_size),
        IndexList[2](padding, padding),
        IndexList[2](stride, stride),
        IndexList[2](dilation, dilation),
    )


# ===----------------------------------------------------------------------===#
# Conv2dLayer
# ===----------------------------------------------------------------------===#


struct Conv2dLayer(Copyable, Layer, Module, Movable):
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

    def forward(mut self, input: Expr) -> Expr:
        return Expr(
            input.graph,
            Conv2d(
                input.graph[],
                input.symbol,
                self.out_channels,
                self.kernel_size,
                self.padding,
                self.stride,
                self.dilation,
            ),
        )

    def __call__(mut self, input: Expr) -> Expr:
        return self.forward(input)


# ===----------------------------------------------------------------------===#
# ConvTranspose2d (functional)
# ===----------------------------------------------------------------------===#


def ConvTranspose2d(
    mut g: Graph,
    inputs: Symbol,
    out_channels: Int,
    kernel_size: IndexList[2],
    padding: IndexList[2] = IndexList[2](0, 0),
    stride: IndexList[2] = IndexList[2](1, 1),
    dilation: IndexList[2] = IndexList[2](1, 1),
    output_padding: IndexList[2] = IndexList[2](0, 0),
) -> Symbol:
    """
    A 2D transposed convolution layer (CPU only).

    Parameters
        inputs.shape     [batch, in_channels, iX, iY]
        kernel.shape     [in_channels, out_channels, kX, kY] (or weights)
        bias.shape       [out_channels].
        output.shape     [batch, out_channels, oX, oY].

    Notes:
        `output_padding` must satisfy `0 <= output_padding < stride` on each
        axis, matching PyTorch's `ConvTranspose2d`.
    """

    var in_channels: Int = inputs.shape[1]
    var fan_in: Scalar[f32] = Scalar[f32](
        in_channels * kernel_size[0] * kernel_size[1]
    )
    var bound = q_sqrt(fan_in)
    var weights = g.param(
        TensorShape(in_channels, out_channels, kernel_size[0], kernel_size[1]),
        init=Param("random_uniform", -bound, bound),
    )
    var bias = g.param(
        TensorShape(out_channels), init=Param("random_uniform", -bound, bound)
    )

    return g.op(
        OP.CONVTRANSPOSE2D,
        inputs,
        weights,
        bias,
        attributes=AttributeVector(
            Attribute("padding", padding),
            Attribute("stride", stride),
            Attribute("dilation", dilation),
            Attribute("output_padding", output_padding),
        ),
    )


def ConvTranspose2d(
    out_channels: Int,
    kernel_size: Int,
    padding: Int = 0,
    stride: Int = 1,
    dilation: Int = 1,
    output_padding: Int = 0,
) -> ConvTranspose2dLayer:
    """Create a square ConvTranspose2d layer for `Sequential` or a module."""
    return ConvTranspose2dLayer(
        out_channels,
        IndexList[2](kernel_size, kernel_size),
        IndexList[2](padding, padding),
        IndexList[2](stride, stride),
        IndexList[2](dilation, dilation),
        IndexList[2](output_padding, output_padding),
    )


# ===----------------------------------------------------------------------===#
# ConvTranspose2dLayer
# ===----------------------------------------------------------------------===#


struct ConvTranspose2dLayer(Copyable, Layer, Module, Movable):
    """
    `Layer`-conforming wrapper around `ConvTranspose2d`, for use in a
    reflection-based Module struct.
    """

    var out_channels: Int
    var kernel_size: IndexList[2]
    var padding: IndexList[2]
    var stride: IndexList[2]
    var dilation: IndexList[2]
    var output_padding: IndexList[2]

    def __init__(
        out self,
        out_channels: Int,
        kernel_size: IndexList[2],
        padding: IndexList[2] = IndexList[2](0, 0),
        stride: IndexList[2] = IndexList[2](1, 1),
        dilation: IndexList[2] = IndexList[2](1, 1),
        output_padding: IndexList[2] = IndexList[2](0, 0),
    ):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.output_padding = output_padding

    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        return ConvTranspose2d(
            g,
            input,
            self.out_channels,
            self.kernel_size,
            self.padding,
            self.stride,
            self.dilation,
            self.output_padding,
        )

    def forward(mut self, input: Expr) -> Expr:
        return Expr(
            input.graph,
            ConvTranspose2d(
                input.graph[],
                input.symbol,
                self.out_channels,
                self.kernel_size,
                self.padding,
                self.stride,
                self.dilation,
                self.output_padding,
            ),
        )

    def __call__(mut self, input: Expr) -> Expr:
        return self.forward(input)


# ===----------------------------------------------------------------------===#
# Conv1d (functional)
# ===----------------------------------------------------------------------===#


def Conv1d(
    mut g: Graph,
    inputs: Symbol,
    out_channels: Int,
    kernel_size: Int,
    padding: Int = 0,
    stride: Int = 1,
    dilation: Int = 1,
) -> Symbol:
    """
    A 1D Convolution Layer, implemented as `Conv2d` over a singleton spatial
    axis.

    Parameters
        inputs.shape     [batch, in_channels, length]
        weights.shape    [out_channels, in_channels, kernel_size]
        bias.shape       [out_channels].
        output.shape     [batch, out_channels, out_length].
    """
    var batch = inputs.shape[0]
    var in_channels = inputs.shape[1]
    var length = inputs.shape[2]

    var reshaped = g.op(
        OP.RESHAPE,
        inputs,
        attributes=AttributeVector(
            Attribute("shape", TensorShape(batch, in_channels, 1, length))
        ),
    )
    var conv_out = Conv2d(
        g,
        reshaped,
        out_channels,
        IndexList[2](1, kernel_size),
        IndexList[2](0, padding),
        IndexList[2](1, stride),
        IndexList[2](1, dilation),
    )
    var out_length = conv_out.shape[3]
    return g.op(
        OP.RESHAPE,
        conv_out,
        attributes=AttributeVector(
            Attribute(
                "shape", TensorShape(batch, out_channels, out_length)
            )
        ),
    )


def Conv1d(
    out_channels: Int,
    kernel_size: Int,
    padding: Int = 0,
    stride: Int = 1,
    dilation: Int = 1,
) -> Conv1dLayer:
    """Create a Conv1d layer for `Sequential` or a module."""
    return Conv1dLayer(out_channels, kernel_size, padding, stride, dilation)


# ===----------------------------------------------------------------------===#
# Conv1dLayer
# ===----------------------------------------------------------------------===#


struct Conv1dLayer(Copyable, Layer, Module, Movable):
    """
    `Layer`-conforming wrapper around `Conv1d`, for use in a reflection-based
    Module struct.
    """

    var out_channels: Int
    var kernel_size: Int
    var padding: Int
    var stride: Int
    var dilation: Int

    def __init__(
        out self,
        out_channels: Int,
        kernel_size: Int,
        padding: Int = 0,
        stride: Int = 1,
        dilation: Int = 1,
    ):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        return Conv1d(
            g,
            input,
            self.out_channels,
            self.kernel_size,
            self.padding,
            self.stride,
            self.dilation,
        )

    def forward(mut self, input: Expr) -> Expr:
        return Expr(
            input.graph,
            Conv1d(
                input.graph[],
                input.symbol,
                self.out_channels,
                self.kernel_size,
                self.padding,
                self.stride,
                self.dilation,
            ),
        )

    def __call__(mut self, input: Expr) -> Expr:
        return self.forward(input)

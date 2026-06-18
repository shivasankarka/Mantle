# ===----------------------------------------------------------------------=== #
# Mantle: Neural Networks
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""NN (mantle.nn)
------------------------------------------------
High-level neural network abstractions: layers, models, loss functions, optimizers.
"""
from mantle.core.tensor import Tensor, TensorShape
from .model import Model
from .module import Layer, build_graph, FlattenLayer, Sequential

from .layers.linear import Linear, LinearLayer
from .layers.conv import Conv2d, Conv2dLayer
from .layers.pool import MaxPool2d, MaxPool2dLayer

from .loss import MSELoss, CrossEntropyLoss
from .activations import (
    Softmax,
    LogSoftmax,
    ReLU,
    ReLULayer,
    LeakyReLU,
    Sigmoid,
    Tanh,
)

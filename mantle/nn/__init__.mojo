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
from .module import (
    Expr,
    Layer,
    Module,
    build_graph,
    build_module_graph,
    Flatten,
    FlattenLayer,
    Sequential,
)

from .layers.linear import Linear, LinearLayer
from .layers.conv import Conv2d, Conv2dLayer
from .layers.pool import MaxPool2d, MaxPool2dLayer
from .layers.dropout import Dropout, DropoutLayer
from .layers.batchnorm import BatchNorm2d, BatchNorm2dLayer
from .layers.embedding import (
    Embedding,
    EmbeddingLayer,
    PositionalEmbedding,
    PositionalEmbeddingLayer,
)
from .layers.layernorm import LayerNorm, LayerNormLayer
from .layers.attention import (
    MultiHeadAttention,
    MultiHeadAttentionLayer,
    causal_mask,
)
from .layers.feedforward import FeedForward, FeedForwardLayer
from .layers.transformer_block import TransformerBlock, TransformerBlockLayer

from .loss import (
    Loss,
    MSELoss,
    CrossEntropyLoss,
    L1Loss,
    classification_graph,
    supervised_graph,
)
from .metrics import accuracy
import mantle.nn.optim as optim
from .activations import (
    Softmax,
    SoftmaxLayer,
    LogSoftmax,
    ReLU,
    ReLULayer,
    LeakyReLU,
    Sigmoid,
    Tanh,
    GELU,
    GELULayer,
)

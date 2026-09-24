# ===----------------------------------------------------------------------=== #
# Mantle: Loss Functions
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Loss (mantle.nn.loss)
------------------------------------------------
Loss function implementations (MSE, Cross-Entropy, L1).
"""
from std.reflection import reflect_fn

import mantle.nn as nn
from mantle.core.tensor import Tensor, TensorShape
from mantle.autograd.graph import Graph
from mantle.autograd.symbol import Symbol
from mantle.autograd.ops import OP
from mantle.autograd.attributes import Attribute, AttributeVector
from mantle.nn.module import Expr, Layer, Module, build_graph


# ===----------------------------------------------------------------------===#
# MSELoss
# ===----------------------------------------------------------------------===#


def MSELoss(
    mut g: Graph,
    y_pred: Symbol,
    y_true: Symbol,
) -> Symbol:
    # 1/N * sum( (outputs - targets)^2 )

    var before = len(g.nodes)
    var diff = g.op(OP.SUB, y_true, y_pred)
    var loss = g.op(OP.POW, diff, 2)
    var mean_loss = g.op(OP.MEAN, loss)

    g.set_scope_from(before, reflect_fn[MSELoss].display_name())
    return mean_loss


# ===----------------------------------------------------------------------===#
# CrossEntropyLoss
# ===----------------------------------------------------------------------===#


# ===----------------------------------------------------------------------===#
# L1Loss (MAE)
# ===----------------------------------------------------------------------===#


def L1Loss(
    mut g: Graph,
    y_pred: Symbol,
    y_true: Symbol,
) -> Symbol:
    # 1/N * sum( abs(outputs - targets) )

    var before = len(g.nodes)
    var diff = g.op(OP.SUB, y_pred, y_true)
    var abs_diff = g.op(OP.ABS, diff)
    var mean_loss = g.op(OP.MEAN, abs_diff)

    g.set_scope_from(before, reflect_fn[L1Loss].display_name())
    return mean_loss


def CrossEntropyLoss(
    mut g: Graph,
    y_pred: Symbol,
    y_true: Symbol,
    label_smoothing: Float64 = 0.0,
) -> Symbol:
    # -1/N * sum( targets * log_softmax(outputs) )
    #
    # `label_smoothing` (0.0 disables it) blends the one-hot `y_true`
    # towards the uniform distribution before computing the loss:
    # y' = y * (1 - eps) + eps / C. Standard regularizer for classification
    # — keeps the model from driving logits to +-inf chasing a hard 0/1
    # target.

    var before = len(g.nodes)
    var log_softmax = nn.LogSoftmax(g, y_pred, axis=1)

    var targets = y_true
    if label_smoothing != 0.0:
        var num_classes = Float64(y_true.shape[-1])
        targets = g.op(
            OP.ADD,
            g.op(OP.MUL, y_true, 1.0 - label_smoothing),
            label_smoothing / num_classes,
        )

    # CrossEntropy (reduction Mean)
    var targets_log_softmax = g.op(OP.MUL, targets, log_softmax)
    var ret = g.op(OP.SUM, targets_log_softmax)
    var negDivN = g.op(OP.MUL, ret, -1.0 / Float64(y_pred.shape[0]))

    g.set_scope_from(before, reflect_fn[CrossEntropyLoss].display_name())
    return negDivN


def classification_graph[T: AnyType](
    mut network: T,
    input_shape: TensorShape,
    label_smoothing: Float64 = 0.0,
) -> Graph:
    """Build a static one-hot classification graph from a layer network.

    This is intentionally a graph builder, not another model wrapper:
    callers keep the familiar ``comptime graph = ...`` followed by
    ``Model[graph]()`` API.  It owns graph inputs, output registration, and
    cross-entropy wiring so an architecture never needs to thread ``Graph``
    through every layer call.
    """
    var g = Graph()
    var inputs = g.input(input_shape)
    var logits: Symbol
    comptime if conforms_to(T, Module):
        logits = network.forward(Expr(g, inputs)).symbol
    elif conforms_to(T, Layer):
        logits = network.forward(g, inputs)
    else:
        logits = build_graph(network, g, inputs)
    g.out(logits)

    var targets = g.input(TensorShape(input_shape[0], logits.shape[-1]))
    g.loss(CrossEntropyLoss(g, logits, targets, label_smoothing))
    return g^


# ===----------------------------------------------------------------------===#
# BCELoss
# ===----------------------------------------------------------------------===#


def BCELoss(
    mut g: Graph,
    y_pred: Symbol,
    y_true: Symbol,
) -> Symbol:
    """Binary cross-entropy: -mean( y*log(p) + (1-y)*log(1-p) ).

    `y_pred` is expected to already be a probability in [0, 1] (i.e. the
    output of a Sigmoid) — clipped to [eps, 1-eps] before the log for
    numerical stability, matching PyTorch's `BCELoss` behaviour.
    """

    var before = len(g.nodes)
    var p = g.op(
        OP.CLIP,
        y_pred,
        attributes=AttributeVector(
            Attribute("min", 1e-7), Attribute("max", 1.0 - 1e-7)
        ),
    )

    var term1 = g.op(OP.MUL, y_true, g.op(OP.LOG, p))
    var one_minus_y = g.op(OP.SUB, 1.0, y_true)
    var one_minus_p = g.op(OP.SUB, 1.0, p)
    var term2 = g.op(OP.MUL, one_minus_y, g.op(OP.LOG, one_minus_p))

    var mean_loss = g.op(OP.MEAN, g.op(OP.ADD, term1, term2))
    var neg_mean_loss = g.op(OP.MUL, mean_loss, -1.0)

    g.set_scope_from(before, reflect_fn[BCELoss].display_name())
    return neg_mean_loss

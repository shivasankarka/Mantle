# ===----------------------------------------------------------------------=== #
# Mantle: Loss Functions
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Loss (mantle.nn.loss)
------------------------------------------------
Loss function implementations (MSE, Cross-Entropy).
"""
from std.reflection import reflect_fn

import mantle.nn as nn
from mantle.core.tensor import Tensor, TensorShape
from mantle.autograd.graph import Graph
from mantle.autograd.symbol import Symbol
from mantle.autograd.ops import OP


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

def CrossEntropyLoss(
    mut g: Graph,
    y_pred: Symbol,
    y_true: Symbol,
) -> Symbol:
    # -1/N * sum( targets * log_softmax(outputs) )

    var before = len(g.nodes)
    var log_softmax = nn.LogSoftmax(g, y_pred, axis=1)

    # CrossEntropy (reduction Mean)
    var targets_log_softmax = g.op(OP.MUL, y_true, log_softmax)
    var ret = g.op(OP.SUM, targets_log_softmax)
    var negDivN = g.op(OP.MUL, ret, -1.0 / Float64(y_pred.shape[0]))

    g.set_scope_from(before, reflect_fn[CrossEntropyLoss].display_name())
    return negDivN

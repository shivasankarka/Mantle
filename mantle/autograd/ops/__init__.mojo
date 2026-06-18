# ===----------------------------------------------------------------------=== #
# Mantle: Ops
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Ops (mantle.autograd.ops)
------------------------------------------------
Differentiable operator implementations: arithmetic, activations, convolution, pooling.
"""
from .ops import (
    OP,
    static_result_shape,
    dynamic_result_shape,
    forward_op,
    backward_op,
)

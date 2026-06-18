# ===----------------------------------------------------------------------=== #
# Mantle: Core
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Core (mantle.core)
------------------------------------------------
Foundation types and utilities.
"""
from .tensor import Tensor, TensorShape
from .math_util import q_sqrt
from .bytes import Bytes, scalar_to_bytes, bytes_to_scalar
from .rand_utils import rand_uniform, rand_normal
from .tensorutils import (
    fill,
    elwise_transform,
    elwise_op,
    broadcast_shapes,
    # broadcast_arrays,
    tsum,
    tmean,
    tmax,
    tstd,
    transpose,
    transpose_2D,
    accumulate_grad,
    accumulate_op,
)

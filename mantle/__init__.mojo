# ===----------------------------------------------------------------------=== #
# Mantle: A high performance machine learning framework written in pure mojo.
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Mantle
----------
A high performance machine learning framework written in pure mojo.
"""

from std.sys.info import simd_width_of

from mantle.core.tensor import Tensor, TensorShape
from mantle.autograd import Graph, Symbol, OP
from mantle.autograd.collection import Collection

comptime f32 = DType.float32
"""32-bit floating-point data type."""
comptime nelts = 2 * simd_width_of[f32]()
"""Number of elements in a SIMD vector for the specified data type (f32)."""
comptime seed = 42
"""Seed value for random number generation, ensuring reproducibility."""
comptime epsilon = 1e-12
"""Small constant used to prevent division by zero in numerical computations."""

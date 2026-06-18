# ===----------------------------------------------------------------------=== #
# Mantle: Symbol
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Symbol (mantle.autograd.symbol)
------------------------------------------------
Represents a named typed tensor in the compute graph, used as data-flow tokens
between graph nodes.
"""
from mantle.core.tensor import Tensor, TensorShape


# ===----------------------------------------------------------------------===#
# Symbol
# ===----------------------------------------------------------------------===#

@fieldwise_init
struct Symbol(
    Copyable,
    Equatable,
    ImplicitlyCopyable,
    Movable,
    TrivialRegisterPassable,
    Writable,
):
    var name: UInt32  # name of the symbol
    var dtype: DType
    var shape: TensorShape
    var trainable: Bool

    def __eq__(self, other: Self) -> Bool:
        return self.name == other.name

    def __ne__(self, other: Self) -> Bool:
        return self.name != other.name

    def __str__(self) -> String:
        return self.json()

    def json(self) -> String:
        var shape_str: String = ""
        for i in range(self.shape.rank()):
            shape_str += String(self.shape[i])
            if i < self.shape.rank() - 1:
                shape_str += "x"

        return (
            '{"name": "'
            + String(self.name)
            + '", "dtype": "'
            + String(self.dtype)
            + '", "shape": "'
            + shape_str
            + '", "trainable": "'
            + String(self.trainable)
            + '"}'
        )

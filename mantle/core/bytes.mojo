# ===----------------------------------------------------------------------=== #
# Mantle: A high performance machine learning framework written in pure mojo.
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Bytes (mantle.utils.bytes)
------------------------------------------------
This module defines the `Bytes` struct, which represents a static sequence of bytes.
It also provides functions to convert between scalar values and their byte representations.
"""
from std.math import nan
from std.sys.info import size_of
from std.utils.numerics import inf
from std.utils.static_tuple import StaticTuple

comptime ScalarBytes = size_of[DType.uint64]()
"""Number of bytes required to represent a scalar value (64 bits)."""

struct Bytes[capacity: Int](
    Copyable, Equatable, Movable, TrivialRegisterPassable, Writable
):
    """
    Static sequence of bytes.
    """

    var data: StaticTuple[UInt8, Self.capacity]

    def __init__(out self):
        var data = StaticTuple[UInt8, Self.capacity](0)

        for i in range(Self.capacity):
            data[i] = 0

        self.data = data

    def __init__(out self, s: String):
        var data = StaticTuple[UInt8, Self.capacity](0)
        var length = s.byte_length()

        for i in range(Self.capacity):
            data[i] = UInt8(ord(s[byte=i])) if i < length else 0

        self.data = data

    @always_inline("nodebug")
    def __len__(self) -> Int:
        return Self.capacity

    @always_inline("nodebug")
    def __setitem__(mut self, index: Int, value: UInt8):
        self.data[index] = value

    @always_inline("nodebug")
    def __getitem__(self, index: Int) -> UInt8:
        return self.data[index]

    @always_inline("nodebug")
    def __eq__(self, other: Self) -> Bool:
        for i in range(Self.capacity):
            if self[i] != other[i]:
                return False
        return True

    @always_inline("nodebug")
    def __ne__(self, other: Self) -> Bool:
        for i in range(Self.capacity):
            if self[i] != other[i]:
                return True
        return False

    @always_inline("nodebug")
    def write_to[W: Writer](self, mut writer: W):
        for i in range(Self.capacity):
            var val = self[i]
            if val != 0:
                writer.write(chr(Int(val)))

    @always_inline("nodebug")
    def __str__(self) -> String:
        return String.write(self)


def scalar_to_bytes[
    dtype: DType, size: Int = ScalarBytes
](value: Scalar[dtype]) -> Bytes[size]:
    """Convert a scalar value to its byte representation.

    Parameters:
        dtype: The data type of the scalar value.
        size: The number of bytes to represent the scalar value (default is 8 bytes for 64-bit types).

    Args:
        value: The scalar value to convert.

    Returns:
        A `Bytes` instance containing the byte representation of the scalar value.
    """
    comptime assert size >= ScalarBytes, "Size must be at least ${ScalarBytes}"

    var bits = bitcast[DType.uint64](value.cast[expand_type[dtype]()]())
    var data = Bytes[size]()

    for i in range(ScalarBytes):
        data[i] = (bits >> UInt64((i << 3))).cast[DType.uint8]()

    return data


def bytes_to_scalar[dtype: DType](data: Bytes) -> Scalar[dtype]:
    comptime assert (
        data.capacity >= ScalarBytes
    ), "Size must be at least ${ScalarBytes}"

    var bits: UInt64 = 0

    for i in range(ScalarBytes):
        bits |= data[i].cast[DType.uint64]() << UInt64((i << 3))

    return bitcast[expand_type[dtype]()](bits).cast[dtype]()


def expand_type[dtype: DType]() -> DType:
    comptime if dtype.is_floating_point():
        return DType.float64
    elif dtype.is_signed():
        return DType.int64
    elif dtype.is_integral():
        return DType.uint64
    # comptime assert False, "Unsupported data type: ${dtype}"
    return DType.invalid

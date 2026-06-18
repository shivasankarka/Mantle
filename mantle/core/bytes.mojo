# ===----------------------------------------------------------------------=== #
# Mantle: A high performance machine learning framework written in pure mojo.
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Bytes (mantle.core.bytes)
------------------------------------------------
This module defines the `Bytes` struct, which represents a static sequence of bytes.
It also provides functions to convert between scalar values and their byte representations.
"""
from std.math import nan
from std.sys.info import size_of
from std.utils.numerics import inf
from std.utils.static_tuple import StaticTuple
from std.memory.unsafe import bitcast

# ===----------------------------------------------------------------------===#
# Constants
# ===----------------------------------------------------------------------===#

comptime ScalarBytes = size_of[DType.uint64]()
"""Number of bytes required to represent a scalar value (64 bits)."""

# ===----------------------------------------------------------------------===#
# Bytes
# ===----------------------------------------------------------------------===#

struct Bytes[capacity: Int](
    Copyable, Equatable, Movable, TrivialRegisterPassable, Writable
):
    """
    Static sequence of bytes.
    """

    var data: StaticTuple[UInt8, Self.capacity]
    """The underlying byte storage."""

    def __init__(out self):
        var data = StaticTuple[UInt8, Self.capacity](0)

        for i in range(Self.capacity):
            data[i] = 0

        self.data = data

    def __init__(out self, s: String):
        """
        Create a Bytes instance from a string.

        Args:
            s: The string to encode as bytes.
        """
        var data = StaticTuple[UInt8, Self.capacity](0)
        var length = s.byte_length()

        for i in range(Self.capacity):
            data[i] = UInt8(ord(s[byte=i])) if i < length else 0

        self.data = data

    @always_inline("nodebug")
    def __len__(self) -> Int:
        """
        Returns:
            The capacity of the byte buffer.
        """
        return Self.capacity

    @always_inline("nodebug")
    def __setitem__(mut self, index: Int, value: UInt8):
        """
        Set the byte at the given index.

        Args:
            index: The position to set.
            value: The byte value.
        """
        self.data[index] = value

    @always_inline("nodebug")
    def __getitem__(self, index: Int) -> UInt8:
        """
        Get the byte at the given index.

        Args:
            index: The position to read.

        Returns:
            The byte at the given index.
        """
        return self.data[index]

    @always_inline("nodebug")
    def __eq__(self, other: Self) -> Bool:
        """
        Check equality with another Bytes instance.

        Args:
            other: The other Bytes instance.

        Returns:
            True if all bytes are equal.
        """
        for i in range(Self.capacity):
            if self[i] != other[i]:
                return False
        return True

    @always_inline("nodebug")
    def __ne__(self, other: Self) -> Bool:
        """
        Check inequality with another Bytes instance.

        Args:
            other: The other Bytes instance.

        Returns:
            True if any byte differs.
        """
        for i in range(Self.capacity):
            if self[i] != other[i]:
                return True
        return False

    @always_inline("nodebug")
    def write_to[W: Writer](self, mut writer: W):
        """
        Write the non-null bytes to a writer.

        Args:
            writer: The writer to write to.
        """
        for i in range(Self.capacity):
            var val = self[i]
            if val != 0:
                writer.write(chr(Int(val)))

    @always_inline("nodebug")
    def __str__(self) -> String:
        return String.write(self)


# ===----------------------------------------------------------------------===#
# Scalar Conversion
# ===----------------------------------------------------------------------===#

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
    """
    Convert a byte representation to a scalar value.

    Parameters:
        dtype: The target data type.

    Args:
        data: The Bytes instance to convert.

    Returns:
        The scalar value reconstructed from the byte representation.
    """
    comptime assert (
        data.capacity >= ScalarBytes
    ), "Size must be at least ${ScalarBytes}"

    var bits: UInt64 = 0

    for i in range(ScalarBytes):
        bits |= data[i].cast[DType.uint64]() << UInt64((i << 3))

    return bitcast[expand_type[dtype]()](bits).cast[dtype]()


# ===----------------------------------------------------------------------===#
# Type Helpers
# ===----------------------------------------------------------------------===#

def expand_type[dtype: DType]() -> DType:
    """
    Map a data type to its widest representation.

    Parameters:
        dtype: The input data type.

    Returns:
        Float64 for floating-point types, int64 for signed types, uint64 for unsigned types.
    """
    comptime if dtype.is_floating_point():
        return DType.float64
    elif dtype.is_signed():
        return DType.int64
    elif dtype.is_integral():
        return DType.uint64
    # comptime assert False, "Unsupported data type: ${dtype}"
    return DType.invalid

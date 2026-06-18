# ===----------------------------------------------------------------------=== #
# Mantle: Math Utilities
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Math Util (mantle.core.math_util)
------------------------------------------------
SIMD math wrappers for arithmetic and transcendental operations.
"""
from std.math import exp as _std_exp, log as _std_log, sqrt as _std_sqrt
from std.memory.unsafe import bitcast


# ===----------------------------------------------------------------------===#
# Fast Inverse Square Root
# ===----------------------------------------------------------------------===#

@always_inline("nodebug")
def q_sqrt(value: Float32) -> Float32:
    """
    Fast inverse square root using the Quake III algorithm.

    Args:
        value: The input value.

    Returns:
        An approximation of 1 / sqrt(value).
    """
    var y = bitcast[DType.float32](
        0x5F3759DF - (bitcast[DType.uint32](value) >> 1)
    )
    return -y * ((0.5 * value * y).fma(y, -1.5))


@always_inline
def add[
    dtype: DType, simd_width: Int
](a: SIMD[dtype, simd_width], b: SIMD[dtype, simd_width]) -> SIMD[
    dtype, simd_width
]:
    """
    Element-wise addition.

    Parameters:
        dtype: The data type of the operands.
        simd_width: The SIMD vector width.

    Args:
        a: First operand.
        b: Second operand.

    Returns:
        The sum of a and b.
    """
    return a + b


@always_inline
def sub[
    dtype: DType, simd_width: Int
](a: SIMD[dtype, simd_width], b: SIMD[dtype, simd_width]) -> SIMD[
    dtype, simd_width
]:
    """
    Element-wise subtraction.

    Parameters:
        dtype: The data type of the operands.
        simd_width: The SIMD vector width.

    Args:
        a: First operand.
        b: Second operand.

    Returns:
        The difference of a and b.
    """
    return a - b


@always_inline
def mul[
    dtype: DType, simd_width: Int
](a: SIMD[dtype, simd_width], b: SIMD[dtype, simd_width]) -> SIMD[
    dtype, simd_width
]:
    """
    Element-wise multiplication.

    Parameters:
        dtype: The data type of the operands.
        simd_width: The SIMD vector width.

    Args:
        a: First operand.
        b: Second operand.

    Returns:
        The product of a and b.
    """
    return a * b


@always_inline
def div[
    dtype: DType, simd_width: Int
](a: SIMD[dtype, simd_width], b: SIMD[dtype, simd_width]) -> SIMD[
    dtype, simd_width
]:
    """
    Element-wise division.

    Parameters:
        dtype: The data type of the operands.
        simd_width: The SIMD vector width.

    Args:
        a: First operand.
        b: Second operand.

    Returns:
        The quotient of a and b.
    """
    return a / b


@always_inline
def round_simd[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """
    Element-wise rounding to the nearest integer.

    Parameters:
        dtype: The data type.
        simd_width: The SIMD vector width.

    Args:
        x: The input value.

    Returns:
        X rounded to the nearest integer.
    """
    return round(x)


@always_inline
def exp[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[
    dtype, simd_width
] where dtype.is_floating_point():
    """
    Element-wise exponential.

    Constraints:
        Dtype must be a floating-point type.

    Parameters:
        dtype: The data type.
        simd_width: The SIMD vector width.

    Args:
        x: The input value.

    Returns:
        E raised to the power of x.
    """
    return _std_exp(x.cast[DType.float32]()).cast[dtype]()


@always_inline
def log[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[
    dtype, simd_width
] where dtype.is_floating_point():
    """
    Element-wise natural logarithm.

    Constraints:
        Dtype must be a floating-point type.

    Parameters:
        dtype: The data type.
        simd_width: The SIMD vector width.

    Args:
        x: The input value.

    Returns:
        The natural logarithm of x.
    """
    return _std_log(x.cast[DType.float32]()).cast[dtype]()


@always_inline
def sqrt_simd[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[
    dtype, simd_width
] where dtype.is_floating_point():
    """
    Element-wise square root.

    Constraints:
        Dtype must be a floating-point type.

    Parameters:
        dtype: The data type.
        simd_width: The SIMD vector width.

    Args:
        x: The input value.

    Returns:
        The square root of x.
    """
    return _std_sqrt(x.cast[DType.float32]()).cast[dtype]()


@always_inline
def max_simd[
    dtype: DType, simd_width: Int
](a: SIMD[dtype, simd_width], b: SIMD[dtype, simd_width]) -> SIMD[
    dtype, simd_width
]:
    """
    Element-wise maximum.

    Parameters:
        dtype: The data type.
        simd_width: The SIMD vector width.

    Args:
        a: First operand.
        b: Second operand.

    Returns:
        The element-wise maximum of a and b.
    """
    return a.gt(b).select(a, b)

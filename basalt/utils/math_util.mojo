from std.math import exp as _std_exp, log as _std_log, sqrt as _std_sqrt


@always_inline
def add[
    dtype: DType, simd_width: Int
](a: SIMD[dtype, simd_width], b: SIMD[dtype, simd_width]) -> SIMD[
    dtype, simd_width
]:
    return a + b


@always_inline
def sub[
    dtype: DType, simd_width: Int
](a: SIMD[dtype, simd_width], b: SIMD[dtype, simd_width]) -> SIMD[
    dtype, simd_width
]:
    return a - b


@always_inline
def mul[
    dtype: DType, simd_width: Int
](a: SIMD[dtype, simd_width], b: SIMD[dtype, simd_width]) -> SIMD[
    dtype, simd_width
]:
    return a * b


@always_inline
def div[
    dtype: DType, simd_width: Int
](a: SIMD[dtype, simd_width], b: SIMD[dtype, simd_width]) -> SIMD[
    dtype, simd_width
]:
    return a / b


@always_inline
def round_simd[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    return round(x)


@always_inline
def exp[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width] where dtype.is_floating_point():
    return _std_exp(x.cast[DType.float32]()).cast[dtype]()


@always_inline
def log[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width] where dtype.is_floating_point():
    return _std_log(x.cast[DType.float32]()).cast[dtype]()


@always_inline
def sqrt_simd[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width] where dtype.is_floating_point():
    return _std_sqrt(x.cast[DType.float32]()).cast[dtype]()


@always_inline
def max_simd[
    dtype: DType, simd_width: Int
](a: SIMD[dtype, simd_width], b: SIMD[dtype, simd_width]) -> SIMD[
    dtype, simd_width
]:
    return a.gt(b).select(a, b)

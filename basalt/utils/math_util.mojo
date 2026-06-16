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

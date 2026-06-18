from std.random import rand, randn
from std.algorithm import vectorize
from std.utils.static_tuple import StaticTuple

from mantle.core.tensor import Tensor


@always_inline
def rand_uniform[
    dtype: DType
](mut res: Tensor[dtype], low: Scalar[dtype], high: Scalar[dtype]):
    var scale = high - low

    rand[dtype](res.mut_ptr(), res.num_elements())

    def vecscale[nelts: Int](idx: Int) {mut res, read scale, read low}:
        res.store[nelts](idx, res.load[nelts](idx).fma(scale, low))

    vectorize[nelts](res.num_elements(), vecscale)


@always_inline
def rand_normal[
    dtype: DType
](mut res: Tensor[dtype], mean: Float64, std: Float64):
    randn[dtype](res.mut_ptr(), res.num_elements(), mean, std**2)


struct MersenneTwister(TrivialRegisterPassable):
    """
    Pseudo-random generator Mersenne Twister (MT19937-32bit).
    """

    comptime N: Int = 624
    comptime M: Int = 397
    comptime MATRIX_A: Int32 = 0x9908B0DF
    comptime UPPER_MASK: Int32 = 0x80000000
    comptime LOWER_MASK: Int32 = 0x7FFFFFFF
    comptime TEMPERING_MASK_B: Int32 = 0x9D2C5680
    comptime TEMPERING_MASK_C: Int32 = 0xEFC60000

    var state: StaticTuple[Int32, Self.N]
    var index: Int

    def __init__(out self, seed: Int):
        comptime W: Int = 32
        comptime F: Int32 = 1812433253
        comptime D: Int32 = 0xFFFFFFFF

        self.index = Self.N
        self.state = StaticTuple[Int32, Self.N]()
        self.state[0] = Int32(seed) & D

        for i in range(1, Self.N):
            var prev = self.state[i - 1]
            self.state[i] = (F * (prev ^ (prev >> Int32(W - 2))) + Int32(i)) & D

    def next(mut self) -> Int32:
        if self.index >= Self.N:
            for i in range(Self.N):
                var x = (self.state[i] & Self.UPPER_MASK) + (
                    self.state[(i + 1) % Self.N] & Self.LOWER_MASK
                )
                var xA = x >> 1
                if x % 2 != 0:
                    xA ^= Self.MATRIX_A
                self.state[i] = self.state[(i + Self.M) % Self.N] ^ xA
            self.index = 0

        var y = self.state[self.index]
        y ^= y >> 11
        y ^= (y << 7) & Self.TEMPERING_MASK_B
        y ^= (y << 15) & Self.TEMPERING_MASK_C
        y ^= y >> 18
        self.index += 1

        return y

    def next_ui8(mut self) -> UInt8:
        return UInt8(self.next() & Int32(0xFF))

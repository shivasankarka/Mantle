from math import nan
from utils.numerics import inf
from utils.static_tuple import StaticTuple
from sys.info import size_of

comptime ScalarBytes = size_of[DType.uint64]()

@register_passable("trivial")
struct Bytes[capacity: Int](Stringable, Copyable, Movable, EqualityComparable):
    """
    Static sequence of bytes.
    """

    var data: StaticTuple[UInt8, capacity]

    def __init__(out self):
        var data = StaticTuple[UInt8, capacity](0)

        for i in range(capacity):
            data[i] = 0

        self.data = data

    def __init__(out self, s: String):
        var data = StaticTuple[UInt8, capacity](0)
        var length = len(s)

        for i in range(capacity):
            data[i] = ord(s[i]) if i < length else 0

        self.data = data

    @always_inline("nodebug")
    def __len__(self) -> Int:
        return capacity

    @always_inline("nodebug")
    def __setitem__(mut self, index: Int, value: UInt8):
        self.data[index] = value

    @always_inline("nodebug")
    def __getitem__(self, index: Int) -> UInt8:
        return self.data[index]

    @always_inline("nodebug")
    def __eq__(self, other: Self) -> Bool:
        for i in range(capacity):
            if self[i] != other[i]:
                return False
        return True

    @always_inline("nodebug")
    def __ne__(self, other: Self) -> Bool:
        for i in range(capacity):
            if self[i] != other[i]:
                return True
        return False

    @always_inline("nodebug")
    def __str__(self) -> String:
        var result: String = ""

        for i in range(capacity):
            var val = self[i]
            if val != 0:
                result += chr(Int(val))

        return result


def scalar_to_bytes[
    dtype: DType, Size: Int = ScalarBytes
](value: Scalar[dtype]) -> Bytes[Size]:
    constrained[Size >= ScalarBytes, "Size must be at least ${ScalarBytes}"]()

    var bits = bitcast[DType.uint64](value.cast[expand_type[dtype]()]())
    var data = Bytes[Size]()

    for i in range(ScalarBytes):
        data[i] = (bits >> (i << 3)).cast[DType.uint8]()

    return data


def bytes_to_scalar[dtype: DType](data: Bytes) -> Scalar[dtype]:
    constrained[data.capacity >= ScalarBytes, "Size must be at least ${ScalarBytes}"]()

    var bits: UInt64 = 0

    for i in range(ScalarBytes):
        bits |= data[i].cast[DType.uint64]() << (i << 3)

    return bitcast[expand_type[dtype]()](bits).cast[dtype]()


def expand_type[dtype: DType]() -> DType:
    @parameter
    if dtype.is_floating_point():
        return DType.float64
    elif dtype.is_signed():
        return DType.int64
    elif dtype.is_integral():
        return DType.uint64

    constrained[False, "Type must be numeric"]()
    return DType.invalid

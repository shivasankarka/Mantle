from testing import assert_true
from algorithm import vectorize
from utils.index import IndexList
from memory import memset_zero, memcpy, UnsafePointer


comptime MAX_RANK = 8


@register_passable("trivial")
struct TensorShape(Stringable):
    var _rank: Int
    var _shape: IndexList[MAX_RANK]

    def __init__(out self, *shape: Int):
        self._rank = len(shape)
        self._shape = IndexList[MAX_RANK]()
        for i in range(min(self._rank, MAX_RANK)):
            self._shape[i] = shape[i]

    def __init__(out self, shapes: VariadicList[Int]):
        self._rank = len(shapes)
        self._shape = IndexList[MAX_RANK]()
        for i in range(min(self._rank, MAX_RANK)):
            self._shape[i] = shapes[i]

    def __init__(out self, shape: List[Int]):
        self._rank = len(shape)
        self._shape = IndexList[MAX_RANK]()
        for i in range(min(self._rank, MAX_RANK)):
            self._shape[i] = shape[i]

    def __init__[num: Int](out self, shape: IndexList[num]):
        self._rank = num
        self._shape = IndexList[MAX_RANK]()
        for i in range(min(self._rank, MAX_RANK)):
            self._shape[i] = shape[i]

    def __init__(out self, rank: Int, shape: IndexList[MAX_RANK]):
        self._rank = rank
        self._shape = shape

    @always_inline("nodebug")
    def __getitem__(self, index: Int) -> Int:
        return self._shape[index if index >= 0 else self._rank + index]

    @always_inline("nodebug")
    def __setitem__(mut self, index: Int, value: Int):
        self._shape[index if index >= 0 else self._rank + index] = value

    @always_inline("nodebug")
    def rank(self) -> Int:
        return self._rank

    def num_elements(self) -> Int:
        var result = 1
        for i in range(self._rank):
            result *= self._shape[i]
        return result

    def strides(self) -> IndexList[MAX_RANK]:
        var result = IndexList[MAX_RANK](0)
        result[self._rank - 1] = 1
        for i in range(self._rank - 2, -1, -1):
            result[i] = result[i + 1] * self._shape[i + 1]
        return result

    def __str__(self) -> String:
        var s: String = "("
        for i in range(self._rank):
            s += String(self._shape[i])
            if i < self._rank - 1:
                s += ", "
        return s + ")"

    @always_inline("nodebug")
    def __eq__(self, other: TensorShape) -> Bool:
        if self.rank() != other.rank():
            return False
        for i in range(self.rank()):
            if self[i] != other[i]:
                return False
        return True

    @always_inline("nodebug")
    def __ne__(self, other: TensorShape) -> Bool:
        return not self.__eq__(other)

    def __contains__(self, value: Int) -> Bool:
        for i in range(self.rank()):
            if self[i] == value:
                return True
        return False

    def to_list(self) -> List[Int]:
        var result = List[Int]()
        for i in range(self.rank()):
            result.append(self[i])
        return result^


struct Tensor[dtype: DType](Stringable, Movable, Copyable, Movable):
    var _data: UnsafePointer[Scalar[dtype], MutExternalOrigin]
    var _shape: TensorShape

    def __init__(out self, *dims: Int):
        self._shape = TensorShape(dims)
        self._data = alloc[Scalar[dtype]](self._shape.num_elements())
        memset_zero(self._data, self._shape.num_elements())

    def __init__(out self, var shape: TensorShape):
        self._data = alloc[Scalar[dtype]](shape.num_elements())
        memset_zero(self._data, shape.num_elements())
        self._shape = shape

    def __init__(out self, shapes: VariadicList[Int]):
        self._shape = TensorShape(shapes)
        self._data = alloc[Scalar[dtype]](self._shape.num_elements())
        memset_zero(self._data, self._shape.num_elements())

    def __init__(
        out self, var data: UnsafePointer[Scalar[dtype]], var shape: TensorShape
    ):
        self._data = alloc[Scalar[dtype]](shape.num_elements())
        self._shape = shape

        memcpy(dest=self._data, src=data, count=self._shape.num_elements())
        _ = data

    def __moveinit__(out self, deinit other: Tensor[dtype]):
        self._data = other._data
        self._shape = other._shape

    def __copyinit__(out self, other: Tensor[dtype]):
        self._data = alloc[Scalar[dtype]](other._shape.num_elements())
        memcpy(dest=self._data, src=other._data, count=other.num_elements())
        self._shape = other._shape

    @always_inline("nodebug")
    def __getitem__(self, index: Int) -> Scalar[dtype]:
        return self._data[index]

    @always_inline("nodebug")
    def __setitem__(self, index: Int, value: Scalar[dtype]):
        self._data[index] = value

    @always_inline("nodebug")
    def data(self) -> UnsafePointer[Scalar[dtype], MutExternalOrigin]:
        return self._data

    @always_inline("nodebug")
    def shape(self) -> TensorShape:
        return self._shape

    @always_inline("nodebug")
    def load[simd_width: Int](self, index: Int) -> SIMD[dtype, simd_width]:
        return self._data.load[width=simd_width](index)

    @always_inline("nodebug")
    def store[simd_width: Int](self, index: Int, value: SIMD[dtype, simd_width]):
        self._data.store(index, value)

    @always_inline("nodebug")
    def strides(self) -> IndexList[MAX_RANK]:
        return self._shape.strides()

    @always_inline("nodebug")
    def rank(self) -> Int:
        return self._shape.rank()

    @always_inline("nodebug")
    def num_elements(self) -> Int:
        return self._shape.num_elements()

    @always_inline("nodebug")
    def dim(self, index: Int) -> Int:
        return self._shape[index]

    @always_inline("nodebug")
    def zero(self):
        memset_zero(self._data, self.num_elements())

    @always_inline("nodebug")
    def ireshape(inout self, new_shape: TensorShape) raises:
        # NOTE Consider not raising on error
        assert_true(self.num_elements() == new_shape.num_elements())
        self._shape = new_shape

    def __str__(self) -> String:
        # temp fix
        var s: String = "["
        for i in range(self.num_elements()):
            s += String(self[i])
            if i < self.num_elements() - 1:
                s += ", "
        return s + "]"


    @always_inline("nodebug")
    def __del__(deinit self):
        self._data.free()

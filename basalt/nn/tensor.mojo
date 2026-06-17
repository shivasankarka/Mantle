from std.testing import assert_true
from std.algorithm import vectorize
from std.collections.optional import Optional
from std.utils.index import IndexList
from std.memory import memset_zero, memcpy, UnsafePointer

comptime MAX_RANK = 8


struct TensorShape(Equatable, TrivialRegisterPassable, Writable):
    var _rank: Int
    var _shape: IndexList[MAX_RANK]

    def __init__(out self, *shape: Int):
        self._rank = len(shape)
        self._shape = IndexList[MAX_RANK]()
        for i in range(min(self._rank, MAX_RANK)):
            self._shape[i] = shape[i]

    def __init__(out self, shapes: VariadicList[Int, _]):
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

    def write_to(self, mut writer: Some[Writer]):
        """Writes the array to a writer.

        Args:
            writer: The writer to write the array to.
        """
        writer.write(self.__str__())


struct Tensor[dtype: DType](Copyable, Movable, Writable):
    var _data_owner: Optional[UnsafePointer[Scalar[Self.dtype], MutExternalOrigin]]
    var _data_ref: UnsafePointer[Scalar[Self.dtype], MutExternalOrigin]
    var _shape: TensorShape

    def __init__(out self, *dims: Int):
        self._shape = TensorShape(dims)
        self._data_owner = alloc[Scalar[Self.dtype]](self._shape.num_elements())
        self._data_ref = self._data_owner.value()
        memset_zero(self._data_ref, self._shape.num_elements())

    def __init__(out self, var shape: TensorShape):
        self._data_owner = alloc[Scalar[Self.dtype]](shape.num_elements())
        self._data_ref = self._data_owner.value()
        memset_zero(self._data_ref, shape.num_elements())
        self._shape = shape

    def __init__(out self, shapes: VariadicList[Int, _]):
        self._shape = TensorShape(shapes)
        self._data_owner = alloc[Scalar[Self.dtype]](self._shape.num_elements())
        self._data_ref = self._data_owner.value()
        memset_zero(self._data_ref, self._shape.num_elements())

    def __init__[
        origin: MutOrigin
    ](
        out self,
        var data: UnsafePointer[Scalar[Self.dtype], origin],
        var shape: TensorShape,
    ):
        self._data_owner = alloc[Scalar[Self.dtype]](shape.num_elements())
        self._data_ref = self._data_owner.value()
        self._shape = shape

        memcpy(dest=self._data_ref, src=data, count=self._shape.num_elements())
        _ = data

    def __init__(out self, *, deinit take: Tensor[Self.dtype]):
        self._data_owner = take._data_owner^
        self._data_ref = self._data_owner.value()
        self._shape = take._shape

    def __init__(out self, *, copy: Tensor[Self.dtype]):
        self._data_owner = alloc[Scalar[Self.dtype]](copy._shape.num_elements())
        self._data_ref = self._data_owner.value()
        memcpy(dest=self._data_ref, src=copy._data_ref, count=copy.num_elements())
        self._shape = copy._shape

    @always_inline("nodebug")
    def __getitem__(self, index: Int) -> Scalar[Self.dtype]:
        return self._data_ref[index]

    @always_inline("nodebug")
    def __setitem__(self, index: Int, value: Scalar[Self.dtype]):
        self._data_ref[index] = value

    @always_inline("nodebug")
    def data(self) -> UnsafePointer[Scalar[Self.dtype], MutExternalOrigin]:
        return self._data_ref

    @always_inline("nodebug")
    def shape(self) -> TensorShape:
        return self._shape

    @always_inline("nodebug")
    def load[simd_width: Int](self, index: Int) -> SIMD[Self.dtype, simd_width]:
        return self._data_ref.load[width=simd_width](index)

    @always_inline("nodebug")
    def store[
        simd_width: Int
    ](self, index: Int, value: SIMD[Self.dtype, simd_width]):
        self._data_ref.store(index, value)

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
        memset_zero(self._data_ref, self.num_elements())

    @always_inline("nodebug")
    def ireshape(mut self, new_shape: TensorShape) raises:
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
        if self._data_owner:
            self._data_owner.value().free()

    def write_to(self, mut writer: Some[Writer]):
        """Writes the tensor to a writer.

        Args:
            writer: The writer to write the tensor to.
        """
        writer.write(self.__str__())

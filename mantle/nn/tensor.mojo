# ===----------------------------------------------------------------------=== #
# Mantle: Tensor
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Tensor (mantle.nn.tensor)
------------------------------------------------
This module defines the `Tensor` struct, which represents a multi-dimensional array of data. It also defines the `TensorShape` struct,
which represents the shape of a tensor. The `Tensor` struct includes reference counting for memory management and supports basic
operations such as indexing, reshaping, and zeroing out the data.
"""
from std.testing import assert_true
from std.algorithm import vectorize
from std.atomic import Atomic, Ordering, fence
from std.utils.index import IndexList
from std.memory import memset_zero, memcpy, UnsafePointer

comptime MAX_RANK = 8
"""Max rank of a tensor."""
# TODO: make it an explicit input to Tensor

# ===----------------------------------------------------------------------===#
# TensorShape
# ===----------------------------------------------------------------------===#

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


# ===----------------------------------------------------------------------===#
# Tensor
# ===----------------------------------------------------------------------===#

struct Tensor[dtype: DType](Copyable, Movable, Writable):
    var _data: UnsafePointer[Scalar[Self.dtype], MutUntrackedOrigin]
    var _refcount: UnsafePointer[Atomic[DType.uint64], MutUntrackedOrigin]
    var _shape: TensorShape

    def __init__(out self, *dims: Int):
        self._shape = TensorShape(dims)
        self._refcount = alloc[Atomic[DType.uint64]](1)
        self._refcount[] = Atomic[DType.uint64](1)
        if self._shape.num_elements() == 0:
            self._data = UnsafePointer[
                Scalar[Self.dtype], MutUntrackedOrigin
            ].unsafe_dangling()
        else:
            self._data = alloc[Scalar[Self.dtype]](self._shape.num_elements())
            memset_zero(self._data, self._shape.num_elements())

    def __init__(out self, var shape: TensorShape):
        self._shape = shape
        self._refcount = alloc[Atomic[DType.uint64]](1)
        self._refcount[] = Atomic[DType.uint64](1)
        if shape.num_elements() == 0:
            self._data = UnsafePointer[
                Scalar[Self.dtype], MutUntrackedOrigin
            ].unsafe_dangling()
        else:
            self._data = alloc[Scalar[Self.dtype]](shape.num_elements())
            memset_zero(self._data, shape.num_elements())

    def __init__(out self, shapes: VariadicList[Int, _]):
        self._shape = TensorShape(shapes)
        self._refcount = alloc[Atomic[DType.uint64]](1)
        self._refcount[] = Atomic[DType.uint64](1)
        if self._shape.num_elements() == 0:
            self._data = UnsafePointer[
                Scalar[Self.dtype], MutUntrackedOrigin
            ].unsafe_dangling()
        else:
            self._data = alloc[Scalar[Self.dtype]](self._shape.num_elements())
            memset_zero(self._data, self._shape.num_elements())

    def __init__[
        origin: MutOrigin
    ](
        out self,
        var data: UnsafePointer[Scalar[Self.dtype], origin],
        var shape: TensorShape,
    ):
        self._shape = shape
        self._refcount = alloc[Atomic[DType.uint64]](1)
        self._refcount[] = Atomic[DType.uint64](1)

        if shape.num_elements() == 0:
            self._data = UnsafePointer[
                Scalar[Self.dtype], MutUntrackedOrigin
            ].unsafe_dangling()
        else:
            self._data = alloc[Scalar[Self.dtype]](shape.num_elements())
            memcpy(dest=self._data, src=data, count=self._shape.num_elements())
        _ = data

    def __init__(out self, *, deinit take: Tensor[Self.dtype]):
        self._data = take._data
        self._refcount = take._refcount
        self._shape = take._shape

    def __init__(out self, *, copy: Tensor[Self.dtype]):
        self._shape = copy._shape
        self._refcount = alloc[Atomic[DType.uint64]](1)
        self._refcount[] = Atomic[DType.uint64](1)
        if copy.num_elements() == 0:
            self._data = UnsafePointer[
                Scalar[Self.dtype], MutUntrackedOrigin
            ].unsafe_dangling()
        else:
            self._data = alloc[Scalar[Self.dtype]](copy.num_elements())
            memcpy(dest=self._data, src=copy._data, count=copy.num_elements())

    def __init__(
        out self,
        *,
        data: UnsafePointer[Scalar[Self.dtype], MutUntrackedOrigin],
        refcount: UnsafePointer[Atomic[DType.uint64], MutUntrackedOrigin],
        shape: TensorShape,
    ):
        self._data = data
        self._refcount = refcount
        self._shape = shape

    def share(self) -> Self:
        _ = self._refcount[].fetch_add[ordering=Ordering.RELAXED](1)
        var result = Self(
            data=self._data, refcount=self._refcount, shape=self._shape
        )
        return result^

    @always_inline("nodebug")
    def __getitem__(self, index: Int) -> Scalar[Self.dtype]:
        return self._data[index]

    @always_inline("nodebug")
    def __setitem__(self, index: Int, value: Scalar[Self.dtype]):
        self._data[index] = value

    @always_inline("nodebug")
    def ptr(self) -> UnsafePointer[Scalar[Self.dtype], origin_of(self)]:
        """
        Returns a read-only pointer to the tensor's underlying buffer,
        with its origin tied to `self`.

        Returns:
            A pointer to the tensor's data, valid for the lifetime of `self`.
        """
        return self._data.mut_cast[False]().unsafe_origin_cast[origin_of(self)]()

    @always_inline("nodebug")
    def mut_ptr(mut self) -> UnsafePointer[Scalar[Self.dtype], origin_of(self)]:
        """
        Returns a mutable pointer to the tensor's underlying buffer, with
        its origin tied to `self`.

        Returns:
            A pointer to the tensor's data, valid for the lifetime of `self`.
        """
        return self._data.unsafe_origin_cast[origin_of(self)]()

    @always_inline("nodebug")
    def shape(self) -> TensorShape:
        return self._shape

    @always_inline("nodebug")
    def load[simd_width: Int](self, index: Int) -> SIMD[Self.dtype, simd_width]:
        return self._data.load[width=simd_width](index)

    @always_inline("nodebug")
    def store[
        simd_width: Int
    ](self, index: Int, value: SIMD[Self.dtype, simd_width]):
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
        if self._refcount[].fetch_sub[ordering=Ordering.RELEASE](1) != 1:
            return
        fence[ordering=Ordering.ACQUIRE]()
        if self.num_elements() > 0:
            self._data.free()
        self._refcount.free()

    def write_to(self, mut writer: Some[Writer]):
        """Writes the tensor to a writer.

        Args:
            writer: The writer to write the tensor to.
        """
        writer.write(self.__str__())

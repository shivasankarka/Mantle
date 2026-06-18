# ===----------------------------------------------------------------------=== #
# Mantle: Tensor
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Tensor (mantle.core.tensor)
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
    """
    Represents the shape of a tensor.

    Stores the rank and dimension sizes with a maximum rank of MAX_RANK.
    """
    var _rank: Int
    """The number of dimensions."""
    var _shape: IndexList[MAX_RANK]
    """The size of each dimension."""

    def __init__(out self, *shape: Int):
        """
        Create a TensorShape from variadic dimension sizes.

        Args:
            shape: The size of each dimension.
        """
        self._rank = len(shape)
        self._shape = IndexList[MAX_RANK]()
        for i in range(min(self._rank, MAX_RANK)):
            self._shape[i] = shape[i]

    def __init__(out self, shapes: VariadicList[Int, _]):
        """
        Create a TensorShape from a variadic list.

        Args:
            shapes: A variadic list of dimension sizes.
        """
        self._rank = len(shapes)
        self._shape = IndexList[MAX_RANK]()
        for i in range(min(self._rank, MAX_RANK)):
            self._shape[i] = shapes[i]

    def __init__(out self, shape: List[Int]):
        """
        Create a TensorShape from a List.

        Args:
            shape: A list of dimension sizes.
        """
        self._rank = len(shape)
        self._shape = IndexList[MAX_RANK]()
        for i in range(min(self._rank, MAX_RANK)):
            self._shape[i] = shape[i]

    def __init__[num: Int](out self, shape: IndexList[num]):
        """
        Create a TensorShape from an IndexList.

        Parameters:
            num: The number of dimensions.

        Args:
            shape: An IndexList of dimension sizes.
        """
        self._rank = num
        self._shape = IndexList[MAX_RANK]()
        for i in range(min(self._rank, MAX_RANK)):
            self._shape[i] = shape[i]

    def __init__(out self, rank: Int, shape: IndexList[MAX_RANK]):
        """
        Create a TensorShape from a rank and IndexList.

        Args:
            rank: The number of dimensions.
            shape: An IndexList of dimension sizes.
        """
        self._rank = rank
        self._shape = shape

    @always_inline("nodebug")
    def __getitem__(self, index: Int) -> Int:
        """
        Get the size of a dimension.

        Args:
            index: The dimension index (supports negative indexing).

        Returns:
            The size of the dimension at the given index.
        """
        return self._shape[index if index >= 0 else self._rank + index]

    @always_inline("nodebug")
    def __setitem__(mut self, index: Int, value: Int):
        """
        Set the size of a dimension.

        Args:
            index: The dimension index (supports negative indexing).
            value: The new size.
        """
        self._shape[index if index >= 0 else self._rank + index] = value

    @always_inline("nodebug")
    def rank(self) -> Int:
        """
        Returns:
            The number of dimensions.
        """
        return self._rank

    def num_elements(self) -> Int:
        """
        Returns:
            The total number of elements (product of all dimension sizes).
        """
        var result = 1
        for i in range(self._rank):
            result *= self._shape[i]
        return result

    def strides(self) -> IndexList[MAX_RANK]:
        """
        Compute the stride for each dimension (row-major order).

        Returns:
            An IndexList where each entry is the number of elements
            between consecutive elements along that dimension.
        """
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
        """
        Check shape equality.

        Args:
            other: The other TensorShape.

        Returns:
            True if both shapes have the same rank and dimension sizes.
        """
        if self.rank() != other.rank():
            return False
        for i in range(self.rank()):
            if self[i] != other[i]:
                return False
        return True

    @always_inline("nodebug")
    def __ne__(self, other: TensorShape) -> Bool:
        """
        Check shape inequality.

        Args:
            other: The other TensorShape.

        Returns:
            True if shapes differ.
        """
        return not self.__eq__(other)

    def __contains__(self, value: Int) -> Bool:
        """
        Check if a dimension size exists in the shape.

        Args:
            value: The dimension size to search for.

        Returns:
            True if any dimension has the given size.
        """
        for i in range(self.rank()):
            if self[i] == value:
                return True
        return False

    def to_list(self) -> List[Int]:
        """
        Convert the shape to a List.

        Returns:
            A List containing the dimension sizes.
        """
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
    """
    A reference-counted multi-dimensional array.

    Parameters:
        dtype: The data type of the tensor elements.
    """
    var _data: UnsafePointer[Scalar[Self.dtype], MutUntrackedOrigin]
    """Pointer to the underlying data buffer."""
    var _refcount: UnsafePointer[Atomic[DType.uint64], MutUntrackedOrigin]
    """Pointer to the atomic reference count."""
    var _shape: TensorShape
    """The shape of the tensor."""

    def __init__(out self, *dims: Int):
        """
        Create a zero-initialized tensor with the given dimension sizes.

        Args:
            dims: The size of each dimension.
        """
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
        """
        Create a zero-initialized tensor with the given shape.

        Args:
            shape: The shape of the tensor.
        """
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
        """
        Create a zero-initialized tensor from a variadic list of dimension sizes.

        Args:
            shapes: A variadic list of dimension sizes.
        """
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
        """
        Create a tensor by copying data from an external pointer.

        Parameters:
            origin: The mutability origin of the source pointer.

        Args:
            data: Pointer to the source data.
            shape: The shape of the tensor.
        """
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
        """
        Move constructor: take ownership of another tensor's data.

        Args:
            take: The tensor to take ownership from.
        """
        self._data = take._data
        self._refcount = take._refcount
        self._shape = take._shape

    def __init__(out self, *, copy: Tensor[Self.dtype]):
        """
        Copy constructor: deep copy of another tensor.

        Args:
            copy: The tensor to copy.
        """
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
        """
        Initialize a tensor from raw components (unsafe).

        Args:
            data: Pointer to the data buffer.
            refcount: Pointer to the reference count.
            shape: The shape of the tensor.
        """
        self._data = data
        self._refcount = refcount
        self._shape = shape

    def share(self) -> Self:
        """
        Create a shallow copy with an incremented reference count.

        Returns:
            A new Tensor that shares the same underlying data buffer.
        """
        _ = self._refcount[].fetch_add[ordering=Ordering.RELAXED](1)
        var result = Self(
            data=self._data, refcount=self._refcount, shape=self._shape
        )
        return result^

    @always_inline("nodebug")
    def __getitem__(self, index: Int) -> Scalar[Self.dtype]:
        """
        Access a single element by flat index.

        Args:
            index: The flat index into the tensor data.

        Returns:
            The element at the given index.
        """
        return self._data[index]

    @always_inline("nodebug")
    def __setitem__(self, index: Int, value: Scalar[Self.dtype]):
        """
        Set a single element by flat index.

        Args:
            index: The flat index into the tensor data.
            value: The value to set.
        """
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
        """
        Returns:
            The shape of the tensor.
        """
        return self._shape

    @always_inline("nodebug")
    def load[simd_width: Int](self, index: Int) -> SIMD[Self.dtype, simd_width]:
        """
        Load a SIMD vector from the given flat index.

        Parameters:
            simd_width: The SIMD vector width.

        Args:
            index: The flat index to load from.

        Returns:
            A SIMD vector of elements starting at the given index.
        """
        return self._data.load[width=simd_width](index)

    @always_inline("nodebug")
    def store[
        simd_width: Int
    ](self, index: Int, value: SIMD[Self.dtype, simd_width]):
        """
        Store a SIMD vector at the given flat index.

        Parameters:
            simd_width: The SIMD vector width.

        Args:
            index: The flat index to store at.
            value: The SIMD vector to store.
        """
        self._data.store(index, value)

    @always_inline("nodebug")
    def strides(self) -> IndexList[MAX_RANK]:
        """
        Returns:
            The stride for each dimension in row-major order.
        """
        return self._shape.strides()

    @always_inline("nodebug")
    def rank(self) -> Int:
        """
        Returns:
            The number of dimensions.
        """
        return self._shape.rank()

    @always_inline("nodebug")
    def num_elements(self) -> Int:
        """
        Returns:
            The total number of elements.
        """
        return self._shape.num_elements()

    @always_inline("nodebug")
    def dim(self, index: Int) -> Int:
        """
        Get the size of a specific dimension.

        Args:
            index: The dimension index.

        Returns:
            The size of the dimension at the given index.
        """
        return self._shape[index]

    @always_inline("nodebug")
    def zero(self):
        """Set all elements to zero."""
        memset_zero(self._data, self.num_elements())

    @always_inline("nodebug")
    def ireshape(mut self, new_shape: TensorShape) raises:
        """
        In-place reshape of the tensor.

        Args:
            new_shape: The new shape (must have the same number of elements).
        """
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
        """
        Decrement the reference count and free memory if it reaches zero.
        """
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

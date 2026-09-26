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
from std.collections.optional import Optional
from std.utils.index import IndexList
from std.memory import unsafe_memset_zero, unsafe_memcpy, Pointer
from std.os import abort
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.ffi import _Global

from std.sys.info import simd_width_of

from mantle.core.device import Device

comptime MAX_RANK = 8
"""Max rank of a tensor."""
# TODO: make it an explicit input to Tensor


def _make_shared_device_context() -> DeviceContext:
    try:
        return DeviceContext()
    except e:
        abort("Tensor: shared DeviceContext creation failed: " + String(e))


comptime _shared_device_context_global = _Global[
    "mantle_tensor_shared_device_context", _make_shared_device_context
]
"""Every `Tensor` construction needs a `DeviceContext` to allocate its
buffer, but `DeviceContext()` creates a brand-new native context (a Metal
command queue on Apple GPUs) on every call, cheap to *copy* (refcounted)
but not to create. So we use a global `_Global` to ensure that the
process keeps exactly one, created lazily on first use, and every `Tensor`
just takes a cheap refcounted copy of it."""


def _shared_device_context() raises -> DeviceContext:
    return _shared_device_context_global.get_or_create_ptr()[
        unsafe_offset=0
    ].copy()


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


struct Tensor[dtype: DType, device: Device = Device.cpu](
    Copyable, Movable, Writable
):
    """
    A multi-dimensional array.

    Parameters:
        dtype: The data type of the tensor elements.
        device: The device the tensor's data lives on. Storage is a MAX
            `HostBuffer` (CPU) or `DeviceBuffer` (GPU), each already
            refcounted by the driver — `Tensor` does no refcounting of its
            own.
    """

    var _shape: TensorShape
    """The shape of the tensor."""
    var _host_buffer: Optional[HostBuffer[Self.dtype]]
    """The CPU data buffer. `None` for GPU tensors."""
    var _device_buffer: Optional[DeviceBuffer[Self.dtype]]
    """The GPU data buffer. `None` for CPU tensors."""

    def __init__(out self, *dims: Int):
        """
        Create a zero-initialized tensor with the given dimension sizes.

        Args:
            dims: The size of each dimension.
        """
        self = Self(TensorShape(dims))

    def __init__(out self, var shape: TensorShape):
        """
        Create a zero-initialized tensor with the given shape.

        Args:
            shape: The shape of the tensor.
        """
        self = Self(shape, uninitialized=False)

    def __init__(out self, var shape: TensorShape, *, uninitialized: Bool):
        """
        Allocate a tensor with the given shape, optionally skipping the
        zero-fill.

        `uninitialized=True` is only safe when the caller is about to fully
        overwrite every element (e.g. a kernel's or matmul's write-only
        output) — the zero-fill `enqueue_fill` performs is itself a real
        GPU kernel launch/memory write, not a free formality, so skipping it
        for write-only temporaries avoids doing that work twice.

        Args:
            shape: The shape of the tensor.
            uninitialized: If True, skip zero-filling the buffer.
        """
        self._shape = shape
        self._host_buffer = None
        self._device_buffer = None
        try:
            var ctx = _shared_device_context()
            comptime if Self.device.id == Device.cpu.id:
                var buf = ctx.enqueue_create_host_buffer[Self.dtype](
                    shape.num_elements()
                )
                if not uninitialized:
                    buf.enqueue_fill(Scalar[Self.dtype](0))
                self._host_buffer = buf
            else:
                var buf = ctx.enqueue_create_buffer[Self.dtype](
                    shape.num_elements()
                )
                if not uninitialized:
                    buf.enqueue_fill(Scalar[Self.dtype](0))
                self._device_buffer = buf
        except e:
            abort("Tensor: allocation failed: " + String(e))

    def __init__(out self, shapes: VariadicList[Int, _]):
        """
        Create a zero-initialized tensor from a variadic list of dimension sizes.

        Args:
            shapes: A variadic list of dimension sizes.
        """
        self = Self(TensorShape(shapes))

    def __init__[
        origin: MutOrigin
    ](
        out self,
        var data: Pointer[Scalar[Self.dtype], origin],
        var shape: TensorShape,
    ) where (Self.device.id == Device.cpu.id):
        """
        Create a tensor by copying data from an external CPU pointer.

        Parameters:
            origin: The mutability origin of the source pointer.

        Args:
            data: Pointer to the source data.
            shape: The shape of the tensor.
        """
        self = Self(shape)
        if shape.num_elements() > 0:
            unsafe_memcpy(
                dest=self._host_buffer.value().unsafe_ptr(),
                src=data,
                count=shape.num_elements(),
            )
        _ = data

    def __init__(out self, *, deinit move: Tensor[Self.dtype, Self.device]):
        """
        Move constructor: take ownership of another tensor's data.

        Args:
            move: The tensor to take ownership from.
        """
        self._shape = move._shape
        self._host_buffer = move._host_buffer^
        self._device_buffer = move._device_buffer^

    def __init__(out self, *, copy: Tensor[Self.dtype, Self.device]):
        """
        Copy constructor: deep copy of another tensor.

        Args:
            copy: The tensor to copy.
        """
        self = Self(copy._shape)
        if copy.num_elements() == 0:
            return
        # A copy constructor may not raise, so a failed device copy aborts
        # instead of propagating — same convention as `HostBuffer`'s and
        # `DeviceBuffer`'s own (non-raising) copy constructors.
        try:
            comptime if Self.device.id == Device.cpu.id:
                unsafe_memcpy(
                    dest=self._host_buffer.value().unsafe_ptr(),
                    src=copy._host_buffer.value().unsafe_ptr(),
                    count=copy.num_elements(),
                )
            else:
                var src = copy._device_buffer.value()
                src.enqueue_copy_to(self._device_buffer.value())
                src.context().synchronize()
        except e:
            abort("Tensor: deep copy failed: " + String(e))

    def copy_from(mut self, source: Self) raises:
        """Copy ``source`` into this tensor's existing storage.

        Copies `self.num_elements()` elements — the destination's own size,
        not `source`'s. Callers must guarantee `source` has exactly this
        tensor's shape; nothing here validates it (same invariant
        `Collection.__setitem__`'s CPU branch already relies on). This holds
        for every current caller because the graph is fully static: each
        `Symbol`'s slot is always written with a tensor of that symbol's
        fixed, comptime-known shape. A mismatched `source` would silently
        under/over-read instead of erroring — don't reuse this for anything
        where that invariant isn't guaranteed by construction.

        GPU copies are deliberately enqueued without synchronizing.  Commands
        submitted to one DeviceContext are ordered, so consumers on that
        context see the copy before their kernels run; callers that need host
        visibility must synchronize explicitly at their boundary.  Keeping
        the destination allocation is particularly important for graph inputs
        and gradients, which are overwritten on every training step.
        """
        comptime if Self.device.id == Device.cpu.id:
            comptime assert Self.device.id == Device.cpu.id
            unsafe_memcpy(
                dest=self._host_buffer.value().unsafe_ptr(),
                src=source._host_buffer.value().unsafe_ptr(),
                count=self.num_elements(),
            )
        else:
            comptime assert Self.device.id == Device.gpu.id
            var src = source._device_buffer.value()
            src.enqueue_copy_to(self._device_buffer.value())

    def __init__(
        out self, *, var host_buffer: HostBuffer[Self.dtype], shape: TensorShape
    ) where Self.device.id == Device.cpu.id:
        """
        Initialize a CPU tensor by wrapping an existing `HostBuffer`.

        Args:
            host_buffer: The CPU buffer backing this tensor.
            shape: The shape of the tensor.
        """
        self._shape = shape
        self._host_buffer = host_buffer^
        self._device_buffer = None

    def __init__(
        out self,
        *,
        var device_buffer: DeviceBuffer[Self.dtype],
        shape: TensorShape,
    ) where Self.device.id == Device.gpu.id:
        """
        Initialize a GPU tensor by wrapping an existing `DeviceBuffer`.

        Args:
            device_buffer: The GPU buffer backing this tensor.
            shape: The shape of the tensor.
        """
        self._shape = shape
        self._host_buffer = None
        self._device_buffer = device_buffer^

    def share(self) -> Self:
        """
        Create a shallow copy sharing the same underlying data buffer — the
        driver-native refcount on the `HostBuffer`/`DeviceBuffer` is
        incremented, no data is copied.

        Returns:
            A new Tensor that shares the same underlying data buffer.
        """
        comptime if Self.device.id == Device.cpu.id:
            comptime assert Self.device.id == Device.cpu.id
            return Self(
                host_buffer=self._host_buffer.value().copy(),
                shape=self._shape,
            )
        else:
            comptime assert Self.device.id == Device.gpu.id
            return Self(
                device_buffer=self._device_buffer.value().copy(),
                shape=self._shape,
            )

    def to_host(self) raises -> Tensor[Self.dtype, Device.cpu]:
        """
        Copy this tensor's data to a new CPU-resident tensor.

        Returns:
            A new `Tensor[dtype, Device.cpu]` with the same data.
        """
        var out = Tensor[Self.dtype, Device.cpu](self._shape)
        comptime if Self.device.id == Device.cpu.id:
            comptime assert Self.device.id == Device.cpu.id
            unsafe_memcpy(
                dest=out._host_buffer.value().unsafe_ptr(),
                src=self._host_buffer.value().unsafe_ptr(),
                count=self.num_elements(),
            )
        else:
            var buf = self._device_buffer.value()
            buf.enqueue_copy_to(out._host_buffer.value())
            buf.context().synchronize()
        return out^

    def to_gpu(self) raises -> Tensor[Self.dtype, Device.gpu]:
        """
        Copy this tensor's data to a new GPU-resident tensor.

        Returns:
            A new `Tensor[dtype, Device.gpu]` with the same data.
        """
        var out = Tensor[Self.dtype, Device.gpu](self._shape)
        comptime if Self.device.id == Device.cpu.id:
            comptime assert Self.device.id == Device.cpu.id
            var buf = out._device_buffer.value()
            buf.enqueue_copy_from(self._host_buffer.value())
            buf.context().synchronize()
        else:
            var src = self._device_buffer.value()
            src.enqueue_copy_to(out._device_buffer.value())
            src.context().synchronize()
        return out^

    @always_inline("nodebug")
    def __getitem__(
        self, index: Int
    ) -> Scalar[Self.dtype] where Self.device.id == Device.cpu.id:
        """
        Access a single element by flat index.

        Args:
            index: The flat index into the tensor data.

        Returns:
            The element at the given index.
        """
        return self._host_buffer.value()[index]

    @always_inline("nodebug")
    def __setitem__(
        self, index: Int, value: Scalar[Self.dtype]
    ) where Self.device.id == Device.cpu.id:
        """
        Set a single element by flat index.

        Args:
            index: The flat index into the tensor data.
            value: The value to set.
        """
        self._host_buffer.value()[index] = value

    @always_inline("nodebug")
    def ptr[
        o: Origin
    ](
        ref[o] self,
    ) -> Pointer[Scalar[Self.dtype], o] where (
        Self.device.id == Device.cpu.id
    ):
        """
        Returns a pointer to the tensor's underlying buffer, with its
        mutability tied to how `self` is referenced — immutable for a
        borrowed `self`, mutable for a `mut self`.

        Returns:
            A pointer to the tensor's data, valid for the lifetime of `self`.
        """
        return (
            self._host_buffer.value()
            .unsafe_ptr()
            .unsafe_mut_cast[o.mut]()
            .unsafe_origin_cast[o]()
        )

    @always_inline("nodebug")
    def shape(self) -> TensorShape:
        """
        Returns:
            The shape of the tensor.
        """
        return self._shape

    @always_inline("nodebug")
    def load[
        simd_width: Int
    ](self, index: Int) -> SIMD[Self.dtype, simd_width] where (
        Self.device.id == Device.cpu.id
    ):
        """
        Load a SIMD vector from the given flat index.

        Parameters:
            simd_width: The SIMD vector width.

        Args:
            index: The flat index to load from.

        Returns:
            A SIMD vector of elements starting at the given index.
        """
        return (
            self._host_buffer.value()
            .unsafe_ptr()
            .unsafe_load[width=simd_width](index)
        )

    @always_inline("nodebug")
    def store[
        simd_width: Int
    ](self, index: Int, value: SIMD[Self.dtype, simd_width]) where (
        Self.device.id == Device.cpu.id
    ):
        """
        Store a SIMD vector at the given flat index.

        Parameters:
            simd_width: The SIMD vector width.

        Args:
            index: The flat index to store at.
            value: The SIMD vector to store.
        """
        self._host_buffer.value().unsafe_ptr().unsafe_store(index, value)

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
    def zero(self) where Self.device.id == Device.cpu.id:
        """Set all elements to zero."""
        unsafe_memset_zero(
            self._host_buffer.value().unsafe_ptr(), self.num_elements()
        )

    @always_inline("nodebug")
    def gpu_ptr[
        o: Origin
    ](
        ref[o] self,
    ) -> Pointer[Scalar[Self.dtype], o] where (
        Self.device.id == Device.gpu.id
    ):
        """
        Returns a raw pointer to the tensor's GPU-resident buffer, suitable
        for passing into a `DeviceContext.enqueue_function` kernel launch.

        Returns:
            A pointer to the tensor's device data.
        """
        return self._device_buffer.value().unsafe_ptr().unsafe_origin_cast[o]()

    @always_inline("nodebug")
    def gpu_context(
        self,
    ) raises -> DeviceContext where Self.device.id == Device.gpu.id:
        """
        Returns:
            The `DeviceContext` that owns this tensor's GPU buffer, for
            launching kernels against it or synchronizing.
        """
        return self._device_buffer.value().context()

    def fill(mut self, value: Scalar[Self.dtype]):
        """Set every element to `value` — works on both devices, unlike
        `zero()` which is CPU-only (kept separate since `zero()` predates
        this and its callers rely on it not raising)."""
        try:
            comptime if Self.device.id == Device.cpu.id:
                comptime assert Self.device.id == Device.cpu.id

                comptime nelts = 2 * simd_width_of[Self.dtype]()

                def vec_fill[nelts: Int](i: Int) {mut self, imm value}:
                    self.store[nelts](i, value)

                vectorize[nelts](self.num_elements(), vec_fill)
            else:
                comptime assert Self.device.id == Device.gpu.id
                var buf = self._device_buffer.value()
                buf.enqueue_fill(value)
                buf.context().synchronize()
        except e:
            abort("Tensor: fill failed: " + String(e))

    def enqueue_fill(
        mut self, value: Scalar[Self.dtype]
    ) raises where Self.device.id == Device.gpu.id:
        """Queue a GPU fill without forcing completion on the host."""
        self._device_buffer.value().enqueue_fill(value)

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
        comptime if Self.device.id == Device.cpu.id:
            comptime assert Self.device.id == Device.cpu.id
            var s: String = "["
            for i in range(self.num_elements()):
                s += String(self[i])
                if i < self.num_elements() - 1:
                    s += ", "
            return s + "]"
        else:
            return "Tensor[gpu, " + String(self.num_elements()) + " elements]"

    def write_to(self, mut writer: Some[Writer]):
        """Writes the tensor to a writer.

        Args:
            writer: The writer to write the tensor to.
        """
        writer.write(self.__str__())

# ===----------------------------------------------------------------------=== #
# Mantle: CpuBuffer
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""CpuBuffer (mantle.core.host_buffer)
------------------------------------------------
Reference-counted heap buffer backing CPU `Tensor`s.

`Tensor` used to allocate its CPU storage through MAX's `DeviceContext`
(`enqueue_create_host_buffer`) — a queue-based API meant for coordinating
with GPU work. That costs a fixed ~40-50us per call regardless of size, which
dwarfs compute for the many small, short-lived tensors most CPU training
steps allocate. `CpuBuffer` allocates with a plain heap allocation instead
(~1ns), and tracks its own atomic refcount so `Tensor.share()` keeps its
existing shared-ownership semantics (see `Collection.__getitem__` and
`DataLoader.__iter__`, both of which alias a tensor's buffer across
independent lifetimes).
"""
from std.atomic import Atomic, fence, Ordering
from std.memory import Pointer, unsafe_memcpy
from std.memory.alloc import unsafe_alloc


struct CpuBuffer[dtype: DType](Movable):
    """A reference-counted, heap-allocated buffer of `dtype` elements.

    Always owns its allocation (unlike NuMojo's more general
    `DataContainer`, this never wraps external/unmanaged memory — every
    `Tensor` CPU buffer is allocated by this type). `.share()` is the only
    way to get a second handle to the same allocation; ordinary moves
    transfer ownership without touching the refcount.

    Notes:
        Deliberately not `Copyable`: an implicit copy that silently bumped
        the refcount would be easy to lose track of. Callers that want a
        second handle to the same buffer must call `.share()`; callers that
        want an independent buffer allocate a new `CpuBuffer` and
        `unsafe_memcpy` into it (see `Tensor`'s own copy constructor).
    """

    comptime origin = MutUntrackedOrigin

    var _ptr: Pointer[Scalar[Self.dtype], Self.origin]
    var _refcount: Pointer[Atomic[UInt64], Self.origin]
    var size: Int
    """Number of elements in the buffer."""

    def __init__(out self, size: Int):
        """Allocate a new, uniquely-owned buffer of `size` elements."""
        self.size = size
        self._refcount = unsafe_alloc[Atomic[UInt64]](1)
        self._refcount[] = Atomic[UInt64](1)
        if size == 0:
            self._ptr = Pointer[
                Scalar[Self.dtype], Self.origin
            ].unsafe_dangling()
        else:
            self._ptr = unsafe_alloc[Scalar[Self.dtype]](size)

    def __init__(out self, *, deinit move: Self):
        """Move constructor: transfers ownership without touching the
        refcount."""
        self._ptr = move._ptr
        self._refcount = move._refcount
        self.size = move.size

    def __init__(
        out self,
        *,
        ptr: Pointer[Scalar[Self.dtype], Self.origin],
        refcount: Pointer[Atomic[UInt64], Self.origin],
        size: Int,
    ):
        """Wrap an existing allocation and refcount without allocating.

        Internal to `share()` — the caller must have already incremented
        `refcount` for this new handle.
        """
        self._ptr = ptr
        self._refcount = refcount
        self.size = size

    def __deinit__(deinit self):
        """Decrement the refcount; free the buffer if this was the last
        handle."""
        if self._refcount[].fetch_sub[ordering=Ordering.RELEASE](1) != 1:
            return
        fence[ordering=Ordering.ACQUIRE]()
        if self.size > 0:
            self._ptr.unsafe_free()
        self._refcount.unsafe_free()

    def share(self) -> Self:
        """Return a new handle to this same allocation, incrementing the
        refcount. Both handles observe the same underlying memory."""
        _ = self._refcount[].fetch_add[ordering=Ordering.RELAXED](1)
        return Self(ptr=self._ptr, refcount=self._refcount, size=self.size)

    @always_inline("nodebug")
    def unsafe_ptr[
        o: Origin, //
    ](ref[o] self) -> Pointer[Scalar[Self.dtype], o]:
        """Returns a pointer to the buffer's data, with its mutability and
        origin tied to how the caller holds `self` — mutable for a `mut`
        borrow, immutable for a read-only one — so the compiler keeps this
        handle (and therefore the allocation) alive for as long as the
        returned pointer is in use.

        Use this when the returned pointer can outlive this single call
        (e.g. stored in a local and used across later statements, like
        `Tensor.ptr()` does) — that's the case where an untracked origin
        would let the compiler free `self` while the caller still holds a
        now-dangling pointer. For an access fully contained in one call
        (load/store/element access), `unsafe_mut_ptr` is simpler and
        matches the rest of the codebase's convention for this pattern.
        """
        return self._ptr.unsafe_mut_cast[o.mut]().unsafe_origin_cast[o]()

    @always_inline("nodebug")
    def unsafe_mut_ptr(self) -> Pointer[Scalar[Self.dtype], MutAnyOrigin]:
        """Returns an always-mutable pointer to the buffer's data, with an
        untracked (`MutAnyOrigin`) lifetime — safe only when the pointer is
        used and discarded within the same call, on a `self` that something
        else (the caller, or a temporary's enclosing statement) is already
        keeping alive for that call's duration. Matches the
        `unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()` idiom
        already used for this exact purpose elsewhere (see
        `mantle/autograd/ops/matmul.mojo`).
        """
        return self._ptr.unsafe_mut_cast[True]().unsafe_origin_cast[
            MutAnyOrigin
        ]()

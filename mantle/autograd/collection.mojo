# ===----------------------------------------------------------------------=== #
# Mantle: Collection
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Collection (mantle.autograd.collection)
------------------------------------------------
Symbol-keyed tensor arena for dense O(1) tensor storage and lookup by symbol id.
"""
from std.collections.optional import Optional
from std.memory.unsafe_pointer import Pointer
from std.memory import unsafe_memset_zero, unsafe_memcpy
from std.memory.alloc import unsafe_alloc

from mantle import f32
from mantle.autograd.symbol import Symbol
from mantle.core.device import Device
from mantle.core.tensor import Tensor


# ===----------------------------------------------------------------------===#
# Collection
# ===----------------------------------------------------------------------===#


struct Collection[device: Device = Device.cpu](Copyable, Movable, Sized):
    """
    Symbol-keyed tensor arena.

    Tensors are stored densely in insertion order (so sparse subsets of the
    global symbol space, e.g. only the trainable parameters, don't waste
    memory). A separate `index_map` array, sized by the largest symbol id
    seen so far, maps `Symbol.name -> dense slot` for O(1) lookup without
    scanning.

    Parameters:
        device: The device the stored tensors' data lives on (compile-time
            tag; only `Device.cpu` is implemented today).
    """

    var size: Int
    var capacity: Int
    var data_owner: Optional[
        Pointer[Tensor[f32, Self.device], MutUntrackedOrigin]
    ]
    var symbols_owner: Optional[Pointer[UInt32, MutUntrackedOrigin]]
    var data_ref: Pointer[Tensor[f32, Self.device], MutUntrackedOrigin]
    var symbols_ref: Pointer[UInt32, MutUntrackedOrigin]

    var index_map_capacity: Int
    var index_map_owner: Optional[Pointer[Int, MutUntrackedOrigin]]
    var index_map_ref: Pointer[Int, MutUntrackedOrigin]

    @always_inline("nodebug")
    def __init__(out self, *, capacity: Int = 1):
        self.size = 0
        self.capacity = capacity
        self.data_owner = unsafe_alloc[Tensor[f32, Self.device]](capacity)
        self.symbols_owner = unsafe_alloc[UInt32](capacity)
        self.data_ref = self.data_owner.value()
        self.symbols_ref = self.symbols_owner.value()

        self.index_map_capacity = 0
        self.index_map_owner = None
        self.index_map_ref = Pointer[Int, MutUntrackedOrigin].unsafe_dangling()

    @always_inline("nodebug")
    def __init__(out self, *, deinit move: Self):
        self.size = move.size
        self.capacity = move.capacity
        self.data_owner = move.data_owner^
        self.symbols_owner = move.symbols_owner^
        self.data_ref = self.data_owner.value()
        self.symbols_ref = self.symbols_owner.value()

        self.index_map_capacity = move.index_map_capacity
        self.index_map_owner = move.index_map_owner^
        self.index_map_ref = (
            self.index_map_owner.value() if self.index_map_owner else Pointer[
                Int, MutUntrackedOrigin
            ].unsafe_dangling()
        )

    @always_inline("nodebug")
    def __init__(out self, *, copy: Self):
        self.size = copy.size
        self.capacity = copy.capacity
        self.data_owner = unsafe_alloc[Tensor[f32, Self.device]](copy.capacity)
        self.symbols_owner = unsafe_alloc[UInt32](copy.capacity)
        self.data_ref = self.data_owner.value()
        self.symbols_ref = self.symbols_owner.value()

        for i in range(copy.size):
            (self.data_ref.unsafe_offset(i)).unsafe_write(
                copy.data_ref[unsafe_offset=i].copy()
            )
            self.symbols_ref[unsafe_offset=i] = copy.symbols_ref[
                unsafe_offset=i
            ]

        self.index_map_capacity = copy.index_map_capacity
        if copy.index_map_owner:
            var new_map = unsafe_alloc[Int](copy.index_map_capacity)
            unsafe_memcpy(
                dest=new_map,
                src=copy.index_map_owner.value(),
                count=copy.index_map_capacity,
            )
            self.index_map_owner = new_map
            self.index_map_ref = new_map
        else:
            self.index_map_owner = None
            self.index_map_ref = Pointer[
                Int, MutUntrackedOrigin
            ].unsafe_dangling()

    @always_inline("nodebug")
    def __deinit__(deinit self):
        if self.data_owner:
            var data = self.data_owner.value()
            for i in range(self.size):
                (data.unsafe_offset(i)).unsafe_deinit_pointee()
            data.unsafe_free()
        if self.symbols_owner:
            self.symbols_owner.value().unsafe_free()
        if self.index_map_owner:
            self.index_map_owner.value().unsafe_free()

    @always_inline("nodebug")
    def __len__(self) -> Int:
        return self.size

    @always_inline("nodebug")
    def _realloc(mut self, new_capacity: Int):
        var new_data = unsafe_alloc[Tensor[f32, Self.device]](new_capacity)
        var new_symbols = unsafe_alloc[UInt32](new_capacity)

        for i in range(self.size):
            (new_data.unsafe_offset(i)).unsafe_write(
                (self.data_ref.unsafe_offset(i)).unsafe_take_pointee()
            )
            new_symbols[unsafe_offset=i] = self.symbols_ref[unsafe_offset=i]

        if self.data_owner:
            self.data_owner.value().unsafe_free()
        if self.symbols_owner:
            self.symbols_owner.value().unsafe_free()

        self.data_owner = new_data
        self.symbols_owner = new_symbols
        self.data_ref = new_data
        self.symbols_ref = new_symbols
        self.capacity = new_capacity

    @always_inline("nodebug")
    def _ensure_index_map(mut self, min_capacity: Int):
        if min_capacity <= self.index_map_capacity:
            return

        var new_capacity = max(
            min_capacity, max(1, self.index_map_capacity * 2)
        )
        var new_map = unsafe_alloc[Int](new_capacity)
        for i in range(new_capacity):
            new_map[unsafe_offset=i] = -1
        if self.index_map_owner:
            unsafe_memcpy(
                dest=new_map,
                src=self.index_map_owner.value(),
                count=self.index_map_capacity,
            )
            self.index_map_owner.value().unsafe_free()

        self.index_map_owner = new_map
        self.index_map_ref = new_map
        self.index_map_capacity = new_capacity

    @always_inline("nodebug")
    def _set_index(mut self, symbol_name: UInt32, slot: Int):
        var id = Int(symbol_name)
        self._ensure_index_map(id + 1)
        self.index_map_ref[unsafe_offset=id] = slot

    @always_inline("nodebug")
    def append(
        mut self, value: Tensor[f32, Self.device], symbol: Symbol
    ) where Self.device.id == Device.cpu.id:
        self.append(value, symbol.name)

    @always_inline("nodebug")
    def append(
        mut self, value: Tensor[f32, Self.device], symbol_name: UInt32
    ) where Self.device.id == Device.cpu.id:
        if self.size >= self.capacity:
            self._realloc(max(1, self.capacity * 2))
        (self.data_ref.unsafe_offset(self.size)).unsafe_write(value.copy())
        self.symbols_ref[unsafe_offset=self.size] = symbol_name
        self._set_index(symbol_name, self.size)
        self.size += 1

    @always_inline("nodebug")
    def get_index(self, symbol_name: UInt32) -> Int:
        var id = Int(symbol_name)
        if id >= self.index_map_capacity:
            return -1
        return self.index_map_ref[unsafe_offset=id]

    def __getitem__(
        self,
        symbol: Symbol,
    ) -> Tensor[f32, Self.device]:
        var index = self.get_index(symbol.name)
        ref tensor = self.data_ref[unsafe_offset=index]
        return tensor.share()

    def __setitem__(
        mut self, symbol: Symbol, value: Tensor[f32, Self.device]
    ) where Self.device.id == Device.cpu.id:
        var index = self.get_index(symbol.name)
        ref tensor = self.data_ref[unsafe_offset=index]
        unsafe_memcpy(
            dest=tensor.ptr(),
            src=value.ptr(),
            count=tensor.num_elements(),
        )

    @always_inline("nodebug")
    def clear(mut self):
        for i in range(self.size):
            (self.data_ref.unsafe_offset(i)).unsafe_deinit_pointee()
        unsafe_memset_zero(self.symbols_ref, self.capacity)
        if self.index_map_owner:
            for i in range(self.index_map_capacity):
                self.index_map_ref[unsafe_offset=i] = -1
        self.size = 0

    @always_inline("nodebug")
    def set_zero(mut self) where Self.device.id == Device.cpu.id:
        for i in range(self.size):
            self.data_ref[unsafe_offset=i].zero()

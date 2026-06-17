from std.collections.optional import Optional
from std.memory.unsafe_pointer import UnsafePointer
from std.memory import memset_zero, memcpy

from basalt import Symbol, Tensor


struct Collection(Copyable, Movable, Sized):
    """
    Symbol-keyed tensor arena.

    Tensors are stored densely in insertion order (so sparse subsets of the
    global symbol space, e.g. only the trainable parameters, don't waste
    memory). A separate `index_map` array, sized by the largest symbol id
    seen so far, maps `Symbol.name -> dense slot` for O(1) lookup without
    scanning.
    """

    var size: Int
    var capacity: Int
    var data_owner: Optional[UnsafePointer[Tensor[dtype], MutUntrackedOrigin]]
    var symbols_owner: Optional[UnsafePointer[UInt32, MutUntrackedOrigin]]
    var data_ref: UnsafePointer[Tensor[dtype], MutUntrackedOrigin]
    var symbols_ref: UnsafePointer[UInt32, MutUntrackedOrigin]

    var index_map_capacity: Int
    var index_map_owner: Optional[UnsafePointer[Int, MutUntrackedOrigin]]
    var index_map_ref: UnsafePointer[Int, MutUntrackedOrigin]

    @always_inline("nodebug")
    def __init__(out self, *, capacity: Int = 1):
        self.size = 0
        self.capacity = capacity
        self.data_owner = alloc[Tensor[dtype]](capacity)
        self.symbols_owner = alloc[UInt32](capacity)
        self.data_ref = self.data_owner.value()
        self.symbols_ref = self.symbols_owner.value()

        self.index_map_capacity = 0
        self.index_map_owner = None
        self.index_map_ref = UnsafePointer[
            Int, MutUntrackedOrigin
        ].unsafe_dangling()

    @always_inline("nodebug")
    def __init__(out self, *, deinit take: Self):
        self.size = take.size
        self.capacity = take.capacity
        self.data_owner = take.data_owner^
        self.symbols_owner = take.symbols_owner^
        self.data_ref = self.data_owner.value()
        self.symbols_ref = self.symbols_owner.value()

        self.index_map_capacity = take.index_map_capacity
        self.index_map_owner = take.index_map_owner^
        self.index_map_ref = self.index_map_owner.value() if self.index_map_owner else UnsafePointer[
            Int, MutUntrackedOrigin
        ].unsafe_dangling()

    @always_inline("nodebug")
    def __init__(out self, *, copy: Self):
        self.size = copy.size
        self.capacity = copy.capacity
        self.data_owner = alloc[Tensor[dtype]](copy.capacity)
        self.symbols_owner = alloc[UInt32](copy.capacity)
        self.data_ref = self.data_owner.value()
        self.symbols_ref = self.symbols_owner.value()

        for i in range(copy.size):
            UnsafePointer.init_pointee_move(
                self.data_ref + i, copy.data_ref[i].copy()
            )
            self.symbols_ref[i] = copy.symbols_ref[i]

        self.index_map_capacity = copy.index_map_capacity
        if copy.index_map_owner:
            var new_map = alloc[Int](copy.index_map_capacity)
            memcpy(
                dest=new_map,
                src=copy.index_map_owner.value(),
                count=copy.index_map_capacity,
            )
            self.index_map_owner = new_map
            self.index_map_ref = new_map
        else:
            self.index_map_owner = None
            self.index_map_ref = UnsafePointer[
                Int, MutUntrackedOrigin
            ].unsafe_dangling()

    @always_inline("nodebug")
    def __del__(deinit self):
        if self.data_owner:
            var data = self.data_owner.value()
            for i in range(self.size):
                UnsafePointer.destroy_pointee(data + i)
            data.free()
        if self.symbols_owner:
            self.symbols_owner.value().free()
        if self.index_map_owner:
            self.index_map_owner.value().free()

    @always_inline("nodebug")
    def __len__(self) -> Int:
        return self.size

    @always_inline("nodebug")
    def _realloc(mut self, new_capacity: Int):
        var new_data = alloc[Tensor[dtype]](new_capacity)
        var new_symbols = alloc[UInt32](new_capacity)

        for i in range(self.size):
            UnsafePointer.init_pointee_move(
                new_data + i, (self.data_ref + i).take_pointee()
            )
            new_symbols[i] = self.symbols_ref[i]

        if self.data_owner:
            self.data_owner.value().free()
        if self.symbols_owner:
            self.symbols_owner.value().free()

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
        var new_map = alloc[Int](new_capacity)
        for i in range(new_capacity):
            new_map[i] = -1
        if self.index_map_owner:
            memcpy(
                dest=new_map,
                src=self.index_map_owner.value(),
                count=self.index_map_capacity,
            )
            self.index_map_owner.value().free()

        self.index_map_owner = new_map
        self.index_map_ref = new_map
        self.index_map_capacity = new_capacity

    @always_inline("nodebug")
    def _set_index(mut self, symbol_name: UInt32, slot: Int):
        var id = Int(symbol_name)
        self._ensure_index_map(id + 1)
        self.index_map_ref[id] = slot

    @always_inline("nodebug")
    def append(mut self, value: Tensor[dtype], symbol: Symbol):
        self.append(value, symbol.name)

    @always_inline("nodebug")
    def append(mut self, value: Tensor[dtype], symbol_name: UInt32):
        if self.size >= self.capacity:
            self._realloc(max(1, self.capacity * 2))
        UnsafePointer.init_pointee_move(self.data_ref + self.size, value.copy())
        self.symbols_ref[self.size] = symbol_name
        self._set_index(symbol_name, self.size)
        self.size += 1

    @always_inline("nodebug")
    def get_index(self, symbol_name: UInt32) -> Int:
        var id = Int(symbol_name)
        if id >= self.index_map_capacity:
            return -1
        return self.index_map_ref[id]

    def __getitem__(
        self,
        symbol: Symbol,
    ) -> Tensor[dtype]:
        var index = self.get_index(symbol.name)
        ref tensor = self.data_ref[index]
        return tensor.share()

    def __setitem__(mut self, symbol: Symbol, value: Tensor[dtype]):
        var index = self.get_index(symbol.name)
        ref tensor = self.data_ref[index]
        memcpy(
            dest=tensor.mut_ptr(),
            src=value.ptr(),
            count=tensor.num_elements(),
        )

    @always_inline("nodebug")
    def clear(mut self):
        for i in range(self.size):
            UnsafePointer.destroy_pointee(self.data_ref + i)
        memset_zero(self.symbols_ref, self.capacity)
        if self.index_map_owner:
            for i in range(self.index_map_capacity):
                self.index_map_ref[i] = -1
        self.size = 0

    @always_inline("nodebug")
    def set_zero(mut self):
        for i in range(self.size):
            self.data_ref[i].zero()

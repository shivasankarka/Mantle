from std.memory.unsafe_pointer import UnsafePointer
from std.memory import memset_zero, memcpy
from std.collections.optional import Optional

from basalt import Tensor, Symbol


struct Collection(Copyable, Movable, Sized):
    """
    A collection of tensors with associated symbols.
    """

    var size: Int
    var capacity: Int
    var data: Optional[UnsafePointer[Tensor[dtype], MutExternalOrigin]]
    var symbols: Optional[UnsafePointer[Scalar[DType.uint32], MutExternalOrigin]]

    @always_inline("nodebug")
    def __init__(out self, *, capacity: Int = 1):
        """
        Initializes a new Collection with the given capacity.
        """
        self.size = 0
        self.capacity = capacity
        self.data = alloc[Tensor[dtype]](capacity)
        self.symbols = alloc[Scalar[DType.uint32]](capacity)

    @always_inline("nodebug")
    def __init__(out self, *, deinit take: Self):
        """
        Move initializes a Collection from an existing one.
        """
        self.size = take.size
        self.capacity = take.capacity
        self.data = take.data^
        self.symbols = take.symbols^

    @always_inline("nodebug")
    def __init__(out self, *, copy: Self):
        """
        Copy initializes a Collection from an existing one.
        """
        self.capacity = copy.capacity
        self.size = copy.size
        self.data = alloc[Tensor[dtype]](copy.capacity)
        self.symbols = alloc[Scalar[DType.uint32]](copy.capacity)
        var data = self.data.value()
        var symbols = self.symbols.value()
        var copy_data = copy.data.value()
        var copy_symbols = copy.symbols.value()
        memcpy(
            dest=symbols,
            src=copy_symbols,
            count=copy.capacity,
        )

        for i in range(copy.size):
            (data + i).init_pointee_move(Tensor[dtype]())
            data[i] = copy_data[i].copy()

    @always_inline("nodebug")
    def __del__(deinit self):
        """
        Destructor for the Collection.
        """
        if self.data:
            var data = self.data.value()
            for i in range(self.size):
                UnsafePointer.destroy_pointee((data + i))
            data.free()
        if self.symbols:
            self.symbols.value().free()

    @always_inline("nodebug")
    def __len__(self) -> Int:
        """
        Returns the number of elements in the Collection.
        """
        return self.size

    @always_inline("nodebug")
    def _realloc(mut self, new_capacity: Int):
        """
        Reallocates the Collection to the new capacity.
        """
        # var new_data = UnsafePointer[Tensor[dtype]].alloc(new_capacity)
        var new_data = alloc[Tensor[dtype]](new_capacity)
        # var new_symbols = UnsafePointer[Scalar[DType.uint32]].alloc(new_capacity)
        var new_symbols = alloc[Scalar[DType.uint32]](new_capacity)
        var data = self.data.value()
        var symbols = self.symbols.value()

        for i in range(self.size):
            UnsafePointer.init_pointee_move((new_data + i), Tensor[dtype]())
            new_data[i] = data[i].copy()
            new_symbols[i] = symbols[i]

        for i in range(self.size):
            UnsafePointer.destroy_pointee(data + i)
        data.free()
        symbols.free()

        self.data = new_data
        self.symbols = new_symbols
        self.capacity = new_capacity

    @always_inline("nodebug")
    def append(mut self, var value: Tensor[dtype], symbol: Symbol):
        """
        Appends a tensor and its associated symbol to the Collection.
        """
        if self.size >= self.capacity:
            self._realloc(max(1, self.capacity * 2))
        var data = self.data.value()
        var symbols = self.symbols.value()
        UnsafePointer.init_pointee_move((data + self.size), Tensor[dtype]())
        data[self.size] = value.copy()
        symbols[self.size] = symbol.name
        self.size += 1

    @always_inline("nodebug")
    def append(mut self, var value: Tensor[dtype], symbol_name: UInt32):
        """
        Appends a tensor and its associated symbol name to the Collection.
        """
        if self.size >= self.capacity:
            self._realloc(max(1, self.capacity * 2))
        var data = self.data.value()
        var symbols = self.symbols.value()
        UnsafePointer.init_pointee_move((data + self.size), Tensor[dtype]())
        data[self.size] = value.copy()
        symbols[self.size] = symbol_name
        self.size += 1

    @always_inline("nodebug")
    def get_index(self, symbol_name: UInt32) -> Int:
        """
        Returns the index of the tensor with the given symbol name.
        """
        # Keep this linear while the Mojo port settles. The previous SIMD
        # implementation could read past initialized entries for small
        # collections and return unstable indices.
        var symbols = self.symbols.value()
        for index in range(self.size):
            if symbols[index] == symbol_name:
                return index

        return -1

    def __getitem__(
        mut self,
        symbol: Symbol,
    ) -> ref[self.data.value()[0]] Tensor[dtype]:
        # TODO: This is a hack, we should instead use dict, because there can be cases where the object doesn't exist and also self.data[0] can be a value that doesn't exit because the list is empty (but we hack this by assigning an empty value)
        """
        Returns a reference to the tensor with the given symbol.
        """
        var index = self.get_index(symbol.name)

        return (self.data.value() + index)[]

    @always_inline("nodebug")
    def clear(mut self):
        """
        Clears the Collection, removing all tensors and symbols.
        """
        var data = self.data.value()
        for i in range(self.size):
            UnsafePointer.destroy_pointee((data + i))
        memset_zero(self.symbols.value(), self.capacity)
        self.size = 0

    @always_inline("nodebug")
    def set_zero(self):
        """
        Zeroes out all the tensors in the collection.
        """
        var data = self.data.value()
        for i in range(self.size):
            data[i].zero()

from std.memory.unsafe_pointer import UnsafePointer
from std.memory import memset_zero, memcpy

from basalt import Tensor, Symbol


struct Collection(Copyable, Movable, Sized):
    """
    A collection of tensors with associated symbols.
    """

    var size: Int
    var capacity: Int
    var data: UnsafePointer[Tensor[dtype], MutExternalOrigin]
    var symbols: UnsafePointer[Scalar[DType.uint32], MutExternalOrigin]

    @always_inline("nodebug")
    def __init__(out self, *, capacity: Int = 1):
        """
        Initializes a new Collection with the given capacity.
        """
        self.size = 0
        self.capacity = capacity
        self.data = alloc[Tensor[dtype]](capacity)
        UnsafePointer.init_pointee_move(
            (self.data + self.size), Tensor[dtype]()
        )
        # self.symbols = UnsafePointer[Scalar[DType.uint32]].alloc(capacity)
        self.symbols = alloc[Scalar[DType.uint32]](capacity)

    @always_inline("nodebug")
    def __init__(out self, *, deinit take: Self):
        """
        Move initializes a Collection from an existing one.
        """
        self.size = take.size
        self.capacity = take.capacity
        self.data = take.data
        self.symbols = take.symbols

    @always_inline("nodebug")
    def __init__(out self, *, copy: Self):
        """
        Copy initializes a Collection from an existing one.
        """
        self.capacity = copy.capacity
        self.size = copy.size
        self.data = alloc[Tensor[dtype]](copy.capacity)
        self.symbols = alloc[Scalar[DType.uint32]](copy.capacity)
        memcpy(dest=self.symbols, src=copy.symbols, count=copy.capacity)

        for i in range(copy.size):
            UnsafePointer.init_pointee_move(
                (self.data + i), (copy.data + i)[].copy()
            )

    @always_inline("nodebug")
    def __del__(deinit self):
        """
        Destructor for the Collection.
        """
        for i in range(self.size):
            UnsafePointer.destroy_pointee((self.data + i))
        # gotta be careful not to cause double free here. figure out how to use Optional[UnsafePointer]
        self.data.free()
        self.symbols.free()
        # if self.data:
        #     self.data.free()
        # if self.symbols:
        #     self.symbols.free()

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

        for i in range(self.size):
            UnsafePointer.init_pointee_move(
                (new_data + i), (self.data + i)[].copy()
            )
            new_symbols[i] = self.symbols[i]

        self.data.free()
        self.symbols.free()

        self.data = new_data
        self.symbols = new_symbols
        self.capacity = new_capacity

    @always_inline("nodebug")
    def append(mut self, var value: Tensor[dtype], symbol: Symbol):
        """
        Appends a tensor and its associated symbol to the Collection.
        """
        self.append(value^, symbol.name)

    @always_inline("nodebug")
    def append(mut self, var value: Tensor[dtype], symbol_name: UInt32):
        """
        Appends a tensor and its associated symbol name to the Collection.
        """
        if self.size >= self.capacity:
            self._realloc(max(1, self.capacity * 2))
        UnsafePointer.init_pointee_move((self.data + self.size), value^)
        self.symbols[self.size] = symbol_name
        self.size += 1

    @always_inline("nodebug")
    def get_index(self, symbol_name: UInt32) -> Int:
        """
        Returns the index of the tensor with the given symbol name.
        """
        comptime factor = 8
        # 2 -> 5.32s MNIST
        # 4 -> 4.95s MNIST
        # 8 -> 4.85s MNIST
        # 16 -> 5.19s MNIST
        # NOTE: This ideally should just be a hashmap

        for i in range(0, self.size, factor):
            var elems = self.symbols.load[width=factor](i).eq(symbol_name)

            for j in range(factor):
                if elems[j]:
                    return i + j

        var split = divmod(self.size, factor)

        for i in range(split[1]):
            var index = split[0] + i

            if self.symbols[index] == symbol_name:
                return index

        return -1

    def __getitem__(
        self,
        symbol: Symbol,
    ) -> ref[self.data[0]] Tensor[dtype]:
        # TODO: This is a hack, we should instead use dict, because there can be cases where the object doesn't exist and also self.data[0] can be a value that doesn't exit because the list is empty (but we hack this by assigning an empty value)
        """
        Returns a reference to the tensor with the given symbol.
        """
        var index = self.get_index(symbol.name)

        return (self.data + index)[]

    @always_inline("nodebug")
    def clear(mut self):
        """
        Clears the Collection, removing all tensors and symbols.
        """
        for i in range(self.size):
            UnsafePointer.destroy_pointee((self.data + i))
        memset_zero(self.symbols, self.capacity)
        self.size = 0

    @always_inline("nodebug")
    def set_zero(self):
        """
        Zeroes out all the tensors in the collection.
        """
        for i in range(self.size):
            self.data[i].zero()

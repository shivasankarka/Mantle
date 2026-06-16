from std.collections.optional import Optional
from std.memory import UnsafePointer

from basalt import dtype
from basalt import Tensor, TensorShape
from .symbol import Symbol
from .attributes import Attribute


struct Param(Copyable, Movable, Writable):

    var data: Optional[List[Scalar[dtype]]]
    var initializer: Optional[Attribute]

    def __init__(out self):
        self.data = None
        self.initializer = None

    def __init__(out self, data: List[Scalar[dtype]]):
        self.data = data.copy()
        self.initializer = None

    def __init__(out self, data: Scalar[dtype]):
        var data_list = List[Scalar[dtype]]()
        data_list.append(data)
        self.data = data_list^
        self.initializer = None

    def __init__(out self, initializer: String, *args: Scalar[dtype]):
        # Supported initializers:
        #   "random_uniform", lower_bound, upper_bound
        #   "random_normal", mean, std
        #   #TODO: "kaiming_uniform", mode, nonlinearity
        #   #TODO: "kaiming_normal", mode, nonlinearity
        self.initializer = Attribute("initializer", initializer)
        var data = List[Scalar[dtype]]()
        for arg in args:
            data.append(arg)
        self.data = data.copy()

    def __getitem__(self, i: Int) -> Optional[Scalar[dtype]]:
        if self.data:
            return self.data.value()[i]
        else:
            return None

    def __str__(self) -> String:
        var s: String = ""
        if self.data:
            ref data = self.data.value()
            s += "["
            for i in range(len(data)):
                s += String(data[i])
                if i < len(data) - 1:
                    s += ", "
            s += "]"
        return s


struct ParamDict(Sized, Copyable, Movable):
    var symbols: List[Symbol]
    var values: List[Param]

    def __init__(out self):
        self.symbols = List[Symbol]()
        self.values = List[Param]()

    def put(mut self, param_id: Symbol, value: Param = Param()):
        self.symbols.append(param_id)
        self.values.append(value)

    def get_tensor(self, idx: Int) -> Tensor[dtype]:
        # May only be called at runtime
        var num = self.symbols[idx].shape.num_elements()
        # var t = UnsafePointer[Scalar[dtype]].alloc(num)
        var t = alloc[Scalar[dtype]](num)
        for i in range(num):
            t[i] = self.values[idx][i].value()
        return Tensor[dtype](t, self.symbols[idx].shape)

    def __len__(self) -> Int:
        return len(self.symbols)

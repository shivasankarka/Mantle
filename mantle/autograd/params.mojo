# ===----------------------------------------------------------------------=== #
# Mantle: Params
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Params (mantle.autograd.params)
------------------------------------------------
Parameter storage for graph symbols, supporting literal data and initializer specs.
"""
from std.collections.optional import Optional
from std.memory import UnsafePointer

from mantle import f32
from mantle.core.tensor import Tensor, TensorShape
from mantle.autograd.symbol import Symbol
from mantle.autograd.attributes import Attribute


# ===----------------------------------------------------------------------===#
# Param
# ===----------------------------------------------------------------------===#

struct Param(Copyable, Movable, Writable):
    var data: Optional[List[Scalar[f32]]]
    var initializer: Optional[Attribute]

    def __init__(out self):
        self.data = None
        self.initializer = None

    def __init__(out self, data: List[Scalar[f32]]):
        self.data = data.copy()
        self.initializer = None

    def __init__(out self, data: Scalar[f32]):
        var data_list = List[Scalar[f32]]()
        data_list.append(data)
        self.data = data_list^
        self.initializer = None

    def __init__(out self, initializer: String, *args: Scalar[f32]):
        # Supported initializers:
        #   "random_uniform", lower_bound, upper_bound
        #   "random_normal", mean, std
        #   #TODO: "kaiming_uniform", mode, nonlinearity
        #   #TODO: "kaiming_normal", mode, nonlinearity
        self.initializer = Attribute("initializer", initializer)
        var data = List[Scalar[f32]]()
        for arg in args:
            data.append(arg)
        self.data = data.copy()

    def __getitem__(self, i: Int) -> Optional[Scalar[f32]]:
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


struct ParamDict(Copyable, Movable, Sized):
    var symbols: List[Symbol]
    var values: List[Param]

    def __init__(out self):
        self.symbols = List[Symbol]()
        self.values = List[Param]()

    def put(mut self, param_id: Symbol, value: Param = Param()):
        self.symbols.append(param_id)
        self.values.append(value.copy())

    def get_tensor(self, idx: Int) -> Tensor[f32]:
        # May only be called at runtime
        var num = self.symbols[idx].shape.num_elements()
        # var t = UnsafePointer[Scalar[f32]].alloc(num)
        var t = alloc[Scalar[f32]](num)
        for i in range(num):
            t[i] = self.values[idx][i].value()
        return Tensor[f32](t, self.symbols[idx].shape)

    def __len__(self) -> Int:
        return len(self.symbols)

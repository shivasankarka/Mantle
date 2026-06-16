from .autograd import Graph, Symbol, OP
from .nn import Tensor, TensorShape
from std.sys.info import simd_width_of
from basalt.utils.collection import Collection

comptime dtype = DType.float32
comptime nelts = 2 * simd_width_of[dtype]()
comptime seed = 42
comptime epsilon = 1e-12

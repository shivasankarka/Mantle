from .autograd import Graph, Symbol, OP
from .nn import Tensor, TensorShape
from std.sys.info import simd_width_of
from mantle.utils.collection import Collection

comptime f32 = DType.float32
comptime nelts = 2 * simd_width_of[f32]()
comptime seed = 42
comptime epsilon = 1e-12

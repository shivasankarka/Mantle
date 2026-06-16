from .autograd import Graph, Symbol, OP
from .nn import Tensor, TensorShape
from sys.info import simd_width_of
from basalt.utils.collection import Collection

alias dtype = DType.float32
alias nelts = 2 * simd_width_of[dtype]()
alias seed = 42
alias epsilon = 1e-12

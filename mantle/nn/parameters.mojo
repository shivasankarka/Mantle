# ===----------------------------------------------------------------------=== #
# Mantle: Parameters
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Parameters (mantle.nn.parameters)
------------------------------------------------
Runtime tensor/gradient storage used by `Model` and read by every
`forward_op`/`backward_op`.
"""
from mantle.autograd.collection import Collection
from mantle.core.device import Device


# ===----------------------------------------------------------------------===#
# Parameters
# ===----------------------------------------------------------------------===#


struct Parameters[device: Device = Device.cpu]:
    """
    Tensor/gradient storage used by `Model` and read by every
    `forward_op`/`backward_op`.
    """

    var tensors: Collection[Self.device]
    var grads: Collection[Self.device]

    def __init__(out self, tensor_capacity: Int = 1, grad_capacity: Int = 1):
        """Create storage sized for a static graph's known symbol counts.

        A `Model` knows these counts before it allocates its first tensor.
        Reserving them here avoids repeated collection growth (and metadata
        moves) while constructing larger graphs such as transformers.
        """
        self.tensors = Collection[Self.device](capacity=tensor_capacity)
        self.grads = Collection[Self.device](capacity=grad_capacity)

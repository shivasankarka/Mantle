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

    def __init__(out self):
        self.tensors = Collection[Self.device]()
        self.grads = Collection[Self.device]()

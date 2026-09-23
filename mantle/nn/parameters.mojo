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


struct Parameters:
    """
    CPU-resident tensor/gradient storage. Pinned to `Device.cpu`: `Model` and
    the op dispatch layer (`forward_op`/`backward_op`) are not yet
    device-generic, so making this generic too would only produce type
    errors with no GPU implementation behind them. Revisit once GPU tensor
    storage (and device-aware ops) land.
    """

    var tensors: Collection[Device.cpu]
    var grads: Collection[Device.cpu]

    def __init__(out self):
        self.tensors = Collection[Device.cpu]()
        self.grads = Collection[Device.cpu]()

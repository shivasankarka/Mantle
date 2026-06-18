# ===----------------------------------------------------------------------=== #
# Mantle: Data
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Data (mantle.data)
------------------------------------------------
Data loading, dataset management, and NumPy interoperability.
"""
from .dataloader import DataLoader, Batch, slice_rows, cycle_pad_rows
from .datasets import BostonHousing, MNIST
from .tensor_creation_utils import to_numpy, to_tensor, copy_np_data

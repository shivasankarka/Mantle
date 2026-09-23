# ===----------------------------------------------------------------------=== #
# Mantle: Device
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Device (mantle.core.device)
------------------------------------------------
Compile-time device tag threaded through `Tensor`/`Collection`/`Model` as a
parameter (not a runtime field), so a CPU-only build carries no overhead and
picks its storage/dispatch statically.
"""
from mantle.core.bytes import Bytes


# ===----------------------------------------------------------------------===#
# Device
# ===----------------------------------------------------------------------===#


struct Device(TrivialRegisterPassable, Writable):
    """
    Compile time device tag.
    """

    comptime cpu = Device(0, "cpu")
    comptime gpu = Device(1, "gpu")

    var id: UInt8
    var name: Bytes[8]

    def __init__(out self, id: UInt8, name: String):
        self.id = id
        self.name = Bytes[8](name)

    @always_inline("builtin")
    def __eq__(self, other: Device) -> Bool:
        return self.id == other.id

    @always_inline("builtin")
    def __ne__(self, other: Device) -> Bool:
        return self.id != other.id

    def write_to[W: Writer](self, mut writer: W):
        writer.write(String(self.name))

    def __str__(self) -> String:
        return String(self)

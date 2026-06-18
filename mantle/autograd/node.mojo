# ===----------------------------------------------------------------------=== #
# Mantle: Node
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Node (mantle.autograd.node)
------------------------------------------------
Represents an operator node in the compute graph, connecting input and output
symbols with an operator type and optional attributes.
"""
from std.collections.optional import Optional
from std.utils.variant import Variant

from mantle.autograd.symbol import Symbol
from mantle.autograd.ops import OP
from mantle.autograd.attributes import AttributeVector


# ===----------------------------------------------------------------------===#
# Node
# ===----------------------------------------------------------------------===#

struct Node(Copyable, Movable, Writable):
    var operator: OP
    var inputs: List[Symbol]
    var outputs: List[Symbol]
    var attributes: AttributeVector

    def __init__(
        out self,
        operator: OP,
        inputs: List[Symbol],
        outputs: List[Symbol],
        attributes: AttributeVector = AttributeVector(),
    ):
        self.operator = operator
        self.inputs = inputs.copy()
        self.outputs = outputs.copy()
        self.attributes = attributes

    def __str__(self) -> String:
        return self.json()

    def json(self) -> String:
        var s: String = (
            '{"operator": "' + String(self.operator.name) + '", "inputs": ['
        )
        for i in range(len(self.inputs)):
            s += self.inputs[i].json()
            if i < len(self.inputs) - 1:
                s += ", "
        s += '], "outputs": ['
        for i in range(len(self.outputs)):
            s += self.outputs[i].json()
            if i < len(self.outputs) - 1:
                s += ", "
        s += '], "attributes": ['
        for i in range(len(self.attributes)):
            s += self.attributes[i].json()
            if i < len(self.attributes) - 1:
                s += ", "
        s += "]}"
        return s

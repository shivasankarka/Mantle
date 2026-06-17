from std.collections.optional import Optional
from std.utils.variant import Variant

from src.autograd.symbol import Symbol
from src.autograd.ops import OP
from src.autograd.attributes import AttributeVector


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

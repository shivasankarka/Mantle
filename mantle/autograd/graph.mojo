# ===----------------------------------------------------------------------=== #
# Mantle: Graph
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Graph (mantle.autograd.graph)
------------------------------------------------
Compute graph builder: manages symbols, parameters, nodes, and graph compilation.
"""
from std.python.python import Python
from std.collections.optional import Optional, OptionalReg

from mantle.autograd.node import Node
from mantle.autograd.attributes import AttributeVector, Attribute
from mantle.autograd.symbol import Symbol
from mantle.autograd.ops import OP, static_result_shape, dynamic_result_shape
from mantle.autograd.ops.dynamics import SPLIT
from mantle.autograd.params import ParamDict, Param

from mantle.core.bytes import Bytes

from mantle import seed, f32
from mantle.core.tensor import Tensor, TensorShape


# ===----------------------------------------------------------------------===#
# Graph
# ===----------------------------------------------------------------------===#

struct Graph(Copyable, ImplicitlyCopyable, Movable):
    var inputs: List[Symbol]
    var params: ParamDict
    var nodes: List[Node]
    var outputs: List[Symbol]
    var loss_out: OptionalReg[Symbol]
    var symbol_count: UInt32

    def __init__(out self):
        self.inputs = List[Symbol]()
        self.params = ParamDict()
        self.nodes = List[Node]()
        self.outputs = List[Symbol]()
        self.loss_out = None
        self.symbol_count = 0

    def __init__(out self, *, deinit take: Self):
        self.inputs = take.inputs^
        self.params = take.params^
        self.nodes = take.nodes^
        self.outputs = take.outputs^
        self.loss_out = take.loss_out
        self.symbol_count = take.symbol_count

    def __init__(out self, *, copy: Self):
        self.inputs = copy.inputs.copy()
        self.params = copy.params.copy()
        self.nodes = copy.nodes.copy()
        self.outputs = copy.outputs.copy()
        self.loss_out = copy.loss_out
        self.symbol_count = copy.symbol_count

    def create_symbol(
        mut self,
        shape: TensorShape,
        data: Optional[Param] = None,
        trainable: Bool = False,
        is_input: Bool = False,
    ) -> Symbol:
        var symbol = Symbol(self.symbol_count, f32, shape, trainable)
        self.symbol_count += 1

        if is_input:
            self.inputs.append(symbol)
        else:
            if data is not None:
                self.params.put(symbol, data.value().copy())
            else:
                self.params.put(symbol)

        return symbol

    def input(mut self, shape: TensorShape, trainable: Bool = False) -> Symbol:
        return self.create_symbol(shape, trainable=trainable, is_input=True)

    def param(
        mut self, shape: TensorShape, init: Param, trainable: Bool = True
    ) -> Symbol:
        return self.create_symbol(shape, init.copy(), trainable)

    def param(mut self, shape: TensorShape, trainable: Bool = True) -> Symbol:
        return self.create_symbol(shape, trainable=trainable)

    def scalar(mut self, value: Scalar[f32]) -> Symbol:
        return self.create_symbol(TensorShape(1), Param(value), trainable=False)

    def constant(
        mut self, shape: TensorShape, data: List[Scalar[f32]]
    ) -> Symbol:
        return self.create_symbol(shape, Param(data), trainable=False)

    def out(mut self, symbol: Symbol):
        self.outputs.append(symbol)

    def loss(mut self, symbol: Symbol):
        self.loss_out = symbol

    def op(
        mut self,
        op: OP,
        *operands: Symbol,
        attributes: AttributeVector = AttributeVector(),
    ) -> Symbol:
        var res_shape = static_result_shape(op, operands, attributes)
        var res = Symbol(
            self.symbol_count, f32, res_shape, self.result_trainable(operands)
        )
        self.symbol_count += 1

        var inputs = List[Symbol]()
        inputs.reserve(len(operands))

        for operand in operands:
            inputs.append(operand)

        self.nodes.append(Node(op, inputs, [res], attributes))
        return res

    def op(
        mut self,
        op: OP,
        operand_1: Symbol,
        operand_2: Float64,
        attributes: AttributeVector = AttributeVector(),
    ) -> Symbol:
        return self.op(
            op,
            operand_1,
            self.scalar(operand_2.cast[f32]()),
            attributes=attributes,
        )

    def op(
        mut self,
        op: OP,
        operand_1: Float64,
        operand_2: Symbol,
        attributes: AttributeVector = AttributeVector(),
    ) -> Symbol:
        return self.op(
            op,
            self.scalar(operand_1.cast[f32]()),
            operand_2,
            attributes=attributes,
        )

    def create_symbols(
        mut self, shapes: List[TensorShape], trainable: Bool = False
    ) -> List[Symbol]:
        var symbols = List[Symbol]()
        symbols.reserve(len(shapes))

        for shape in shapes:
            symbols.append(Symbol(self.symbol_count, f32, shape, trainable))
            self.symbol_count += 1

        return symbols^

    def add_node(
        mut self,
        op: OP,
        inputs: List[Symbol],
        outputs: List[Symbol],
        attributes: AttributeVector,
    ):
        self.nodes.append(Node(op, inputs, outputs, attributes))

    def concat(mut self, *operands: Symbol, dim: Int = 0) -> Symbol:
        var attributes = AttributeVector(Attribute("dim", dim))
        var res_shape = dynamic_result_shape(OP.CONCAT, operands, attributes)[0]
        var res_symbols = self.create_symbols(
            [res_shape], self.result_trainable(operands)
        )

        var operand_list = List[Symbol]()
        operand_list.reserve(len(operands))
        for operand in operands:
            operand_list.append(operand)

        self.add_node(OP.CONCAT, operand_list, res_symbols, attributes)
        return res_symbols[0]

    def set_scope_from(mut self, start_idx: Int, name: String):
        for i in range(start_idx, len(self.nodes)):
            ref n = self.nodes[i]
            n.scope = Bytes[32](name)

    def split(
        mut self, operand: Symbol, sections: List[Int], dim: Int = 0
    ) -> List[Symbol]:
        var attributes = AttributeVector(
            Attribute("sections", TensorShape(sections)), Attribute("dim", dim)
        )
        var res_shapes = SPLIT.result_shape([operand.shape], attributes)
        var trainable = operand.trainable
        var result_symbols = self.create_symbols(res_shapes, trainable)
        self.add_node(OP.SPLIT, [operand], result_symbols, attributes)
        return result_symbols^

    @staticmethod
    def result_trainable(operands: VariadicList[Symbol, _]) -> Bool:
        for operand in operands:
            if operand.trainable:
                return True
        return False

    def json(self) -> String:
        var result: String = '{"graph_name": "mantle", "nodes": ['
        for i in range(len(self.nodes)):
            result += self.nodes[i].json()
            if i < len(self.nodes) - 1:
                result += ", "
        result += '], "inputs": ['
        for i in range(len(self.inputs)):
            result += self.inputs[i].json()
            if i < len(self.inputs) - 1:
                result += ", "
        result += '], "outputs": ['
        for i in range(len(self.outputs)):
            result += self.outputs[i].json()
            if i < len(self.outputs) - 1:
                result += ", "
        if self.loss_out:
            result += '], "loss": ['
            result += self.loss_out.value().json()
        result += '], "params": ['
        for i in range(len(self.params)):
            result += self.params.symbols[i].json()
            if i < len(self.params) - 1:
                result += ", "
        result += "]}"
        return result

    def render(self, render_type: String = "node") raises:
        Python.add_to_path("./mantle/utils")
        var renderer = Python.import_module("graph_render")
        var json = Python.import_module("json")
        _ = renderer.netron_render(json.loads(self.json()), render_type)

    def visualize(self, mode: String = "execution"):
        if mode == "execution":
            self._visualize_execution()
        elif mode == "architecture":
            self._visualize_architecture()
        else:
            print("Unknown mode:", mode, "(use 'execution' or 'architecture')")

    def _visualize_execution(self):
        # Assign layers via BFS: inputs/params at layer 0, each node output = max(input layer) + 1
        var sym_names = List[UInt32]()
        var sym_layers = List[Int]()

        for i in range(len(self.inputs)):
            sym_names.append(self.inputs[i].name)
            sym_layers.append(0)
        for i in range(len(self.params)):
            sym_names.append(self.params.symbols[i].name)
            sym_layers.append(0)

        for i in range(len(self.nodes)):
            ref node = self.nodes[i]
            var max_input_layer = 0
            for j in range(len(node.inputs)):
                var name = node.inputs[j].name
                for k in range(len(sym_names)):
                    if sym_names[k] == name and sym_layers[k] > max_input_layer:
                        max_input_layer = sym_layers[k]
            var out_layer = max_input_layer + 1
            for j in range(len(node.outputs)):
                sym_names.append(node.outputs[j].name)
                sym_layers.append(out_layer)

        var max_layer = 0
        for i in range(len(sym_layers)):
            if sym_layers[i] > max_layer:
                max_layer = sym_layers[i]

        print("Execution Graph:")
        print("  Layer 0  [inputs/params]:")
        for i in range(len(self.inputs)):
            var sym = self.inputs[i]
            print("    s" + String(sym.name), String(sym.dtype), String(sym.shape), "[input]")
        for i in range(len(self.params)):
            var sym = self.params.symbols[i]
            if sym.trainable:
                print("    s" + String(sym.name), String(sym.dtype), String(sym.shape), "[param, trainable]")
            else:
                print("    s" + String(sym.name), String(sym.dtype), String(sym.shape), "[param]")

        for layer in range(1, max_layer + 1):
            print("  Layer", layer, ":")
            for n in range(len(self.nodes)):
                ref node = self.nodes[n]
                var at_this_layer = False
                for j in range(len(node.outputs)):
                    for k in range(len(sym_names)):
                        if sym_names[k] == node.outputs[j].name and sym_layers[k] == layer:
                            at_this_layer = True
                            break
                    if at_this_layer:
                        break
                if not at_this_layer:
                    continue

                var input_desc = String("")
                for j in range(len(node.inputs)):
                    if j > 0:
                        input_desc += ", "
                    input_desc += "s" + String(node.inputs[j].name)

                var output_desc = String("")
                for j in range(len(node.outputs)):
                    if j > 0:
                        output_desc += ", "
                    output_desc += "s" + String(node.outputs[j].name)

                var scope_tag = ""
                if String(node.scope).byte_length() > 0:
                    scope_tag = "  [" + String(node.scope) + "]"
                print("    " + String(node.operator) + "(" + input_desc + ") -> " + output_desc + scope_tag)

        var out_str = "  Outputs:"
        for i in range(len(self.outputs)):
            out_str += " s" + String(self.outputs[i].name)
        print(out_str)
        if self.loss_out:
            print("  Loss: s" + String(self.loss_out.value().name))

    def _visualize_architecture(self):
        # Collect unique scopes in order of first appearance
        var scope_names = List[String]()
        for n in range(len(self.nodes)):
            ref node = self.nodes[n]
            var s = String(node.scope)
            var found = False
            for i in range(len(scope_names)):
                if scope_names[i] == s:
                    found = True
                    break
            if not found:
                scope_names.append(s)

        print("Architecture:")
        for si in range(len(scope_names)):
            var scope = scope_names[si]
            if scope.byte_length() == 0:
                scope = "(unnamed)"

            # Collect input and output symbols for this scope group
            var group_inputs = List[UInt32]()
            var group_outputs = List[UInt32]()
            for n in range(len(self.nodes)):
                ref node = self.nodes[n]
                var s = String(node.scope)
                if s != scope_names[si]:
                    continue
                for j in range(len(node.inputs)):
                    var found = False
                    for k in range(len(group_inputs)):
                        if group_inputs[k] == node.inputs[j].name:
                            found = True
                            break
                    if not found:
                        group_inputs.append(node.inputs[j].name)
                for j in range(len(node.outputs)):
                    var found = False
                    for k in range(len(group_outputs)):
                        if group_outputs[k] == node.outputs[j].name:
                            found = True
                            break
                    if not found:
                        group_outputs.append(node.outputs[j].name)

            # Remove symbols produced within this group from inputs
            var actual_inputs = List[UInt32]()
            for ii in range(len(group_inputs)):
                var is_internal = False
                for oo in range(len(group_outputs)):
                    if group_inputs[ii] == group_outputs[oo]:
                        is_internal = True
                        break
                if not is_internal:
                    actual_inputs.append(group_inputs[ii])

            print("  " + scope + ":")
            for n in range(len(self.nodes)):
                ref node = self.nodes[n]
                var s = String(node.scope)
                if s != scope_names[si]:
                    continue
                var input_desc = String("")
                for j in range(len(node.inputs)):
                    if j > 0:
                        input_desc += ", "
                    input_desc += "s" + String(node.inputs[j].name)
                var output_desc = String("")
                for j in range(len(node.outputs)):
                    if j > 0:
                        output_desc += ", "
                    output_desc += "s" + String(node.outputs[j].name)
                print("    " + String(node.operator) + "(" + input_desc + ") -> " + output_desc)

        # Flow summary
        print("  ---")
        var flow = "  Flow:"
        for n in range(len(self.nodes)):
            ref node = self.nodes[n]
            var s = String(node.scope)
            if s.byte_length() == 0:
                s = "?"
            flow += " " + String(node.operator) + "[" + s + "] ->"
        print(flow)
        var out_str = "  Outputs:"
        for i in range(len(self.outputs)):
            out_str += " s" + String(self.outputs[i].name)
        print(out_str)
        if self.loss_out:
            print("  Loss: s" + String(self.loss_out.value().name))

    def compile(mut self):
        # 0. Sorting the graph
        # The staticlly defined graph has an implicit topological sorted order because,
        # each new operation is added the list of nodes after its dependencies have been calculated.
        # This eliminates the need for explicit topological sorting.

        # Possibilities:
        # - 1. Graph layout transformation (graph rewrite)
        #       - Layer pruning (removing nodes that have no effect - with common sub-tree identification)
        #       - Eliminate redundant intermediate data copies
        #       - Operator replacement (e.g. replacing (combination of) costly ops with more efficient ones)
        #       - (exmple of graph rewrite: https://dl.acm.org/doi/pdf/10.1145/3453483.3454083  -  Table 4)
        #       - Other intra-block optimizations: (e.g. data layout transformation BCHW -> BHWC, etc.)
        # - 2. Operator fusion (combining ops without materializing intermediate results)
        #       - Fusion plan exploration
        #       - Fusion plan generation (with subsequent intra-block optimizations)
        #       - (example fusion plan algorithm: https://dl.acm.org/doi/pdf/10.1145/3453483.3454083   -   Listing 1)
        # - 3. Fusion Code generation (behaviour)
        #       - Code generation for planned fusion blocks
        #       - Other inter-block optimizations (e.g. data layout transformation BCHW -> BHWC, etc.)
        # - 4. Auto-tuning (of vectorization-, parallelization-, tiling-, unrolling-parameters)
        #       - (Might only work when memory is initialized)

        # Other considerations:
        # - Efficient Memory management:
        #       - Memory reuse (in-place operations)
        #       - Data layout from BCHW (batch, channel, height, width) to BHWC can lead to better utilization and efficiency
        # - VJP, JVP (for automatic differentiation)

        pass

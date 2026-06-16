from collections.optional import Optional, OptionalReg
from pathlib import Path

from sys import env_get_int

from basalt import Graph, Symbol, Tensor, TensorShape
from basalt.autograd.ops import forward_op, backward_op
from basalt.utils.collection import Collection
from basalt.utils.tensorutils import fill
from .initializers import initialize_tensor
from basalt.utils.perf_utils import PerfMetrics
from basalt.utils.onnx_utils import load_onnx_model, export_onnx_model


# When runing mojo -D DEBUG=1 -I . file, a crash happens at some point at runtime because of an error in linking it seems (because of using -I .)
# For now it seems one has to change this variable manually to be able to run model with performance metrics.
comptime DEBUG = env_get_int["DEBUG", 0]()


# TODO: remove when ability to concatenate graphs (modules)
def dv_contains(dv: List[Symbol], symbol: Symbol) -> Bool:
    for i in range(len(dv)):
        if dv[i] == symbol:
            return True
    return False


# TODO: remove when ability to concatenate graphs (modules)
def n_inference_nodes(g: Graph) -> OptionalReg[Int]:
    """
    Calculate the index of the node up to wich the forward pass should be executed for a model inference.
    When looping in revers: Equals the first index on which the node output is also a graph output.
    The number of inference nodes is that index + 1.
    """
    for i in range(len(g.nodes) - 1, -1, -1):
        for j in range(len(g.nodes[i].outputs)):
            if dv_contains(g.outputs, g.nodes[i].outputs[j]):
                return i + 1
    return None


struct Parameters:
    var tensors: Collection
    var grads: Collection

    def __init__(out self):
        self.tensors = Collection()
        self.grads = Collection()


struct Model[
    g: Graph,
    n_inference_nodes: OptionalReg[Int] = n_inference_nodes(g),
]():
    var parameters: Parameters
    var perf_metrics: PerfMetrics

    def __init__(out self, inference_only: Bool = False):
        self.parameters = Parameters()
        var graph = materialize[g]()
        @parameter
        if DEBUG == 1:
            self.perf_metrics = PerfMetrics(graph)
        else:
            self.perf_metrics = PerfMetrics()

        self.allocate_tensor_memory()
        self.allocate_grad_memory()

        # TODO: remove this when ability to concatenate graphs (modules)
        # NOTE: inference_only only used for surpressing the warning.
        if not inference_only and not g.loss_out:
            print("\n\n[WARNING]: No loss defined, model.forward() unavailable!\n\n")
        if not n_inference_nodes:
            print(
                "\n\n[WARNING]: No graph out defined, model.inference()"
                " unavailable!\n\n"
            )

    # TODO: remove when ability to concatenate graphs (modules)
    # Removes the need for splitting in forward and inference mode
    def forward(mut self, *t_inputs: Tensor[dtype]) -> ref[origin_of(self)] Tensor[dtype]:
        # NOTE: Important detail here is that the order of the inputs must be the same as the order the inputs were defined in the graph.
        # Example: If you were te define the y_true before the x when creating the graph
        #
        #   var g = Graph()
        #   var y_true = g.input(TensorShape(batch_size, n_outputs))
        #   var x = g.input(TensorShape(batch_size, n_inputs))
        #
        # Then the order of the inputs in the forward call must be the same:
        #
        #   model.forward(batch.labels, batch.inputs)

        # 1. Execute a full forward pass (model inference + loss)
        self.execute[len(g.nodes)](t_inputs ^)

        # 2. Return loss from allocated output memory
        # TODO: known copy (reference?)
        return self.parameters.tensors[g.loss_out.value()]

    def inference(mut self, *t_inputs: Tensor[dtype]) -> List[Tensor[dtype]]:
        # 1. Execute forward pass up to model out
        self.execute[n_inference_nodes.value()](t_inputs)

        # 2. Return outputs from allocated output memory
        # TODO: known copies (reference?)
        var outputs = List[Tensor[dtype]]()
        for i in range(len(g.outputs)):
            outputs.append(self.parameters.tensors[g.outputs[i]])
        return outputs ^

    def execute[num_nodes: Int](mut self, t_input: VariadicListMem[Tensor[dtype]]):
        # 1. Write inputs to allocated input memory
        var graph = materialize[g]()
        for i in range(len(graph.inputs)):
            self.parameters.tensors[graph.inputs[i]] = t_input[i].copy()

        # 2. Loop over all nodes and execute forward operations
        @parameter
        for i in range(num_nodes):
            comptime op = g.nodes[i].operator
            comptime attrs = g.nodes[i].attributes

            # Save start time for performance metrics
            @parameter
            if DEBUG == 1:
                self.perf_metrics.start_forward_pass()

            @parameter
            if op.dynamic:
                forward_op[op, attrs](
                    graph.nodes[i].inputs,
                    graph.nodes[i].outputs,
                    self.parameters,
                )
            else:
                # Statically known shapes and number of operands
                comptime num_operands = len(g.nodes[i].inputs)
                comptime t1 = g.nodes[i].inputs[0]
                comptime out = g.nodes[i].outputs[0]

                @parameter
                if num_operands == 1:
                    # Unary operator
                    forward_op[op, t1.shape, attrs](
                        self.parameters.tensors[out], self.parameters.tensors[t1]
                    )
                elif num_operands == 2:
                    # Binary operator
                    comptime t2 = g.nodes[i].inputs[1]
                    forward_op[op, t1.shape, t2.shape, attrs](
                        self.parameters.tensors[out],
                        self.parameters.tensors[t1],
                        self.parameters.tensors[t2],
                    )
                elif num_operands == 3:
                    # Ternary operator
                    comptime t2 = g.nodes[i].inputs[1]
                    comptime t3 = g.nodes[i].inputs[2]
                    forward_op[op, t1.shape, t2.shape, t3.shape, attrs](
                        self.parameters.tensors[out],
                        self.parameters.tensors[t1],
                        self.parameters.tensors[t2],
                        self.parameters.tensors[t3],
                    )

            # Save end time for performance metrics
            @parameter
            if DEBUG == 1:
                self.perf_metrics.end_forward_pass(i)

    def backward(mut self, *upper_grads: Tensor[dtype]):
        """
        Main entrypoint of backward pass.
        """
        var graph = materialize[g]()
        # 1. Initialize output gradient at the beginning of the backward pass
        if len(upper_grads) == 0:
            # TODO remove loss_out tag
            fill(self.parameters.grads[g.loss_out.value()], 1.0)
        else:
            var node_outputs = graph.nodes[len(graph.nodes)- 1].outputs.copy()
            if len(upper_grads) != len(node_outputs):
                print(
                    "[WARNING] Number of upper grads does not match number of node"
                    " outputs!"
                )
            for i in range(len(node_outputs)):
                self.parameters.grads[node_outputs[i]] = upper_grads[i].copy()

        # 2. Loop over all nodes in reverse order and execute backward operations
        @parameter
        for i in range(len(g.nodes)):
            comptime reverse_i = len(g.nodes) - i - 1
            comptime op = g.nodes[reverse_i].operator
            comptime attrs = g.nodes[reverse_i].attributes
            comptime num_operands = len(g.nodes[reverse_i].inputs)

            # Save start time for performance metrics
            @parameter
            if DEBUG == 1:
                self.perf_metrics.start_backward_pass()

            @parameter
            if op.dynamic:

                @parameter
                for j in range(num_operands):
                    @parameter
                    if g.nodes[reverse_i].inputs[j].trainable:
                        backward_op[j, op, attrs](
                            graph.nodes[reverse_i].inputs,
                            graph.nodes[reverse_i].outputs,
                            self.parameters.grads[graph.nodes[reverse_i].inputs[j]],
                            self.parameters,
                        )
            else:
                # Statically known shapes and number of operands
                comptime out = g.nodes[reverse_i].outputs[0]  # or upper_grad symbol
                comptime t1 = g.nodes[reverse_i].inputs[0]

                @parameter
                if num_operands == 1:
                    # Unary operator
                    @parameter
                    if t1.trainable:
                        backward_op[0, op, out.shape, t1.shape, attrs](
                            self.parameters.grads[out],
                            self.parameters.tensors[t1],
                            self.parameters.grads[t1],  # grad to be updated: inputs[0]
                        )

                elif num_operands == 2:
                    # Binary operator
                    comptime t2 = g.nodes[reverse_i].inputs[1]

                    @parameter
                    if t1.trainable:
                        backward_op[0, op, out.shape, t1.shape, t2.shape, attrs](
                            self.parameters.grads[out],
                            self.parameters.tensors[t1],
                            self.parameters.tensors[t2],
                            self.parameters.grads[t1],  # grad to be updated: inputs[0]
                        )

                    @parameter
                    if t2.trainable:
                        backward_op[1, op, out.shape, t1.shape, t2.shape, attrs](
                            self.parameters.grads[out],
                            self.parameters.tensors[t1],
                            self.parameters.tensors[t2],
                            self.parameters.grads[t2],  # grad to be updated: inputs[1]
                        )

                elif num_operands == 3:
                    # Ternary operator
                    comptime t2 = g.nodes[reverse_i].inputs[1]
                    comptime t3 = g.nodes[reverse_i].inputs[2]

                    @parameter
                    if t1.trainable:
                        backward_op[
                            0, op, out.shape, t1.shape, t2.shape, t3.shape, attrs
                        ](
                            self.parameters.grads[out],
                            self.parameters.tensors[t1],
                            self.parameters.tensors[t2],
                            self.parameters.tensors[t3],
                            self.parameters.grads[t1],  # grad to be updated: inputs[0]
                        )

                    @parameter
                    if t2.trainable:
                        backward_op[
                            1, op, out.shape, t1.shape, t2.shape, t3.shape, attrs
                        ](
                            self.parameters.grads[out],
                            self.parameters.tensors[t1],
                            self.parameters.tensors[t2],
                            self.parameters.tensors[t3],
                            self.parameters.grads[t2],  # grad to be updated: inputs[1]
                        )

                    @parameter
                    if t3.trainable:
                        backward_op[
                            2, op, out.shape, t1.shape, t2.shape, t3.shape, attrs
                        ](
                            self.parameters.grads[out],
                            self.parameters.tensors[t1],
                            self.parameters.tensors[t2],
                            self.parameters.tensors[t3],
                            self.parameters.grads[t3],  # grad to be updated: inputs[2]
                        )

            # Save end time for performance metrics
            @parameter
            if DEBUG == 1:
                self.perf_metrics.end_backward_pass(i)

    def allocate_tensor_memory(mut self):
        var graph = materialize[g]()
        for i in range(len(graph.inputs)):
            self.parameters.tensors.append(
                Tensor[dtype](graph.inputs[i].shape), graph.inputs[i]
            )

        for i in range(len(graph.params)):
            var p = graph.params.symbols[i]
            var p_init = graph.params.values[i].copy()

            var par: Tensor[dtype]
            if p_init.initializer:
                # 1. Specific parameter initialization defined
                var initializer_attr = p_init.initializer.value()
                par = initialize_tensor(
                    shape=p.shape,
                    type=initializer_attr.to_string(),
                    data=p_init.data.value(),
                )
            elif p_init.data:
                # 2. Parameter initialized with data only
                # Data is assumed to contain the tensor
                par = graph.params.get_tensor(i).copy()
            else:
                # Default parameter initialization to zero
                par = Tensor[dtype](p.shape)

            self.parameters.tensors.append(par ^, p)

        for i in range(len(graph.nodes)):
            # Assumption: An input or a param cannot be an output of a node
            for j in range(len(graph.nodes[i].outputs)):
                self.parameters.tensors.append(
                    Tensor[dtype](graph.nodes[i].outputs[j].shape), graph.nodes[i].outputs[j]
                )

    def allocate_grad_memory(mut self):
        # Gradient have same shape as the tensor
        var graph = materialize[g]()
        for i in range(len(graph.inputs)):
            if graph.inputs[i].trainable:
                self.parameters.grads.append(
                    Tensor[dtype](graph.inputs[i].shape), graph.inputs[i]
                )

        for i in range(len(graph.params)):
            var grad = graph.params.symbols[i]
            if grad.trainable:
                self.parameters.grads.append(Tensor[dtype](grad.shape), grad)

        for i in range(len(graph.nodes)):
            for j in range(len(graph.nodes[i].outputs)):
                var out = graph.nodes[i].outputs[j]
                if out.trainable:
                    self.parameters.grads.append(Tensor[dtype](out.shape), out)

    def print_perf_metrics(self, time_format: String = "ns", print_shape: Bool = False):
        self.perf_metrics.print_forward_perf_metrics(time_format, print_shape)
        self.perf_metrics.print_backward_perf_metrics(time_format, print_shape)

    def load_model_data(mut self, model_path: String):
        var path = Path(model_path)
        print("Loading model data from:", path)

        try:
            if path.suffix() == ".onnx":
                load_onnx_model(model_path, self.parameters, self.g)
            else:
                print("Model file format not supported:", path.suffix())
        except e:
            print("Error loading model data:", e)

    def export_model(mut self, model_path: String):
        var path = Path(model_path)
        print("Exporting model to:", path)

        try:
            if path.suffix() == ".onnx":
                export_onnx_model(model_path, self.parameters, self.g)
            else:
                print("Model file format not supported:", path.suffix())
        except e:
            print("Error exporting model:", e)

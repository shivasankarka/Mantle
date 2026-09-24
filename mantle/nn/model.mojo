# ===----------------------------------------------------------------------=== #
# Mantle: Model
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Model (mantle.nn.model)
------------------------------------------------
Graph executor for forward/backward passes with memory management and ONNX I/O.
"""
from std.collections.optional import Optional, OptionalReg
from std.pathlib import Path
from std.random import random_ui64

from std.os import abort

from mantle import f32
from mantle.autograd.graph import Graph
from mantle.autograd.symbol import Symbol
from mantle.core.tensor import Tensor, TensorShape
from mantle.core.device import Device
from mantle.autograd.ops import forward_op, backward_op
from mantle.nn.parameters import Parameters
from .initializers import initialize_tensor
from mantle.autograd.ops.gpu_elementwise import (
    gpu_rand_uniform,
    gpu_rand_normal,
)
from mantle.serialize.onnx_utils import load_onnx_model, export_onnx_model


# ===----------------------------------------------------------------------===#
# Helpers
# ===----------------------------------------------------------------------===#


def dv_contains(dv: List[Symbol], symbol: Symbol) -> Bool:
    for i in range(len(dv)):
        if dv[i] == symbol:
            return True
    return False


def gradient_contributor_count(g: Graph, symbol: Symbol) -> Int:
    """Return the number of backward paths that write `symbol`'s gradient.

    A symbol with one downstream use has exactly one gradient producer.  Its
    pre-zeroed gradient buffer can therefore be written directly, rather than
    allocating a temporary gradient and adding it back in a second kernel.
    Symbols shared by multiple nodes still use the normal accumulation path.
    """
    var count = 0
    for i in range(len(g.nodes)):
        for j in range(len(g.nodes[i].inputs)):
            if g.nodes[i].inputs[j] == symbol:
                count += 1
    return count


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


# ===----------------------------------------------------------------------===#
# Model
# ===----------------------------------------------------------------------===#


struct Model[
    g: Graph,
    n_inference_nodes: OptionalReg[Int] = OptionalReg[Int](len(g.nodes)),
    device: Device = Device.cpu,
]():
    var parameters: Parameters[Self.device]
    var step_seed: UInt64
    """Seed drawn fresh on every `forward()` call and reused by the matching
    `backward()` call, so stochastic ops (e.g. Dropout) get a different mask
    per training step but backward reconstructs the same mask forward used."""

    def __init__(out self, inference_only: Bool = False) raises:
        self.parameters = Parameters[Self.device]()
        self.step_seed = 0

        self.allocate_tensor_memory()
        self.allocate_grad_memory()

        # TODO: remove this when ability to concatenate graphs (modules)
        # NOTE: inference_only only used for surpressing the warning.
        if not inference_only and not Self.g.loss_out:
            print(
                "\n\n[WARNING]: No loss defined, model.forward()"
                " unavailable!\n\n"
            )
        if not Self.n_inference_nodes:
            print(
                "\n\n[WARNING]: No graph out defined, model.inference()"
                " unavailable!\n\n"
            )

    # TODO: remove when ability to concatenate graphs (modules)
    # Removes the need for splitting in forward and inference mode
    def forward(
        mut self, *t_inputs: Tensor[f32, Self.device]
    ) raises -> Tensor[f32, Self.device]:
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
        self.step_seed = random_ui64(0, UInt64.MAX)
        self.execute[len(Self.g.nodes)](t_inputs, self.step_seed, training=True)

        # 2. Return loss from allocated output memory
        # TODO: known copy (reference?)
        return self.parameters.tensors[Self.g.loss_out.value()]

    def inference(
        mut self, *t_inputs: Tensor[f32, Self.device]
    ) raises -> List[Tensor[f32, Self.device]]:
        # 1. Execute forward pass up to model out. training=False disables
        # stochastic ops (e.g. Dropout becomes the identity function), as
        # is standard for eval-mode inference.
        self.execute[Self.n_inference_nodes.value()](
            t_inputs, seed=0, training=False
        )

        # 2. Return outputs from allocated output memory
        # TODO: known copies (reference?)
        var outputs = List[Tensor[f32, Self.device]]()
        comptime for i in range(len(Self.g.outputs)):
            comptime sym = Self.g.outputs[i]
            outputs.append(self.parameters.tensors[sym].copy())
        return outputs^

    def execute[
        num_nodes: Int
    ](
        mut self,
        t_input: VariadicList[Tensor[f32, Self.device], _],
        seed: UInt64 = 0,
        training: Bool = True,
    ) raises:
        # 1. Write inputs to allocated input memory
        comptime for i in range(len(Self.g.inputs)):
            comptime sym = Self.g.inputs[i]
            self.parameters.tensors[sym] = t_input[i].copy()

        # 2. Loop over all nodes and execute forward operations
        comptime for i in range(num_nodes):
            comptime op = Self.g.nodes[i].operator
            comptime attrs = Self.g.nodes[i].attributes

            comptime if op.dynamic:
                comptime num_inputs = len(Self.g.nodes[i].inputs)
                comptime num_outputs = len(Self.g.nodes[i].outputs)
                var dyn_inputs = List[Symbol]()
                var dyn_outputs = List[Symbol]()
                comptime for j in range(num_inputs):
                    comptime sym = Self.g.nodes[i].inputs[j]
                    dyn_inputs.append(materialize[sym]())
                comptime for j in range(num_outputs):
                    comptime sym = Self.g.nodes[i].outputs[j]
                    dyn_outputs.append(materialize[sym]())
                forward_op[op, attrs, Self.device](
                    dyn_inputs,
                    dyn_outputs,
                    self.parameters,
                )
            else:
                # Statically known shapes and number of operands
                comptime num_operands = len(Self.g.nodes[i].inputs)
                comptime t1 = Self.g.nodes[i].inputs[0]
                comptime out = Self.g.nodes[i].outputs[0]

                comptime if num_operands == 1:
                    # Unary operator
                    forward_op[op, t1.shape, attrs, Self.device](
                        self.parameters.tensors[out],
                        self.parameters.tensors[t1],
                        seed,
                        training,
                    )
                elif num_operands == 2:
                    # Binary operator
                    comptime t2 = Self.g.nodes[i].inputs[1]
                    forward_op[op, t1.shape, t2.shape, attrs, Self.device](
                        self.parameters.tensors[out],
                        self.parameters.tensors[t1],
                        self.parameters.tensors[t2],
                    )
                elif num_operands == 3:
                    # Ternary operator
                    comptime t2 = Self.g.nodes[i].inputs[1]
                    comptime t3 = Self.g.nodes[i].inputs[2]
                    forward_op[
                        op, t1.shape, t2.shape, t3.shape, attrs, Self.device
                    ](
                        self.parameters.tensors[out],
                        self.parameters.tensors[t1],
                        self.parameters.tensors[t2],
                        self.parameters.tensors[t3],
                    )

    def backward(mut self, *upper_grads: Tensor[f32, Self.device]) raises:
        """
        Main entrypoint of backward pass.
        """
        # 1. Initialize output gradient at the beginning of the backward pass
        if len(upper_grads) == 0:
            # TODO remove loss_out tag
            self.parameters.grads[Self.g.loss_out.value()].fill(1.0)
        else:
            comptime last = len(Self.g.nodes) - 1
            comptime num_last_outputs = len(Self.g.nodes[last].outputs)
            if len(upper_grads) != num_last_outputs:
                print(
                    "[WARNING] Number of upper grads does not match number of"
                    " node outputs!"
                )
            comptime for i in range(num_last_outputs):
                comptime sym = Self.g.nodes[last].outputs[i]
                self.parameters.grads[sym] = upper_grads[i].copy()

        # 2. Loop over all nodes in reverse order and execute backward operations
        comptime for i in range(len(Self.g.nodes)):
            comptime reverse_i = len(Self.g.nodes) - i - 1
            comptime op = Self.g.nodes[reverse_i].operator
            comptime attrs = Self.g.nodes[reverse_i].attributes
            comptime num_operands = len(Self.g.nodes[reverse_i].inputs)

            comptime if op.dynamic:
                comptime for j in range(num_operands):
                    comptime if Self.g.nodes[reverse_i].inputs[j].trainable:
                        comptime num_inputs = len(
                            Self.g.nodes[reverse_i].inputs
                        )
                        comptime num_outputs = len(
                            Self.g.nodes[reverse_i].outputs
                        )
                        var dyn_inputs = List[Symbol]()
                        var dyn_outputs = List[Symbol]()
                        comptime for k in range(num_inputs):
                            comptime sym = Self.g.nodes[reverse_i].inputs[k]
                            dyn_inputs.append(materialize[sym]())
                        comptime for k in range(num_outputs):
                            comptime sym = Self.g.nodes[reverse_i].outputs[k]
                            dyn_outputs.append(materialize[sym]())
                        comptime input_sym = Self.g.nodes[reverse_i].inputs[j]
                        backward_op[j, op, attrs, Self.device](
                            dyn_inputs,
                            dyn_outputs,
                            self.parameters.grads[input_sym],
                            self.parameters,
                        )
            else:
                # Statically known shapes and number of operands
                comptime out = Self.g.nodes[reverse_i].outputs[
                    0
                ]  # or upper_grad symbol
                comptime t1 = Self.g.nodes[reverse_i].inputs[0]

                comptime if num_operands == 1:
                    # Unary operator
                    comptime if t1.trainable:
                        backward_op[
                            0, op, out.shape, t1.shape, attrs, Self.device
                        ](
                            self.parameters.grads[out],
                            self.parameters.tensors[t1],
                            self.parameters.grads[
                                t1
                            ],  # grad to be updated: inputs[0]
                            self.step_seed,
                            True,  # training: backward only runs during training
                        )

                elif num_operands == 2:
                    # Binary operator
                    comptime t2 = Self.g.nodes[reverse_i].inputs[1]

                    comptime if t1.trainable:
                        backward_op[
                            0,
                            op,
                            out.shape,
                            t1.shape,
                            t2.shape,
                            attrs,
                            Self.device,
                        ](
                            self.parameters.grads[out],
                            self.parameters.tensors[t1],
                            self.parameters.tensors[t2],
                            self.parameters.grads[
                                t1
                            ],  # grad to be updated: inputs[0]
                        )

                    comptime if t2.trainable:
                        backward_op[
                            1,
                            op,
                            out.shape,
                            t1.shape,
                            t2.shape,
                            attrs,
                            Self.device,
                        ](
                            self.parameters.grads[out],
                            self.parameters.tensors[t1],
                            self.parameters.tensors[t2],
                            self.parameters.grads[
                                t2
                            ],  # grad to be updated: inputs[1]
                        )

                elif num_operands == 3:
                    # Ternary operator
                    comptime t2 = Self.g.nodes[reverse_i].inputs[1]
                    comptime t3 = Self.g.nodes[reverse_i].inputs[2]

                    comptime if t1.trainable:
                        comptime if gradient_contributor_count(Self.g, t1) == 1:
                            backward_op[
                                0,
                                op,
                                out.shape,
                                t1.shape,
                                t2.shape,
                                t3.shape,
                                attrs,
                                Self.device,
                                overwrite_grad=True,
                            ](
                                self.parameters.grads[out],
                                self.parameters.tensors[t1],
                                self.parameters.tensors[t2],
                                self.parameters.tensors[t3],
                                self.parameters.grads[t1],
                            )
                        else:
                            backward_op[
                                0,
                                op,
                                out.shape,
                                t1.shape,
                                t2.shape,
                                t3.shape,
                                attrs,
                                Self.device,
                            ](
                                self.parameters.grads[out],
                                self.parameters.tensors[t1],
                                self.parameters.tensors[t2],
                                self.parameters.tensors[t3],
                                self.parameters.grads[t1],
                            )

                    comptime if t2.trainable:
                        comptime if gradient_contributor_count(Self.g, t2) == 1:
                            backward_op[
                                1,
                                op,
                                out.shape,
                                t1.shape,
                                t2.shape,
                                t3.shape,
                                attrs,
                                Self.device,
                                overwrite_grad=True,
                            ](
                                self.parameters.grads[out],
                                self.parameters.tensors[t1],
                                self.parameters.tensors[t2],
                                self.parameters.tensors[t3],
                                self.parameters.grads[t2],
                            )
                        else:
                            backward_op[
                                1,
                                op,
                                out.shape,
                                t1.shape,
                                t2.shape,
                                t3.shape,
                                attrs,
                                Self.device,
                            ](
                                self.parameters.grads[out],
                                self.parameters.tensors[t1],
                                self.parameters.tensors[t2],
                                self.parameters.tensors[t3],
                                self.parameters.grads[t2],
                            )

                    comptime if t3.trainable:
                        backward_op[
                            2,
                            op,
                            out.shape,
                            t1.shape,
                            t2.shape,
                            t3.shape,
                            attrs,
                            Self.device,
                        ](
                            self.parameters.grads[out],
                            self.parameters.tensors[t1],
                            self.parameters.tensors[t2],
                            self.parameters.tensors[t3],
                            self.parameters.grads[
                                t3
                            ],  # grad to be updated: inputs[2]
                        )

    def allocate_tensor_memory(mut self) raises:
        comptime for i in range(len(Self.g.inputs)):
            comptime sym = Self.g.inputs[i]
            self.parameters.tensors.append(
                Tensor[f32, Self.device](sym.shape), sym
            )

        comptime for i in range(len(Self.g.params)):
            comptime p = Self.g.params.symbols[i]
            comptime p_init = Self.g.params.values[i]

            comptime if p_init.initializer:
                # 1. Specific parameter initialization defined.
                comptime initializer_attr = p_init.initializer.value()
                comptime init_type = initializer_attr.to_string()
                comptime init_data = p_init.data.value()
                comptime init_arg0 = init_data[0]
                comptime init_arg1 = init_data[1]

                comptime if Self.device.id == Device.cpu.id:
                    comptime assert Self.device.id == Device.cpu.id
                    var init_args = List[Scalar[f32]]()
                    init_args.append(materialize[init_arg0]())
                    init_args.append(materialize[init_arg1]())
                    var par_cpu = initialize_tensor(
                        shape=p.shape,
                        type=init_type,
                        data=init_args,
                    )
                    self.parameters.tensors.append(
                        rebind[Tensor[f32, Self.device]](par_cpu), p
                    )
                else:
                    # Fill directly on-device via the Philox RNG kernels
                    # (`gpu_rand_uniform`/`gpu_rand_normal`) instead of
                    # building on the host and copying over — avoids a
                    # host round-trip for every randomly-initialized
                    # parameter.
                    var par_gpu = Tensor[f32, Device.gpu](p.shape)
                    comptime if init_type == "random_uniform":
                        gpu_rand_uniform(
                            par_gpu,
                            materialize[init_arg0](),
                            materialize[init_arg1](),
                        )
                    elif init_type == "random_normal":
                        gpu_rand_normal(
                            par_gpu,
                            materialize[init_arg0](),
                            materialize[init_arg1](),
                        )
                    elif init_type == "constant":
                        par_gpu.fill(materialize[init_arg0]())
                    else:
                        abort(
                            "Model: unsupported GPU parameter initializer: "
                            + init_type
                        )
                    self.parameters.tensors.append(
                        rebind[Tensor[f32, Self.device]](par_gpu), p
                    )
            elif p_init.data:
                # 2. Parameter initialized with data only. Literal data is
                # small and known at compile time either way, so build it on
                # the CPU and move it over for a GPU model — unlike the
                # random-initializer case above, there's no RNG-on-device
                # question here.
                var par_cpu = Tensor[f32, Device.cpu](p.shape)
                comptime init_data = p_init.data.value()
                comptime for j in range(len(init_data)):
                    comptime value = init_data[j]
                    par_cpu[j] = materialize[value]()
                comptime if Self.device.id == Device.cpu.id:
                    comptime assert Self.device.id == Device.cpu.id
                    self.parameters.tensors.append(
                        rebind[Tensor[f32, Self.device]](par_cpu), p
                    )
                else:
                    self.parameters.tensors.append(
                        rebind[Tensor[f32, Self.device]](par_cpu.to_gpu()), p
                    )
            else:
                # Default parameter initialization to zero
                self.parameters.tensors.append(
                    Tensor[f32, Self.device](p.shape), p
                )

        comptime for i in range(len(Self.g.nodes)):
            # Assumption: An input or a param cannot be an output of a node
            comptime for j in range(len(Self.g.nodes[i].outputs)):
                comptime sym = Self.g.nodes[i].outputs[j]
                self.parameters.tensors.append(
                    Tensor[f32, Self.device](sym.shape),
                    sym,
                )

    def allocate_grad_memory(mut self):
        # Gradient have same shape as the tensor
        comptime for i in range(len(Self.g.inputs)):
            comptime sym = Self.g.inputs[i]
            comptime if sym.trainable:
                self.parameters.grads.append(
                    Tensor[f32, Self.device](sym.shape), sym
                )

        comptime for i in range(len(Self.g.params)):
            comptime grad = Self.g.params.symbols[i]
            comptime if grad.trainable:
                self.parameters.grads.append(
                    Tensor[f32, Self.device](grad.shape), grad
                )

        comptime for i in range(len(Self.g.nodes)):
            comptime for j in range(len(Self.g.nodes[i].outputs)):
                comptime out = Self.g.nodes[i].outputs[j]
                comptime if out.trainable:
                    self.parameters.grads.append(
                        Tensor[f32, Self.device](out.shape), out
                    )

    def print_perf_metrics(
        self, time_format: String = "ns", print_shape: Bool = False
    ):
        pass

    def load_model_data(
        mut self, model_path: String
    ) where Self.device.id == Device.cpu.id:
        var path = Path(model_path)
        print("Loading model data from:", path)

        try:
            if path.suffix() == ".onnx":
                load_onnx_model(
                    model_path,
                    rebind[Parameters[Device.cpu]](self.parameters),
                    materialize[Self.g](),
                )
            else:
                print("Model file format not supported:", path.suffix())
        except e:
            print("Error loading model data:", e)

    def export_model(
        mut self, model_path: String
    ) where Self.device.id == Device.cpu.id:
        var path = Path(model_path)
        print("Exporting model to:", path)

        try:
            if path.suffix() == ".onnx":
                export_onnx_model(
                    model_path,
                    rebind[Parameters[Device.cpu]](self.parameters),
                    materialize[Self.g](),
                )
            else:
                print("Model file format not supported:", path.suffix())
        except e:
            print("Error exporting model:", e)

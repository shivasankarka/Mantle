from std.python import Python, PythonObject
from std.pathlib import Path
from std.collections import Set

from mantle.utils.parameters import Parameters
from mantle.nn.tensor import Tensor, TensorShape
from mantle.autograd.attributes import Attribute, AttributeType
from mantle.autograd.ops import OP
from mantle.autograd.node import Node
from mantle.autograd.graph import Graph

from .tensor_creation_utils import to_numpy, copy_np_data

# NOTE: Maybe we could create our own model representation and from there convert to onnx or others (well we already have it in reallity)
# NOTE: Torch doesn't import onnx, need onnx2torch and it doesn't support operators like reshape?


def make_onnx_attribute(op: OP, attr: Attribute) raises -> PythonObject:
    var onnx = Python.import_module("onnx")
    var onnx_helper = Python.import_module("onnx.helper")

    var attr_name = String(attr.name)
    var attr_value: PythonObject

    if attr.type == AttributeType.FLOAT:
        attr_value = attr.to_scalar[DType.float64]()
    elif attr.type == AttributeType.INT:
        attr_value = attr.to_int()
    elif attr.type == AttributeType.STRING:
        attr_value = attr.to_string()
    elif attr.type == AttributeType.INTS:
        var temp = attr.to_shape()
        var shape = Python.list()
        for i in range(temp.rank()):
            shape.append(temp[i])
        attr_value = shape
    else:
        raise Error("Unsupported attribute type")

    if op == OP.CONV2D or op == OP.MAXPOOL2D:
        if attr_name == "dilation":
            attr_name = "dilations"
        elif attr_name == "kernel_size":
            attr_name = "kernel_shape"
        elif attr_name == "stride":
            attr_name = "strides"
        elif attr_name == "padding":
            attr_name = "pads"
        else:
            raise Error("Unsupported attribute name for operator " + String(op))

    if (op == OP.CONV2D or op == OP.MAXPOOL2D) and attr_name == "pads":
        # Special case for pads in conv and maxpool, onnx wants pads to be [x1_begin, x2_begin…x1_end, x2_end,…],
        attr_value.append(attr_value[0])
        attr_value.append(attr_value[1])

    return onnx_helper.make_attribute(attr_name, attr_value)


def make_onnx_operator_type(op_type: OP) raises -> String:
    if op_type == OP.ADD:
        return "Add"
    elif op_type == OP.SUB:
        return "Sub"
    elif op_type == OP.MUL:
        return "Mul"
    elif op_type == OP.DOT:
        return "MatMul"
    elif op_type == OP.DIV:
        return "Div"
    elif op_type == OP.EXP:
        return "Exp"
    elif op_type == OP.LOG:
        return "Log"
    elif op_type == OP.SUM:
        # Special case, axis isn't an attribute, instead it is an input, because it can be dynamic
        raise Error(
            String(op_type)
            + " is not supported right now for conversion to onnx"
        )
        # return "ReduceSum"
    elif op_type == OP.MEAN:
        raise Error(
            String(op_type)
            + " is not supported right now for conversion to onnx"
        )
        # return "ReduceMean"
    elif op_type == OP.MAX:
        raise Error(
            String(op_type)
            + " is not supported right now for conversion to onnx"
        )
        # return "ReduceMax"
    elif op_type == OP.CONV2D:
        return "Conv"
    elif op_type == OP.MAXPOOL2D:
        return "MaxPool"
    elif op_type == OP.RELU:
        return "Relu"
    elif op_type == OP.TANH:
        return "Tanh"
    elif op_type == OP.SIGMOID:
        return "Sigmoid"
    elif op_type == OP.RESHAPE:
        return "Reshape"
    elif op_type == OP.TRANSPOSE:
        return "Transpose"
    elif op_type == OP.FLATTEN:
        return "Flatten"
    elif op_type == OP.SQUEEZE:
        return "Squeeze"
    elif op_type == OP.UNSQUEEZE:
        return "Unsqueeze"
    elif op_type == OP.CONCAT:
        return "Concat"
    elif op_type == OP.SPLIT:
        return "Split"
    elif op_type == OP.CLIP:
        return "Clip"
    elif op_type == OP.FMA:
        raise Error(String(op_type) + " operator is not supported in onnx")
    else:
        raise Error("Unsupported operator type " + String(op_type))


# --- Loader and exporter ---
def load_onnx_model(
    model_path: Path, mut model_parameters: Parameters, g: Graph
) raises:
    # Simple onnx data loader where we load the data in order (so we need to have the correct order of the weights and biases in the model. We don't use the names for the loading)
    var onnx = Python.import_module("onnx")
    var onnx_model = onnx.load(String(model_path))

    for i in range(len(onnx_model.graph.initializer)):
        var tensor = onnx_model.graph.initializer[i]

        if (
            tensor.data_type == onnx.TensorProto.FLOAT
            or tensor.data_type == onnx.TensorProto.INT32
            or tensor.data_type == onnx.TensorProto.INT64
        ):
            var data_np = onnx.numpy_helper.to_array(tensor)

            # Get the shape of data onnx
            var temp = List[Int]()
            for j in range(len(data_np.shape)):
                temp.append(Int(py=data_np.shape[j]))
            var data_shape = TensorShape(temp)

            # Compare the shape of the data with the shape of the model tensor
            var model_tensor_shape = g.params.symbols[i].shape

            if data_shape != model_tensor_shape:
                # check if the shape is transposed (reversed), we do this comparison because torch can save sove weights transposed (like gemm operator)

                var raise_error_flag = True
                if data_shape.rank() == model_tensor_shape.rank():
                    var count = 0
                    for j in range(model_tensor_shape.rank()):
                        if (
                            data_shape[data_shape.rank() - j - 1]
                            != model_tensor_shape[j]
                        ):
                            break
                        count += 1

                    if count == model_tensor_shape.rank():
                        print("Transposing tensor before copying " + String(i))
                        raise_error_flag = False
                        data_np = data_np.transpose()

                if raise_error_flag:
                    raise Error(
                        "Shape mismatch for tensor "
                        + String(i)
                        + ". Expected shape: "
                        + String(model_tensor_shape)
                        + ", got shape: "
                        + String(data_shape)
                    )

            copy_np_data(model_parameters.tensors[g.params.symbols[i]], data_np)
        else:
            raise Error("Unsupported data type")


def create_attributes_and_constant_inputs(
    node: Node, node_number: Int
) raises -> Tuple[List[PythonObject], List[PythonObject]]:
    var onnx = Python.import_module("onnx")
    var np = Python.import_module("numpy")

    var attributes = List[PythonObject]()
    var inputs = List[PythonObject]()

    for i in range(len(node.attributes)):
        var attr = node.attributes[i]

        def to_np_array(attr: Attribute) capturing raises -> PythonObject:
            if not attr.type == AttributeType.INTS:
                raise Error("Attribute is not a shape")

            var values_np: PythonObject
            if attr.type == AttributeType.INTS:
                var shape = attr.to_shape()
                values_np = Python.list()
                for j in range(shape.rank()):
                    values_np.append(shape[j])
            elif attr.type == AttributeType.FLOAT:
                values_np = attr.to_scalar[DType.float64]()
            elif attr.type == AttributeType.INT:
                values_np = attr.to_int()
            else:
                raise Error("Unsupported attribute type")

            var np_array = np.array(values_np, dtype=np.int64)

            return onnx.numpy_helper.from_array(np_array)

        # Special cases where attributes are considered as inputs, so we create Constant inputs
        if node.operator == OP.RESHAPE:
            if String(attr.name) == "shape":
                var outputs = Python.list()
                outputs.append(
                    String(node.operator)
                    + "_"
                    + String(attr.name)
                    + "_"
                    + String(node_number)
                )
                var temp_node = onnx.helper.make_node(
                    op_type="Constant",
                    inputs=[],
                    outputs=outputs,
                    value=to_np_array(attr),
                )

                inputs.append(temp_node)
        elif node.operator == OP.CLIP:
            if String(attr.name) == "min" or String(attr.name) == "max":
                var outputs = Python.list()
                outputs.append(
                    String(node.operator)
                    + "_"
                    + String(attr.name)
                    + "_"
                    + String(node_number)
                )
                var temp_node = onnx.helper.make_node(
                    op_type="Constant",
                    inputs=[],
                    outputs=outputs,
                    value=to_np_array(attr),
                )

                inputs.append(temp_node)
        elif node.operator == OP.SQUEEZE or node.operator == OP.UNSQUEEZE:
            if String(attr.name) == "dims":
                var outputs = Python.list()
                outputs.append(
                    String(node.operator)
                    + "_"
                    + String(attr.name)
                    + "_"
                    + String(node_number)
                )
                var temp_node = onnx.helper.make_node(
                    op_type="Constant",
                    inputs=[],
                    outputs=outputs,
                    value=to_np_array(attr),
                )

                inputs.append(temp_node)
        else:
            var attr_value = make_onnx_attribute(node.operator, attr)

            attributes.append(attr_value)

    return (attributes^, inputs^)


def export_onnx_model(
    model_path: Path, mut model_parameters: Parameters, g: Graph
) raises:
    # Create onnx model with data and nodes
    var onnx = Python.import_module("onnx")
    var onnx_helper = Python.import_module("onnx.helper")

    var graph = onnx_helper.make_graph(
        nodes=[],
        name="mantle_model",
        inputs=[],
        outputs=[],
        initializer=[],
        value_info=[],
    )

    var visited = Set[String]()

    # Create onnx initializers
    for i in range(len(g.params.symbols)):
        var tensor = g.params.symbols[i]
        var tensor_data = model_parameters.tensors[tensor].copy()
        var tensor_np = to_numpy(tensor_data)

        # Create onnx tensor
        var onnx_tensor_data = onnx_helper.make_tensor(
            name=String(tensor.name),
            data_type=onnx.TensorProto.FLOAT,
            dims=tensor_np.shape,
            vals=tensor_np,
        )

        graph.initializer.append(onnx_tensor_data)

    # Create onnx nodes
    for i in range(len(g.nodes)):
        var node = g.nodes[i].copy()

        var op_type = make_onnx_operator_type(node.operator)
        var inputs = Python.list()
        var outputs = Python.list()
        var name = String(node.operator) + "_node" + String(i)

        for j in range(len(node.inputs)):
            inputs.append(String(node.inputs[j].name))
        for j in range(len(node.outputs)):
            outputs.append(String(node.outputs[j].name))

            # Process intermediate
            if String(node.outputs[j].name) not in visited:
                visited.add(String(node.outputs[j].name))
                var intermediate_tensor = node.outputs[j]
                var intermediate_shape = intermediate_tensor.shape

                var name = String(intermediate_tensor.name)
                var dtype = onnx.TensorProto.FLOAT  # TODO
                var shape = Python.list()
                for j in range(intermediate_shape.rank()):
                    shape.append(intermediate_shape[j])

                # Create onnx tensor information
                var onnx_output = onnx_helper.make_tensor_value_info(
                    name, dtype, shape
                )
                graph.value_info.append(onnx_output)

        # Process attributes
        var attributes_and_inputs = create_attributes_and_constant_inputs(
            node, i
        )
        var attributes = attributes_and_inputs[0].copy()
        var inputs_constant = attributes_and_inputs[1].copy()
        for j in range(len(inputs_constant)):
            inputs.append(inputs_constant[j].output[0])
            graph.node.append(inputs_constant[j])

        # Create onnx node
        var onnx_node = onnx_helper.make_node(
            op_type,
            inputs,
            outputs,
            name,
        )
        for attribute in attributes:
            onnx_node.attribute.append(attribute)

        graph.node.append(onnx_node)

    # Create onnx inputs
    for i in range(len(g.inputs)):
        var input_tensor = g.inputs[i]
        var input_shape = input_tensor.shape

        var name = String(input_tensor.name)
        var dtype = onnx.TensorProto.FLOAT  # TODO
        var shape = Python.list()
        for j in range(input_shape.rank()):
            shape.append(input_shape[j])

        # Create onnx tensor information
        var onnx_input = onnx_helper.make_tensor_value_info(name, dtype, shape)
        graph.input.append(onnx_input)

    # Create onnx outputs
    for i in range(len(g.outputs)):
        var output_tensor = g.outputs[i]
        var output_shape = output_tensor.shape

        var name = String(output_tensor.name)
        var dtype = onnx.TensorProto.FLOAT  # TODO
        var shape = Python.list()
        for j in range(output_shape.rank()):
            shape.append(output_shape[j])

        # Create onnx tensor information
        var onnx_output = onnx_helper.make_tensor_value_info(name, dtype, shape)
        graph.output.append(onnx_output)

    # Create onnx model
    var onnx_model = onnx_helper.make_model(graph, producer_name="mantle")

    # Save onnx model
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, String(model_path))

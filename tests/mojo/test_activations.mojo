from std.testing import assert_equal

from mantle import f32
from mantle.nn import (
    Tensor,
    TensorShape,
    Model,
    Softmax,
    LogSoftmax,
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
)
from mantle.autograd import Graph, Symbol
from mantle.utils.tensorutils import fill

from tests import assert_tensors_equal


comptime Activation = def (mut g: Graph, input: Symbol) thin -> Symbol
comptime AxisActivation = def (mut g: Graph, input: Symbol, axis: Int) thin -> Symbol
comptime LeakyReLUActivation = def (
    mut g: Graph, input: Symbol, negative_slope: Scalar[f32]
) thin -> Symbol


def create_graph[
    shape: TensorShape,
    func: AxisActivation,
    axis: Int,
]() -> Graph:
    var g = Graph()
    var x = g.input(shape)
    var activation = func(g, x, axis)
    g.out(activation)
    return g^


def create_graph[
    shape: TensorShape,
    func: LeakyReLUActivation,
    negative_slope: Scalar[f32],
]() -> Graph:
    var g = Graph()
    var x = g.input(shape)
    var activation = func(g, x, negative_slope)
    g.out(activation)
    return g^


def create_graph[shape: TensorShape, func: Activation]() -> Graph:
    var g = Graph()
    var x = g.input(shape)
    var activation = func(g, x)
    g.out(activation)
    return g^


def test_graph[
    shape: TensorShape,
    func: AxisActivation,
    nodes: Int,
    axis: Int,
](var input: Tensor[f32], expected: Tensor[f32]) raises:
    comptime graph = create_graph[shape, func, axis]()

    var model = Model[graph](inference_only=True)
    var res = model.inference(input^)[0].copy()

    assert_tensors_equal["almost"](res, expected)
    assert_equal(comptime(len(graph.nodes)), nodes)


def test_graph[
    shape: TensorShape,
    func: LeakyReLUActivation,
    nodes: Int,
    negative_slope: Scalar[f32],
](var input: Tensor[f32], expected: Tensor[f32]) raises:
    comptime graph = create_graph[shape, func, negative_slope]()

    var model = Model[graph](inference_only=True)
    var res = model.inference(input^)[0].copy()

    assert_tensors_equal["almost"](res, expected)
    assert_equal(comptime(len(graph.nodes)), nodes)


# TODO: All these overloads feel redundant. Find a way to condense them
def test_graph[
    shape: TensorShape,
    func: Activation,
    nodes: Int,
](var input: Tensor[f32], expected: Tensor[f32]) raises:
    comptime graph = create_graph[shape, func]()

    var model = Model[graph](inference_only=True)
    var res = model.inference(input^)[0].copy()

    assert_tensors_equal["almost", "Tensor equality failed"](res, expected)
    assert_equal(comptime(len(graph.nodes)), nodes, "Node count failed")


def test_SOFTMAX() raises:
    comptime shape = TensorShape(2, 3, 2)
    comptime nodes = 5

    var input = Tensor[f32](shape)
    fill(input, 4)

    var expected = Tensor[f32](shape)

    fill(expected, 0.5)
    test_graph[shape, Softmax, nodes, 0](input.copy(), expected)

    fill(expected, 1.0 / 3.0)
    test_graph[shape, Softmax, nodes, 1](input.copy(), expected)

    fill(expected, 0.5)
    test_graph[shape, Softmax, nodes, 2](input.copy(), expected)


def test_LOGSOFTMAX() raises:
    comptime shape = TensorShape(2, 3, 2)
    comptime nodes = 6

    var input = Tensor[f32](shape)
    fill(input, 4)

    var expected = Tensor[f32](shape)

    fill(expected, -0.69314718)
    test_graph[shape, LogSoftmax, nodes, 0](input.copy(), expected)

    fill(expected, -1.09861231)
    test_graph[shape, LogSoftmax, nodes, 1](input.copy(), expected)

    fill(expected, -0.69314718)
    test_graph[shape, LogSoftmax, nodes, 2](input.copy(), expected)


def test_RELU() raises:
    comptime shape = TensorShape(2, 3)
    comptime nodes = 1

    var input = Tensor[f32](shape)

    for i in range(6):
        input[i] = 3 if i < 3 else -3

    var expected = Tensor[f32](shape)

    for i in range(6):
        expected[i] = 3 if i < 3 else 0

    test_graph[shape, ReLU, nodes](input.copy(), expected)


def test_LEAKYRELU() raises:
    comptime negative_slope = Float32(0.1)

    comptime shape = TensorShape(2, 3)
    comptime nodes = 1

    var input = Tensor[f32](shape)

    for i in range(6):
        input[i] = Float32(i - 3)

    var expected = Tensor[f32](shape)

    for i in range(6):
        expected[i] = Float32(i - 3) if i - 3 > 0 else negative_slope * Float32(i - 3)

    test_graph[shape, LeakyReLU, nodes, negative_slope](input.copy(), expected)


def test_SIGMOID() raises:
    comptime shape = TensorShape(2, 3)
    comptime nodes = 1

    var input = Tensor[f32](shape)
    fill(input, 0)

    var expected = Tensor[f32](shape)

    fill(expected, 0.5)
    test_graph[shape, Sigmoid, nodes](input.copy(), expected)


def test_TANH() raises:
    comptime shape = TensorShape(2, 3)
    comptime nodes = 1

    var input = Tensor[f32](shape)
    fill(input, 0)

    var expected = Tensor[f32](shape)

    fill(expected, 0.0)
    test_graph[shape, Tanh, nodes](input.copy(), expected)


def main():
    try:
        test_SOFTMAX()
        test_LOGSOFTMAX()
        test_RELU()
        test_LEAKYRELU()
        test_SIGMOID()
        test_TANH()
    except e:
        print("[ERROR] Error in activations")
        print(e)

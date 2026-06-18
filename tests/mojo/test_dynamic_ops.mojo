from mantle import f32, nelts
from mantle.autograd import Graph, Symbol, OP
from mantle.autograd.ops.dynamics import CONCAT, SPLIT
from mantle.nn import Model, Tensor, TensorShape
from mantle.core.tensorutils import fill

from tests import assert_tensors_equal, create_graph_concat, create_graph_split


def test_CONCAT_0() raises:
    # default: dim = 0
    # FORWARD
    comptime t1_shape = TensorShape(1, 2, 3)
    comptime t2_shape = TensorShape(1, 2, 3)
    comptime t3_shape = TensorShape(2, 2, 3)
    var t1: Tensor[f32] = Tensor[f32](t1_shape)
    var t2: Tensor[f32] = Tensor[f32](t2_shape)
    var t3: Tensor[f32] = Tensor[f32](t3_shape)
    fill(t1, 5.0)
    fill(t2, 10.0)
    fill(t3, 15.0)

    var expected = Tensor[f32](4, 2, 3)
    for i in range(4):
        for j in range(2):
            for k in range(3):
                if i < 1:  # i because dim = 0
                    expected[i * 2 * 3 + j * 3 + k] = 5.0
                elif i >= 1 and i < 2:
                    expected[i * 2 * 3 + j * 3 + k] = 10.0
                else:
                    expected[i * 2 * 3 + j * 3 + k] = 15.0

    comptime graph = create_graph_concat(t1_shape, t2_shape, t3_shape, dim=0)
    var model = Model[graph]()
    ref res = model.forward(t1.copy(), t2.copy(), t3.copy())
    assert_tensors_equal["almost"](res, expected)

    # BACKWARD
    var ug = Tensor[f32](4, 2, 3)
    for i in range(4):
        for j in range(2):
            for k in range(3):
                if i < 1:  # i because dim = 0
                    ug[i * 2 * 3 + j * 3 + k] = 1.0
                elif i >= 1 and i < 2:
                    ug[i * 2 * 3 + j * 3 + k] = 2.0
                else:
                    ug[i * 2 * 3 + j * 3 + k] = 3.0

    model.backward(ug.copy())

    var grad1_expected = Tensor[f32](t1_shape)
    var grad2_expected = Tensor[f32](t2_shape)
    var grad3_expected = Tensor[f32](t3_shape)
    fill(grad1_expected, 1.0)
    fill(grad2_expected, 2.0)
    fill(grad3_expected, 3.0)

    comptime grad1 = graph.nodes[0].inputs[0]
    comptime grad2 = graph.nodes[0].inputs[1]
    comptime grad3 = graph.nodes[0].inputs[2]

    # Extracting the gradients
    assert_tensors_equal["almost"](
        model.parameters.grads[grad1], grad1_expected
    )
    assert_tensors_equal["almost"](
        model.parameters.grads[grad2], grad2_expected
    )
    assert_tensors_equal["almost"](
        model.parameters.grads[grad3], grad3_expected
    )


def test_CONCAT_1() raises:
    # dim = 1
    comptime t1_shape = TensorShape(2, 2, 5)
    comptime t2_shape = TensorShape(2, 4, 5)
    comptime t3_shape = TensorShape(2, 1, 5)
    var t1: Tensor[f32] = Tensor[f32](t1_shape)
    var t2: Tensor[f32] = Tensor[f32](t2_shape)
    var t3: Tensor[f32] = Tensor[f32](t3_shape)
    fill(t1, 5.0)
    fill(t2, 10.0)
    fill(t3, 15.0)

    var expected = Tensor[f32](2, 7, 5)
    for i in range(2):
        for j in range(7):
            for k in range(5):
                if j < 2:  # j because dim = 1
                    expected[i * 7 * 5 + j * 5 + k] = 5.0
                elif j >= 2 and j < 6:
                    expected[i * 7 * 5 + j * 5 + k] = 10.0
                else:
                    expected[i * 7 * 5 + j * 5 + k] = 15.0

    comptime graph = create_graph_concat(t1_shape, t2_shape, t3_shape, dim=1)
    var model = Model[graph]()
    ref res = model.forward(t1.copy(), t2.copy(), t3.copy())
    assert_tensors_equal["almost"](res, expected)

    # BACKWARD
    var ug = Tensor[f32](2, 7, 5)
    for i in range(2):
        for j in range(7):
            for k in range(5):
                if j < 2:  # j because dim = 1
                    ug[i * 7 * 5 + j * 5 + k] = 1.0
                elif j >= 2 and j < 6:
                    ug[i * 7 * 5 + j * 5 + k] = 2.0
                else:
                    ug[i * 7 * 5 + j * 5 + k] = 3.0

    model.backward(ug.copy())

    var grad1_expected = Tensor[f32](t1_shape)
    var grad2_expected = Tensor[f32](t2_shape)
    var grad3_expected = Tensor[f32](t3_shape)
    fill(grad1_expected, 1.0)
    fill(grad2_expected, 2.0)
    fill(grad3_expected, 3.0)

    comptime grad1 = graph.nodes[0].inputs[0]
    comptime grad2 = graph.nodes[0].inputs[1]
    comptime grad3 = graph.nodes[0].inputs[2]

    # Extracting the gradients
    assert_tensors_equal["almost"](
        model.parameters.grads[grad1], grad1_expected
    )
    assert_tensors_equal["almost"](
        model.parameters.grads[grad2], grad2_expected
    )
    assert_tensors_equal["almost"](
        model.parameters.grads[grad3], grad3_expected
    )


def test_CONCAT_2() raises:
    # dim = 2
    comptime t1_shape = TensorShape(2, 3, 1)
    comptime t2_shape = TensorShape(2, 3, 2)
    comptime t3_shape = TensorShape(2, 3, 3)
    var t1: Tensor[f32] = Tensor[f32](t1_shape)
    var t2: Tensor[f32] = Tensor[f32](t2_shape)
    var t3: Tensor[f32] = Tensor[f32](t3_shape)
    fill(t1, 5.0)
    fill(t2, 10.0)
    fill(t3, 15.0)

    var expected = Tensor[f32](2, 3, 6)
    for i in range(2):
        for j in range(3):
            for k in range(6):
                if k < 1:  # k because dim = 2
                    expected[i * 3 * 6 + j * 6 + k] = 5.0
                elif k >= 1 and k < 3:
                    expected[i * 3 * 6 + j * 6 + k] = 10.0
                else:
                    expected[i * 3 * 6 + j * 6 + k] = 15.0

    comptime graph = create_graph_concat(t1_shape, t2_shape, t3_shape, dim=2)
    var model = Model[graph]()
    ref res = model.forward(t1.copy(), t2.copy(), t3.copy())
    assert_tensors_equal["almost"](res, expected)

    # BACKWARD
    var ug = Tensor[f32](2, 3, 6)
    for i in range(2):
        for j in range(3):
            for k in range(6):
                if k < 1:  # k because dim = 2
                    ug[i * 3 * 6 + j * 6 + k] = 1.0
                elif k >= 1 and k < 3:
                    ug[i * 3 * 6 + j * 6 + k] = 2.0
                else:
                    ug[i * 3 * 6 + j * 6 + k] = 3.0

    model.backward(ug.copy())

    var grad1_expected = Tensor[f32](t1_shape)
    var grad2_expected = Tensor[f32](t2_shape)
    var grad3_expected = Tensor[f32](t3_shape)
    fill(grad1_expected, 1.0)
    fill(grad2_expected, 2.0)
    fill(grad3_expected, 3.0)

    comptime grad1 = graph.nodes[0].inputs[0]
    comptime grad2 = graph.nodes[0].inputs[1]
    comptime grad3 = graph.nodes[0].inputs[2]

    assert_tensors_equal["almost"](
        model.parameters.grads[grad1], grad1_expected
    )
    assert_tensors_equal["almost"](
        model.parameters.grads[grad2], grad2_expected
    )
    assert_tensors_equal["almost"](
        model.parameters.grads[grad3], grad3_expected
    )


def test_SPLIT_0() raises:
    comptime t_shape = TensorShape(4, 5, 6)
    comptime sections: List[Int] = [1, 2, 1]

    var t: Tensor[f32] = Tensor[f32](t_shape)
    for i in range(4):
        for j in range(5):
            for k in range(6):
                if i < 1:
                    t[i * 5 * 6 + j * 6 + k] = 5.0
                elif i >= 1 and i < 3:
                    t[i * 5 * 6 + j * 6 + k] = 10.0
                else:
                    t[i * 5 * 6 + j * 6 + k] = 15.0

    var expected1 = Tensor[f32](1, 5, 6)
    var expected2 = Tensor[f32](2, 5, 6)
    var expected3 = Tensor[f32](1, 5, 6)
    fill(expected1, 5.0)
    fill(expected2, 10.0)
    fill(expected3, 15.0)

    comptime graph = create_graph_split(t_shape, sections, dim=0)
    var model = Model[graph]()
    var results = model.inference(t.copy())

    assert_tensors_equal["almost"](results[0], expected1)
    assert_tensors_equal["almost"](results[1], expected2)
    assert_tensors_equal["almost"](results[2], expected3)

    # BACKWARD
    var ug1 = Tensor[f32](1, 5, 6)
    var ug2 = Tensor[f32](2, 5, 6)
    var ug3 = Tensor[f32](1, 5, 6)
    fill(ug1, 1.0)
    fill(ug2, 2.0)
    fill(ug3, 3.0)

    model.backward(ug1.copy(), ug2.copy(), ug3.copy())

    var grad_expected = Tensor[f32](t_shape)
    for i in range(4):
        for j in range(5):
            for k in range(6):
                if i < 1:
                    grad_expected[i * 5 * 6 + j * 6 + k] = 1.0
                elif i >= 1 and i < 3:
                    grad_expected[i * 5 * 6 + j * 6 + k] = 2.0
                else:
                    grad_expected[i * 5 * 6 + j * 6 + k] = 3.0

    comptime grad = graph.nodes[0].inputs[0]
    assert_tensors_equal["almost"](
        model.parameters.grads[grad], grad_expected
    )


def test_SPLIT_1() raises:
    comptime t_shape = TensorShape(4, 5, 6)
    comptime sections: List[Int] = [1, 3, 1]

    var t: Tensor[f32] = Tensor[f32](t_shape)
    for i in range(4):
        for j in range(5):
            for k in range(6):
                if j < 1:
                    t[i * 5 * 6 + j * 6 + k] = 5.0
                elif j >= 1 and j < 4:
                    t[i * 5 * 6 + j * 6 + k] = 10.0
                else:
                    t[i * 5 * 6 + j * 6 + k] = 15.0

    var expected1 = Tensor[f32](4, 1, 6)
    var expected2 = Tensor[f32](4, 3, 6)
    var expected3 = Tensor[f32](4, 1, 6)
    fill(expected1, 5.0)
    fill(expected2, 10.0)
    fill(expected3, 15.0)

    comptime graph = create_graph_split(t_shape, sections, dim=1)
    var model = Model[graph]()
    var results = model.inference(t.copy())

    assert_tensors_equal["almost"](results[0], expected1)
    assert_tensors_equal["almost"](results[1], expected2)
    assert_tensors_equal["almost"](results[2], expected3)

    # BACKWARD
    var ug1 = Tensor[f32](4, 1, 6)
    var ug2 = Tensor[f32](4, 3, 6)
    var ug3 = Tensor[f32](4, 1, 6)
    fill(ug1, 1.0)
    fill(ug2, 2.0)
    fill(ug3, 3.0)

    model.backward(ug1.copy(), ug2.copy(), ug3.copy())

    var grad_expected = Tensor[f32](t_shape)
    for i in range(4):
        for j in range(5):
            for k in range(6):
                if j < 1:
                    grad_expected[i * 5 * 6 + j * 6 + k] = 1.0
                elif j >= 1 and j < 4:
                    grad_expected[i * 5 * 6 + j * 6 + k] = 2.0
                else:
                    grad_expected[i * 5 * 6 + j * 6 + k] = 3.0

    comptime grad = graph.nodes[0].inputs[0]
    assert_tensors_equal["almost"](
        model.parameters.grads[grad], grad_expected
    )


def test_SPLIT_2() raises:
    comptime t_shape = TensorShape(4, 5, 6)
    comptime sections: List[Int] = [1, 4, 1]

    var t: Tensor[f32] = Tensor[f32](t_shape)
    for i in range(4):
        for j in range(5):
            for k in range(6):
                if k < 1:
                    t[i * 5 * 6 + j * 6 + k] = 5.0
                elif k >= 1 and k < 5:
                    t[i * 5 * 6 + j * 6 + k] = 10.0
                else:
                    t[i * 5 * 6 + j * 6 + k] = 15.0

    var expected1 = Tensor[f32](4, 5, 1)
    var expected2 = Tensor[f32](4, 5, 4)
    var expected3 = Tensor[f32](4, 5, 1)
    fill(expected1, 5.0)
    fill(expected2, 10.0)
    fill(expected3, 15.0)

    comptime graph = create_graph_split(t_shape, sections, dim=2)
    var model = Model[graph]()
    var results = model.inference(t.copy())

    assert_tensors_equal["almost"](results[0], expected1)
    assert_tensors_equal["almost"](results[1], expected2)
    assert_tensors_equal["almost"](results[2], expected3)

    # BACKWARD
    var ug1 = Tensor[f32](4, 5, 1)
    var ug2 = Tensor[f32](4, 5, 4)
    var ug3 = Tensor[f32](4, 5, 1)
    fill(ug1, 1.0)
    fill(ug2, 2.0)
    fill(ug3, 3.0)

    model.backward(ug1.copy(), ug2.copy(), ug3.copy())

    var grad_expected = Tensor[f32](t_shape)
    for i in range(4):
        for j in range(5):
            for k in range(6):
                if k < 1:
                    grad_expected[i * 5 * 6 + j * 6 + k] = 1.0
                elif k >= 1 and k < 5:
                    grad_expected[i * 5 * 6 + j * 6 + k] = 2.0
                else:
                    grad_expected[i * 5 * 6 + j * 6 + k] = 3.0

    comptime grad = graph.nodes[0].inputs[0]
    assert_tensors_equal["almost"](
        model.parameters.grads[grad], grad_expected
    )


def main():
    try:
        test_CONCAT_0()
        test_CONCAT_1()
        test_CONCAT_2()
        test_SPLIT_0()
        test_SPLIT_1()
        test_SPLIT_2()
    except e:
        print("[ERROR] Error in dynamic ops")
        print(e)
        return

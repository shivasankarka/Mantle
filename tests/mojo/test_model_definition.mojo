"""Regression test for the reflected, static-graph model-definition API."""
from std.testing import assert_true

import mantle.nn as nn
from mantle import Graph, Tensor, TensorShape, f32


@fieldwise_init
struct TinyClassifier(Copyable, Movable):
    var hidden: nn.LinearLayer
    var activation: nn.ReLULayer
    var head: nn.LinearLayer


def make_graph(batch_size: Int) -> Graph:
    var architecture = TinyClassifier(
        nn.Linear(4), nn.ReLU(), nn.Linear(3)
    )
    return nn.classification_graph(
        architecture, TensorShape(batch_size, 2)
    )


def make_sequential_graph() -> Graph:
    var network = nn.Sequential(
        nn.Linear(4), nn.BatchNorm2d(), nn.ReLU(), nn.Linear(3)
    )
    return nn.classification_graph(network, TensorShape(2, 2))


def test_reflected_model_definition() raises:
    comptime graph = make_graph(2)
    assert_true(
        comptime(len(graph.inputs) == 2),
        "classifier should create data and labels",
    )
    comptime logits = graph.outputs[0]
    assert_true(
        logits.shape == TensorShape(2, 3),
        "reflected layers should determine the logits shape",
    )

    var model = nn.Model[graph]()
    model.summary()
    var inputs = Tensor[f32](TensorShape(2, 2))
    var labels = Tensor[f32](TensorShape(2, 3))
    var loss = model.forward(inputs, labels)
    assert_true(loss[0] == loss[0], "loss should be finite")
    var prediction = model.predict(inputs)
    assert_true(
        prediction[0] == prediction[0], "prediction should be finite"
    )
    print("test_reflected_model_definition: PASSED")


def test_sequential_model_definition() raises:
    comptime graph = make_sequential_graph()
    var model = nn.Model[graph]()
    var inputs = Tensor[f32](TensorShape(2, 2))
    var prediction = model.predict(inputs)
    assert_true(
        prediction.shape() == TensorShape(2, 3),
        "Sequential should build classifier logits",
    )
    print("test_sequential_model_definition: PASSED")


def main() raises:
    test_reflected_model_definition()
    test_sequential_model_definition()

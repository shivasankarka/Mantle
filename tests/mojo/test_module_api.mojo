"""Regression test for custom modules built without graph arguments."""
from std.testing import assert_true

import mantle.nn as nn
from mantle import Graph, Tensor, TensorShape, f32


@fieldwise_init
struct ResidualMLP(nn.Module, Movable):
    var hidden: nn.LinearLayer
    var output: nn.LinearLayer

    def forward(mut self, input: nn.Expr) -> nn.Expr:
        return self.output(self.hidden(input).relu()) + input


def make_graph(batch_size: Int) -> Graph:
    var architecture = ResidualMLP(nn.Linear(2), nn.Linear(2))
    return nn.classification_graph(
        architecture, TensorShape(batch_size, 2)
    )


def test_custom_module_builds_static_graph() raises:
    comptime graph = make_graph(2)
    comptime logits = graph.outputs[0]
    assert_true(
        logits.shape == TensorShape(2, 2),
        "residual module should preserve the feature shape",
    )

    var model = nn.Model[graph]()
    var inputs = Tensor[f32](TensorShape(2, 2))
    var labels = Tensor[f32](TensorShape(2, 2))
    var loss = model.forward(inputs, labels)
    assert_true(loss[0] == loss[0], "module graph should execute finitely")
    print("test_custom_module_builds_static_graph: PASSED")


def main() raises:
    test_custom_module_builds_static_graph()

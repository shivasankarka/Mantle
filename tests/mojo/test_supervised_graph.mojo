"""Regression tests for generic module training graphs."""
from std.testing import assert_true

import mantle.nn as nn
import mantle.nn.optim as optim
from mantle import Graph, Tensor, TensorShape, f32


@fieldwise_init
struct Regressor(nn.Module, Movable):
    var head: nn.LinearLayer

    def forward(mut self, input: nn.Expr) -> nn.Expr:
        return self.head(input)


def make_graph() -> Graph:
    var network = Regressor(nn.Linear(1))
    var loss = nn.MSELoss()
    return nn.supervised_graph(
        network, TensorShape(2, 2), TensorShape(2, 1), loss
    )


def test_supervised_graph_executes() raises:
    comptime graph = make_graph()
    var model = nn.Model[graph]()
    var inputs = Tensor[f32](TensorShape(2, 2))
    var targets = Tensor[f32](TensorShape(2, 1))
    var objective = model.forward(inputs, targets)
    assert_true(objective[0] == objective[0], "objective should be finite")
    print("test_supervised_graph_executes: PASSED")


def test_train_step_executes() raises:
    comptime graph = make_graph()
    var model = nn.Model[graph]()
    var optimizer = optim.SGD[graph](model.parameters, lr=0.01)
    var inputs = Tensor[f32](TensorShape(2, 2))
    var targets = Tensor[f32](TensorShape(2, 1))
    var objective = model.train_step(optimizer, inputs, targets)
    assert_true(objective[0] == objective[0], "train_step should be finite")
    print("test_train_step_executes: PASSED")


def main() raises:
    test_supervised_graph_executes()
    test_train_step_executes()

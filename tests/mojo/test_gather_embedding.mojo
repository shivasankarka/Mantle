"""Test OP.GATHER forward/backward and the Embedding layer."""
from std.testing import assert_true

from mantle import f32
from mantle.autograd.graph import Graph
from mantle.autograd.ops import OP
from mantle.autograd.attributes import Attribute, AttributeVector
from mantle.core.tensor import Tensor, TensorShape
from mantle.core.tensorutils import fill
import mantle.nn as nn
from mantle.nn.model import Model
import mantle.nn.optim as optim

from tests import test_binary_op, test_binary_op_backward


# ===----------------------------------------------------------------------===#
# GATHER op
# ===----------------------------------------------------------------------===#


def test_gather_forward() raises:
    comptime table_shape = TensorShape(4, 3)  # vocab=4, dim=3
    comptime indices_shape = TensorShape(2, 2)  # (B=2, T=2)

    var table = Tensor[f32](table_shape)
    for row in range(4):
        for d in range(3):
            table[row * 3 + d] = Float32(row * 10 + d)

    var indices = Tensor[f32](indices_shape)
    indices[0] = 0.0
    indices[1] = 2.0
    indices[2] = 3.0
    indices[3] = 1.0

    var expected = Tensor[f32](2, 2, 3)
    # row 0: [0, 1, 2], row 2: [20, 21, 22], row 3: [30, 31, 32], row 1: [10, 11, 12]
    var expected_vals: List[Float32] = [
        0, 1, 2, 20, 21, 22, 30, 31, 32, 10, 11, 12
    ]
    for i in range(12):
        expected[i] = expected_vals[i]

    test_binary_op[OP.GATHER, table_shape, indices_shape](
        table, indices, expected
    )
    print("test_gather_forward: PASSED")


def test_gather_backward() raises:
    comptime table_shape = TensorShape(3, 2)  # vocab=3, dim=2
    comptime indices_shape = TensorShape(3)
    comptime ug_shape = TensorShape(3, 2)

    var table = Tensor[f32](table_shape)
    var indices = Tensor[f32](indices_shape)
    # repeated index 0 -> gradient should accumulate
    indices[0] = 0.0
    indices[1] = 0.0
    indices[2] = 1.0

    var ug = Tensor[f32](ug_shape)
    fill(ug, 1.0)

    var grad_table_expected = Tensor[f32](3, 2)
    grad_table_expected[0] = 2.0
    grad_table_expected[1] = 2.0
    grad_table_expected[2] = 1.0
    grad_table_expected[3] = 1.0
    grad_table_expected[4] = 0.0
    grad_table_expected[5] = 0.0

    var grad_indices_expected = Tensor[f32](indices_shape)  # unused (no grad)

    test_binary_op_backward[OP.GATHER, table_shape, indices_shape, ug_shape](
        table, indices, ug, grad_table_expected, grad_indices_expected
    )
    print("test_gather_backward: PASSED")


# ===----------------------------------------------------------------------===#
# Embedding layer (training convergence)
# ===----------------------------------------------------------------------===#


def make_embedding_graph(vocab: Int, dim: Int, batch: Int, seq: Int) -> Graph:
    var g = Graph()
    var ids = g.input(TensorShape(batch, seq))
    var emb = nn.Embedding(g, ids, vocab, dim)
    var flat = g.op(
        OP.RESHAPE,
        emb,
        attributes=AttributeVector(
            Attribute("shape", TensorShape(batch, seq * dim))
        ),
    )
    var y = g.input(TensorShape(batch, seq * dim))
    var loss = nn.MSELoss(g, flat, y)
    g.loss(loss)
    return g^


def test_embedding_trains() raises:
    comptime vocab = 5
    comptime dim = 4
    comptime batch = 2
    comptime seq = 3

    comptime g = make_embedding_graph(vocab, dim, batch, seq)
    var model = Model[g]()
    var sgd = optim.SGD[g](model.parameters, lr=0.1)

    var ids = Tensor[f32](TensorShape(batch, seq))
    ids[0] = 0.0; ids[1] = 1.0; ids[2] = 2.0
    ids[3] = 3.0; ids[4] = 4.0; ids[5] = 0.0

    var y = Tensor[f32](TensorShape(batch, seq * dim))
    fill(y, 1.0)

    var initial_loss: Float32 = model.forward(ids, y)[0]
    for _ in range(20):
        _ = model.forward(ids, y)
        model.backward()
        sgd.step()
        sgd.zero_grad()

    var final_loss: Float32 = model.forward(ids, y)[0]
    assert_true(
        final_loss < initial_loss,
        "Embedding training should decrease loss (got "
        + String(initial_loss)
        + " -> "
        + String(final_loss)
        + ")",
    )
    print(
        "test_embedding_trains: PASSED (loss "
        + String(initial_loss)
        + " -> "
        + String(final_loss)
        + ")"
    )


def main() raises:
    test_gather_forward()
    test_gather_backward()
    test_embedding_trains()

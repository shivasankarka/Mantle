from std.testing import assert_equal, assert_true

import mantle.nn as nn
from mantle import Graph, Symbol, OP, Tensor, TensorShape, f32
from mantle.nn.optim import Adam
from mantle.serialize.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    save_checkpoint_with_optim,
    load_checkpoint_with_optim,
)


def linear_regression(batch_size: Int, n_inputs: Int, n_outputs: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch_size, n_inputs))
    var y_true = g.input(TensorShape(batch_size, n_outputs))
    var y_pred = nn.Linear(g, x, n_outputs)
    g.out(y_pred)
    var loss = nn.MSELoss(g, y_pred, y_true)
    g.loss(loss)
    return g^


def test_save_load_checkpoint_plain() raises:
    comptime graph = linear_regression(4, 3, 1)
    comptime param_sym = graph.params.symbols[0]

    var model = nn.Model[graph]()
    var x = Tensor[f32](TensorShape(4, 3))
    var y = Tensor[f32](TensorShape(4, 1))
    for i in range(12):
        x[i] = Float32(i) * 0.1
    for i in range(4):
        y[i] = Float32(i)

    _ = model.forward(x.copy(), y.copy())

    var expected = model.parameters.tensors[param_sym][0]

    save_checkpoint("test_ckpt_plain.bin", model.parameters)

    var model2 = nn.Model[graph]()
    var info = load_checkpoint("test_ckpt_plain.bin", model2.parameters)

    assert_equal(model2.parameters.tensors[param_sym][0], expected)
    assert_true(not info.has_optim_state)


def test_save_load_checkpoint_with_optim() raises:
    comptime graph = linear_regression(4, 3, 1)
    comptime param_sym = graph.params.symbols[0]

    var model = nn.Model[graph]()
    var optim = Adam[graph](model.parameters, lr=0.01)

    var x = Tensor[f32](TensorShape(4, 3))
    var y = Tensor[f32](TensorShape(4, 1))
    for i in range(12):
        x[i] = Float32(i) * 0.1
    for i in range(4):
        y[i] = Float32(i)

    _ = model.forward(x.copy(), y.copy())
    optim.zero_grad()
    model.backward()
    optim.step()

    var expected_param = model.parameters.tensors[param_sym][0]
    var expected_momentum = optim.momentum_grads[param_sym][0]
    var expected_iter = optim.iter

    save_checkpoint_with_optim(
        "test_ckpt_optim.bin",
        model.parameters,
        optim.momentum_grads,
        optim.rms_grads,
        iter=optim.iter,
    )

    var model2 = nn.Model[graph]()
    var optim2 = Adam[graph](model2.parameters, lr=0.01)

    var info = load_checkpoint_with_optim(
        "test_ckpt_optim.bin",
        model2.parameters,
        optim2.momentum_grads,
        optim2.rms_grads,
    )

    assert_equal(model2.parameters.tensors[param_sym][0], expected_param)
    assert_equal(optim2.momentum_grads[param_sym][0], expected_momentum)
    assert_equal(info.iter, expected_iter)
    assert_true(info.has_optim_state)


def main() raises:
    try:
        test_save_load_checkpoint_plain()
        test_save_load_checkpoint_with_optim()
    except e:
        print(e)
        raise e^

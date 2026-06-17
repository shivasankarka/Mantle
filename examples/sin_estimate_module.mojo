from std.random import rand
from std.time import perf_counter_ns as now
import std.math as math

import basalt.nn as nn
from basalt import Tensor, TensorShape
from basalt import dtype
from basalt import Graph, Symbol, OP
from basalt.utils.tensorutils import fill


@fieldwise_init
struct SimpleNN(Copyable, Movable):
    """
    PyTorch-`nn.Module`-style model definition: layers are plain struct
    fields, declared in forward-pass order. `nn.build_graph` reflects over
    them and chains `forward(g, x) -> x` automatically.
    """

    var fc1: nn.LinearLayer
    var act1: nn.ReLULayer
    var fc2: nn.LinearLayer
    var act2: nn.ReLULayer
    var fc3: nn.LinearLayer


def create_simple_nn(batch_size: Int, n_inputs: Int, n_outputs: Int) -> Graph:
    var g = Graph()

    var x = g.input(TensorShape(batch_size, n_inputs))
    var y_true = g.input(TensorShape(batch_size, n_outputs))

    var model_def = SimpleNN(
        fc1=nn.LinearLayer(32),
        act1=nn.ReLULayer(),
        fc2=nn.LinearLayer(32),
        act2=nn.ReLULayer(),
        fc3=nn.LinearLayer(n_outputs),
    )
    var y_pred = nn.build_graph(model_def, g, x)
    g.out(y_pred)

    var loss = nn.MSELoss(g, y_pred, y_true)
    g.loss(loss)

    return g ^


def main():
    comptime batch_size = 32
    comptime n_inputs = 1
    comptime n_outputs = 1
    comptime learning_rate = 0.01

    comptime epochs = 20000

    comptime graph = create_simple_nn(batch_size, n_inputs, n_outputs)

    var model = nn.Model[graph]()
    var optimizer = nn.optim.Adam[graph](model.parameters, lr=learning_rate)

    var x_data = Tensor[dtype](batch_size, n_inputs)
    var y_data = Tensor[dtype](batch_size, n_outputs)

    print("Training started")
    var start = now()
    for i in range(epochs):
        rand[dtype](x_data.mut_ptr(), x_data.num_elements())

        for j in range(batch_size):
            x_data[j] = x_data[j] * 2 - 1
            y_data[j] = math.sin(x_data[j])

        ref out = model.forward(x_data.copy(), y_data.copy())

        if (i + 1) % 1000 == 0:
            print("[", i + 1, "/", epochs, "] \tLoss: ", out[0])

        optimizer.zero_grad()
        model.backward()
        optimizer.step()

    print("Training finished: ", Float64(now() - start) / 1e9, "seconds")

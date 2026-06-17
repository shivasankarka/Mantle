from std.time import perf_counter_ns as now
from std.utils.index import IndexList

import basalt.nn as nn
from basalt import Tensor, TensorShape
from basalt import Graph, Symbol, OP, f32
from basalt.utils.datasets import MNIST
from basalt.utils.dataloader import DataLoader
from basalt.autograd.attributes import AttributeVector, Attribute


@fieldwise_init
struct CNN(Copyable, Movable):
    var conv1: nn.Conv2dLayer
    var act1: nn.ReLULayer
    var pool1: nn.MaxPool2dLayer
    var conv2: nn.Conv2dLayer
    var act2: nn.ReLULayer
    var pool2: nn.MaxPool2dLayer
    var flatten: nn.FlattenLayer
    var fc: nn.LinearLayer


def create_CNN(batch_size: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch_size, 1, 28, 28))

    var model_def = CNN(
        conv1=nn.Conv2dLayer(
            out_channels=16, kernel_size=IndexList[2](5, 5), padding=IndexList[2](2, 2)
        ),
        act1=nn.ReLULayer(),
        pool1=nn.MaxPool2dLayer(kernel_size=IndexList[2](2, 2)),
        conv2=nn.Conv2dLayer(
            out_channels=32, kernel_size=IndexList[2](5, 5), padding=IndexList[2](2, 2)
        ),
        act2=nn.ReLULayer(),
        pool2=nn.MaxPool2dLayer(kernel_size=IndexList[2](2, 2)),
        flatten=nn.FlattenLayer(),
        fc=nn.LinearLayer(10),
    )
    var out = nn.build_graph(model_def, g, x)
    g.out(out)

    var y_true = g.input(TensorShape(batch_size, 10))
    var loss = nn.CrossEntropyLoss(g, out, y_true)
    g.loss(loss)

    return g ^


def main():
    comptime num_epochs = 20
    comptime batch_size = 4
    comptime learning_rate = 1e-3

    comptime graph = create_CNN(batch_size)

    var model = nn.Model[graph]()
    var optim = nn.optim.Adam[graph](model.parameters, lr=learning_rate)

    print("Loading data ...")
    var train_data: MNIST
    try:
        train_data = MNIST(file_path="./examples/data/mnist_test_small.csv")
    except e:
        print("Could not load data")
        print(e)
        return

    var training_loader = DataLoader(
        data=train_data.data, labels=train_data.labels, batch_size=batch_size
    )

    print("Training started.")
    var start = now()

    for epoch in range(num_epochs):
        var num_batches: Int = 0
        var epoch_loss: Float32 = 0.0
        var epoch_start = now()
        for batch in training_loader:
            # [ONE HOT ENCODING!]
            var labels_one_hot = Tensor[f32](batch.labels.dim(0), 10)
            for bb in range(batch.labels.dim(0)):
                labels_one_hot[bb * 10 + Int(batch.labels[bb])] = 1.0

            # Forward pass
            var loss = model.forward(batch.data, labels_one_hot)

            # Backward pass
            optim.zero_grad()
            model.backward()
            optim.step()

            epoch_loss += loss[0]
            num_batches += 1

            print(
                "Epoch [",
                epoch + 1,
                "/",
                num_epochs,
                "],\t Step [",
                num_batches,
                "/",
                train_data.data.dim(0) // batch_size,
                "],\t Loss:",
                epoch_loss / Float32(num_batches),
            )

        print("Epoch time: ", Float64(now() - epoch_start) / 1e9, "seconds")

    print("Training finished: ", Float64(now() - start) / 1e9, "seconds")

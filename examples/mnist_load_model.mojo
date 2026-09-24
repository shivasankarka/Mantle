from std.time import perf_counter_ns as now
from std.pathlib import Path
from std.utils.index import IndexList

import mantle.nn as nn
from mantle import Tensor, TensorShape
from mantle import Graph, Symbol, OP, f32
from mantle.data.datasets import MNIST
from mantle.data.dataloader import DataLoader
from mantle.autograd.attributes import AttributeVector, Attribute


# def plot_image(data: Tensor, num: Int):
#     from python.python import Python, PythonObject

#     np = Python.import_module("numpy")
#     plt = Python.import_module("matplotlib.pyplot")

#     var pyimage: PythonObject = np.empty((28, 28), np.float64)
#     for m in range(28):
#         for n in range(28):
#             pyimage.itemset((m, n), data[num * 28 * 28 + m * 28 + n])

#     plt.imshow(pyimage)
#     plt.show()


def create_CNN(batch_size: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch_size, 1, 28, 28))

    var x1 = nn.Conv2d(
        g,
        x,
        out_channels=16,
        kernel_size=IndexList[2](5, 5),
        padding=IndexList[2](2, 2),
    )
    var x2 = nn.ReLU(g, x1)
    var x3 = nn.MaxPool2d(g, x2, kernel_size=IndexList[2](2, 2))
    var x4 = nn.Conv2d(
        g,
        x3,
        out_channels=32,
        kernel_size=IndexList[2](5, 5),
        padding=IndexList[2](2, 2),
    )
    var x5 = nn.ReLU(g, x4)
    var x6 = nn.MaxPool2d(g, x5, kernel_size=IndexList[2](2, 2))
    var x7 = g.op(
        OP.RESHAPE,
        x6,
        attributes=AttributeVector(
            Attribute(
                "shape",
                TensorShape(
                    x6.shape[0], x6.shape[1] * x6.shape[2] * x6.shape[3]
                ),
            )
        ),
    )
    var out = nn.Linear(g, x7, n_outputs=10)
    g.out(out)

    return g^


def main() raises:
    comptime num_epochs = 1
    comptime batch_size = 4
    comptime learning_rate = 1e-3

    comptime graph = create_CNN(batch_size)

    # try: graph.render("operator")
    # except: print("Could not render graph")

    var model = nn.Model[graph](inference_only=True)
    model.load_model_data("./examples/data/mnist_torch.onnx")

    print("Loading data ...")
    var train_data: MNIST
    try:
        train_data = MNIST(file_path="./examples/data/mnist_test_small.csv")
        # _ = plot_image(train_data.data, 1)
    except e:
        print("Could not load data")
        print(e)
        return

    var training_loader = DataLoader(
        data=train_data.data, labels=train_data.labels, batch_size=batch_size
    )

    # Testing
    print("Testing started")
    var start = now()

    var correct: Scalar[f32] = 0.0
    var total = 0
    for batch in training_loader:
        var labels_one_hot = Tensor[f32](batch.labels.dim(0), 10)
        for bb in range(batch.labels.dim(0)):
            labels_one_hot[bb * 10 + Int(batch.labels[bb])] = 1.0

        var output = model.inference(batch.data)[0].copy()
        correct += nn.accuracy(output, labels_one_hot) * Float32(
            batch.labels.dim(0)
        )
        total += batch.labels.dim(0)

    print("Inference accuracy: ", 100.0 * correct / Float32(total), "%")
    print("Testing finished: ", Float64(now() - start) / 1e9, "seconds")

    # model.print_perf_metrics("ms", True)

    # Keep this example read-only: it evaluates an imported model and must not
    # overwrite an ONNX artifact as a side effect.

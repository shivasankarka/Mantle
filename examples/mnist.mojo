from std.time import perf_counter_ns as now
import mantle.nn as nn
from mantle import Tensor, TensorShape
from mantle import Graph, f32
from mantle.data.datasets import MNIST
from mantle.data.dataloader import DataLoader


@fieldwise_init
struct MNISTCNN(Copyable, Movable):
    """A reflected, PyTorch-style model definition."""

    var conv1: nn.Conv2dLayer
    var relu1: nn.ReLULayer
    var pool1: nn.MaxPool2dLayer
    var conv2: nn.Conv2dLayer
    var relu2: nn.ReLULayer
    var pool2: nn.MaxPool2dLayer
    var flatten: nn.FlattenLayer
    var head: nn.LinearLayer


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
    var network = MNISTCNN(
        nn.Conv2d(16, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(10),
    )
    return nn.classification_graph(
        network, TensorShape(batch_size, 1, 28, 28)
    )


def main() raises:
    comptime num_epochs = 20
    comptime batch_size = 4
    comptime learning_rate = 1e-3

    comptime graph = create_CNN(batch_size)

    # try: graph.render("operator")
    # except: print("Could not render graph")

    var model = nn.Model[graph]()
    var optim = nn.optim.Adam[graph](model.parameters, lr=learning_rate)

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

    print("Training started/")
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
            "Epoch ",
            epoch + 1,
            "/",
            num_epochs,
            " — loss: ",
            epoch_loss / Float32(num_batches),
            " — time: ",
            Float64(now() - epoch_start) / 1e9,
            " seconds",
        )

    print("Training finished: ", Float64(now() - start) / 1e9, "seconds")

    # This bundled CSV is the training sample, not a held-out test split.
    # Running inference over it still verifies the complete prediction path
    # and makes overfitting/regressions visible in the example output.
    var correct: Scalar[f32] = 0.0
    var total = 0
    var inference_start = now()
    for batch in training_loader:
        var labels_one_hot = Tensor[f32](batch.labels.dim(0), 10)
        for bb in range(batch.labels.dim(0)):
            labels_one_hot[bb * 10 + Int(batch.labels[bb])] = 1.0

        var logits = model.predict(batch.data)
        correct += nn.accuracy(logits, labels_one_hot) * Float32(
            batch.labels.dim(0)
        )
        total += batch.labels.dim(0)

    print(
        "Training-set inference accuracy: ",
        100.0 * correct / Float32(total),
        "% (",
        total,
        " images, ",
        Float64(now() - inference_start) / 1e9,
        " seconds)",
    )

    model.print_perf_metrics("ms", True)

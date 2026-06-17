from std.time import perf_counter_ns as now

import mantle.nn as nn
from mantle import Tensor, TensorShape, f32
from mantle import Graph, Symbol, OP
from mantle.utils.datasets import BostonHousing
from mantle.utils.dataloader import DataLoader, slice_rows, cycle_pad_rows


def linear_regression(batch_size: Int, n_inputs: Int, n_outputs: Int) -> Graph:
    var g = Graph()

    var x = g.input(TensorShape(batch_size, n_inputs))
    var y_true = g.input(TensorShape(batch_size, n_outputs))

    var model_def = nn.Sequential(nn.LinearLayer(n_outputs))
    var y_pred = model_def.forward(g, x)
    g.out(y_pred)

    var loss = nn.MSELoss(g, y_pred, y_true)
    g.loss(loss)

    return g^


def main():
    comptime batch_size = 32
    comptime num_epochs = 20
    comptime learning_rate = 0.02
    comptime train_pct = 0.99

    comptime graph = linear_regression(batch_size, 13, 1)

    var model = nn.Model[graph]()
    var optim = nn.optim.Adam[graph](model.parameters, lr=learning_rate)

    print("Loading data...")
    var dataset: BostonHousing
    try:
        dataset = BostonHousing(file_path="./examples/data/housing.csv")
    except:
        print("Could not load data")
        return

    # Train/test split (matches housing.py's TRAIN_PCT split).
    var n_total = dataset.data.dim(0)
    var n_train = Int(train_pct * Float64(n_total))
    var n_test = n_total - n_train

    var train_x = slice_rows(dataset.data, 0, n_train)
    var train_y = slice_rows(dataset.labels, 0, n_train)
    var test_x = slice_rows(dataset.data, n_train, n_test)
    var test_y = slice_rows(dataset.labels, n_train, n_test)

    var training_loader = DataLoader(
        data=train_x, labels=train_y, batch_size=batch_size
    )

    print("Training started.")
    var start = now()
    for epoch in range(num_epochs):
        var num_batches: Int = 0
        var epoch_loss: Float32 = 0.0
        for batch in training_loader:
            var loss = model.forward(batch.data.copy(), batch.labels.copy()).copy()

            optim.zero_grad()
            model.backward()
            optim.step()

            epoch_loss += loss[0]
            num_batches += 1

        print(
            "Epoch: [",
            epoch + 1,
            "/",
            num_epochs,
            "] \t Avg loss per epoch:",
            epoch_loss / Float32(num_batches),
        )

    print("Training finished: ", Float64(now() - start) / 1e9, "seconds")

    # Evaluate on the held-out test set. The model's graph has a fixed
    # batch_size, so the (much smaller) test set is cycled to fill one
    # full batch; the MSE below is only averaged over the `n_test` real
    # rows, ignoring the cycled padding.
    # model.inference() requires a tensor for every graph input in
    # declared order, even ones (like y_true) that the inference subgraph
    # itself doesn't read.
    var test_x_padded = cycle_pad_rows(test_x, batch_size)
    var dummy_y = Tensor[f32](TensorShape(batch_size, 1))
    var inference_outputs = model.inference(test_x_padded, dummy_y)
    var predictions = inference_outputs[0].copy()

    var mse: Float32 = 0.0
    for i in range(n_test):
        var diff = predictions[i] - test_y[i]
        mse += diff * diff
    mse /= Float32(n_test)

    print("Mean Squared Error on Test Data:", mse)

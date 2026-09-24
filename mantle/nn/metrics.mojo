# ===----------------------------------------------------------------------=== #
# Mantle: Metrics
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Metrics (mantle.nn.metrics)
------------------------------------------------
Evaluation metrics computed directly on host tensors (not graph ops) — meant
to be called on `logits`/`targets` pulled out after a forward pass, the same
way a PyTorch training loop calls `.item()`/`accuracy()` outside the graph.
"""
from mantle.core.tensor import Tensor
from mantle import f32


def accuracy(logits: Tensor[f32], targets: Tensor[f32]) -> Scalar[f32]:
    """
    Classification accuracy: fraction of rows where `argmax(logits, -1) ==
    argmax(targets, -1)`.

    Both `logits` and `targets` are rank-2 `(batch, num_classes)` — `targets`
    one-hot (or smoothed one-hot; argmax is unaffected by smoothing since it
    preserves the max-probability class), matching `CrossEntropyLoss`'s
    expected shape.

    Args:
        logits: Model output, `(batch, num_classes)`.
        targets: One-hot (or smoothed one-hot) labels, `(batch, num_classes)`.

    Returns:
        The fraction of correctly-classified rows, in `[0, 1]`.
    """
    var batch = logits.shape()[0]
    var num_classes = logits.shape()[1]
    if batch == 0:
        return 0.0

    var correct: Scalar[f32] = 0
    for i in range(batch):
        var base = i * num_classes

        var pred_idx = 0
        var pred_val = logits[base]
        var true_idx = 0
        var true_val = targets[base]
        for j in range(1, num_classes):
            var l = logits[base + j]
            if l > pred_val:
                pred_val = l
                pred_idx = j
            var t = targets[base + j]
            if t > true_val:
                true_val = t
                true_idx = j

        if pred_idx == true_idx:
            correct += 1

    return correct / Scalar[f32](batch)

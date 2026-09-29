"""GPU CONCAT/SPLIT: forward/backward parity against the CPU kernels.

Notes:
    A full LSTM/GRU graph (see test_recurrent.mojo) unrolls one CONCAT and
    one SPLIT per timestep, which generates mangled symbol names long
    enough to crash this machine's macOS linker (`ld: Assertion failed:
    name.size() <= maxLength`) independent of device — this predates the
    GPU work and reproduces identically on an unmodified checkout. This
    test isolates CONCAT/SPLIT on a small graph that links fine, to verify
    the GPU kernels directly.
"""
from std.random import rand
from std.testing import assert_true

from mantle import Graph, Tensor, TensorShape, f32
from mantle.core.device import Device
from mantle.nn.model import Model
import mantle.nn as nn
import mantle.nn.optim as optim
from mantle.nn.optim import get_trainable_parameters


def make_concat_split_graph() -> Graph:
    var g = Graph()
    var a = g.input(TensorShape(2, 3))
    var b = g.input(TensorShape(2, 5))
    var y = g.input(TensorShape(2, 8))
    var combined = g.concat(a, b, dim=1)
    var parts = g.split(combined, sections=[3, 5], dim=1)
    var recombined = g.concat(parts[0], parts[1], dim=1)
    var predictions = nn.Linear(g, recombined, n_outputs=8)
    g.out(predictions)
    g.loss(nn.MSELoss(g, predictions, y))
    return g^


def test_gpu_concat_split_matches_cpu() raises:
    comptime graph = make_concat_split_graph()
    comptime trainable = get_trainable_parameters(graph)
    var host_a = Tensor[f32](TensorShape(2, 3))
    var host_b = Tensor[f32](TensorShape(2, 5))
    var host_y = Tensor[f32](TensorShape(2, 8))
    rand[f32](host_a.ptr(), host_a.num_elements())
    rand[f32](host_b.ptr(), host_b.num_elements())
    rand[f32](host_y.ptr(), host_y.num_elements())

    var cpu_model = Model[graph]()
    var cpu_loss = cpu_model.forward(host_a, host_b, host_y)
    cpu_model.backward()

    var a = host_a.to_gpu()
    var b = host_b.to_gpu()
    var y = host_y.to_gpu()
    var gpu_model = Model[graph, device=Device.gpu]()
    # Same initial parameters on both devices for a fair comparison.
    comptime for symbol in trainable:
        var cpu_param = cpu_model.parameters.tensors[symbol]
        gpu_model.parameters.tensors[symbol] = cpu_param.to_gpu()
    var gpu_loss = gpu_model.forward(a, b, y)
    gpu_model.backward()
    a.gpu_context().synchronize()

    var gpu_loss_host = gpu_loss.to_host()
    var difference = abs(Float32(cpu_loss[0]) - Float32(gpu_loss_host[0]))
    assert_true(difference < 1e-3, "GPU CONCAT/SPLIT loss must match CPU")

    comptime for symbol in trainable:
        var cpu_grad = cpu_model.parameters.grads[symbol]
        var gpu_grad = gpu_model.parameters.grads[symbol].to_host()
        for j in range(cpu_grad.num_elements()):
            var grad_difference = abs(
                Float32(cpu_grad[j]) - Float32(gpu_grad[j])
            )
            assert_true(
                grad_difference < 1e-3,
                "GPU CONCAT/SPLIT param gradient must match CPU",
            )

    print("test_gpu_concat_split_matches_cpu: PASSED")


def main() raises:
    test_gpu_concat_split_matches_cpu()

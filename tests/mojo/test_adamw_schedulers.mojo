"""Test AdamW optimizer and LR schedulers (WarmupCosineSchedule, StepDecaySchedule)."""
from std.testing import assert_true, assert_almost_equal

from mantle import f32
from mantle.autograd.graph import Graph
from mantle.core.tensor import Tensor, TensorShape
from mantle.core.tensorutils import fill
import mantle.nn as nn
from mantle.nn.model import Model
import mantle.nn.optim as optim


def make_linear_graph(batch_size: Int, n_in: Int, n_out: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch_size, n_in))
    var y_true = g.input(TensorShape(batch_size, n_out))
    var y_pred = nn.Linear(g, x, n_out)
    g.out(y_pred)
    var loss = nn.MSELoss(g, y_pred, y_true)
    g.loss(loss)
    return g^


# ===----------------------------------------------------------------------===#
# AdamW
# ===----------------------------------------------------------------------===#


def test_adamw_trains() raises:
    comptime g = make_linear_graph(8, 4, 1)
    var model = Model[g]()
    var adamw = optim.AdamW[g](model.parameters, lr=0.01, weight_decay=0.01)

    var x = Tensor[f32](TensorShape(8, 4))
    var y = Tensor[f32](TensorShape(8, 1))
    fill(x, 1.0)
    fill(y, 2.0)

    var initial_loss: Float32 = model.forward(x, y)[0]
    for _ in range(10):
        _ = model.forward(x, y)
        model.backward()
        adamw.step()
        adamw.zero_grad()

    var final_loss: Float32 = model.forward(x, y)[0]
    assert_true(
        final_loss < initial_loss,
        "AdamW: loss should decrease (got " + String(initial_loss) + " -> " + String(final_loss) + ")",
    )
    print("test_adamw_trains: PASSED (loss " + String(initial_loss) + " -> " + String(final_loss) + ")")


def test_adamw_weight_decay_shrinks_params() raises:
    # With zero gradient (y == prediction impossible to reach exactly, but we
    # zero the grads manually after backward) weight decay alone should
    # shrink parameter magnitude on each step.
    comptime g = make_linear_graph(8, 4, 1)
    var model = Model[g]()
    comptime param_sym = g.params.symbols[0]

    # Give the weight tensor a known nonzero value.
    fill(model.parameters.tensors[param_sym], 1.0)

    var adamw = optim.AdamW[g](
        model.parameters, lr=0.1, beta1=0.0, beta2=0.0, weight_decay=0.5
    )

    var x = Tensor[f32](TensorShape(8, 4))
    var y = Tensor[f32](TensorShape(8, 1))
    fill(x, 1.0)
    fill(y, 2.0)

    _ = model.forward(x, y)
    model.backward()
    # Zero out the gradient so only weight decay affects the param.
    model.parameters.grads.set_zero()
    adamw.step()

    var w = model.parameters.tensors[param_sym][0]
    # param = param - lr * weight_decay * param = 1.0 - 0.1*0.5*1.0 = 0.95
    assert_almost_equal(w, Float32(0.95), rtol=1e-4)
    print("test_adamw_weight_decay_shrinks_params: PASSED (w=" + String(w) + ")")


# ===----------------------------------------------------------------------===#
# WarmupCosineSchedule
# ===----------------------------------------------------------------------===#


def test_warmup_cosine_schedule() raises:
    var sched = optim.WarmupCosineSchedule(
        base_lr=1.0, warmup_steps=10, total_steps=110, min_lr=0.0
    )

    # Linear warmup: step 0 -> 1/10 of base_lr, step 9 -> 10/10 of base_lr
    assert_almost_equal(sched.get_lr(0), Float32(0.1), rtol=1e-4)
    assert_almost_equal(sched.get_lr(9), Float32(1.0), rtol=1e-4)

    # Cosine decay: at the start of decay (step 10), lr == base_lr
    assert_almost_equal(sched.get_lr(10), Float32(1.0), rtol=1e-2)

    # At the very end (step 109, last decay step), lr approaches min_lr
    assert_true(sched.get_lr(109) < 0.05, "lr should approach min_lr at the end")

    # Midpoint of decay: lr should be roughly half of base_lr (cosine midpoint)
    var mid = sched.get_lr(10 + 50)
    assert_true(mid > 0.3 and mid < 0.7, "midpoint lr should be roughly half base_lr")

    print("test_warmup_cosine_schedule: PASSED")


def test_warmup_cosine_schedule_no_warmup() raises:
    var sched = optim.WarmupCosineSchedule(
        base_lr=1.0, warmup_steps=0, total_steps=100, min_lr=0.1
    )
    assert_almost_equal(sched.get_lr(0), Float32(1.0), rtol=1e-2)
    assert_almost_equal(sched.get_lr(100), Float32(0.1), rtol=1e-2)
    print("test_warmup_cosine_schedule_no_warmup: PASSED")


# ===----------------------------------------------------------------------===#
# StepDecaySchedule
# ===----------------------------------------------------------------------===#


def test_step_decay_schedule() raises:
    var sched = optim.StepDecaySchedule(base_lr=1.0, step_size=10, gamma=0.1)

    assert_almost_equal(sched.get_lr(0), Float32(1.0), rtol=1e-4)
    assert_almost_equal(sched.get_lr(9), Float32(1.0), rtol=1e-4)
    assert_almost_equal(sched.get_lr(10), Float32(0.1), rtol=1e-4)
    assert_almost_equal(sched.get_lr(20), Float32(0.01), rtol=1e-4)

    print("test_step_decay_schedule: PASSED")


def main() raises:
    test_adamw_trains()
    test_adamw_weight_decay_shrinks_params()
    test_warmup_cosine_schedule()
    test_warmup_cosine_schedule_no_warmup()
    test_step_decay_schedule()

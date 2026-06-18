"""Graph visualization examples: MLP, multi-branch, skip-connection, ConvNet."""
import mantle.nn as nn
from mantle.core.tensor import TensorShape
from mantle import Graph, OP
from std.utils.index import IndexList


# ===----------------------------------------------------------------------===#
# 1) 3-layer MLP
# ===----------------------------------------------------------------------===#

@fieldwise_init
struct MLP(Copyable, Movable):
    var fc1: nn.LinearLayer
    var act1: nn.ReLULayer
    var fc2: nn.LinearLayer
    var act2: nn.ReLULayer
    var fc3: nn.LinearLayer

def build_mlp(batch_size: Int, n_inputs: Int, n_outputs: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch_size, n_inputs))
    var y_true = g.input(TensorShape(batch_size, n_outputs))

    var model_def = MLP(
        nn.LinearLayer(128),
        nn.ReLULayer(),
        nn.LinearLayer(64),
        nn.ReLULayer(),
        nn.LinearLayer(n_outputs),
    )
    var y_pred = nn.build_graph(model_def, g, x)
    g.out(y_pred)
    g.loss(nn.MSELoss(g, y_pred, y_true))
    return g^


# ===----------------------------------------------------------------------===#
# 2) Multi-branch
# ===----------------------------------------------------------------------===#

@fieldwise_init
struct BranchA(Copyable, Movable):
    var fc: nn.LinearLayer

@fieldwise_init
struct BranchB(Copyable, Movable):
    var fc: nn.LinearLayer

def build_multi_branch(batch_size: Int, n_inputs: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch_size, n_inputs))
    var y_true = g.input(TensorShape(batch_size, 64))

    var branch_a = BranchA(nn.LinearLayer(64))
    var branch_b = BranchB(nn.LinearLayer(64))
    var b1 = nn.build_graph(branch_a, g, x)
    var b2 = nn.build_graph(branch_b, g, x)

    var merged = g.op(OP.ADD, b1, b2)
    g.out(merged)
    g.loss(nn.MSELoss(g, merged, y_true))
    return g^


# ===----------------------------------------------------------------------===#
# 3) Skip connection (residual)
# ===----------------------------------------------------------------------===#

@fieldwise_init
struct SkipBlock(Copyable, Movable):
    var fc: nn.LinearLayer

def build_skip(batch_size: Int, n_inputs: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch_size, n_inputs))
    var y_true = g.input(TensorShape(batch_size, n_inputs))

    var block = SkipBlock(nn.LinearLayer(n_inputs))
    var out = nn.build_graph(block, g, x)
    var residual = g.op(OP.ADD, out, x)
    g.out(residual)
    g.loss(nn.MSELoss(g, residual, y_true))
    return g^


# ===----------------------------------------------------------------------===#
# 4) Tiny ConvNet
# ===----------------------------------------------------------------------===#

@fieldwise_init
struct TinyConv(Copyable, Movable):
    var conv: nn.Conv2dLayer
    var act: nn.ReLULayer
    var pool: nn.MaxPool2dLayer

def build_conv(batch_size: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch_size, 1, 28, 28))
    var y_true = g.input(TensorShape(batch_size, 10))

    var model_def = TinyConv(
        nn.Conv2dLayer(16, IndexList[2](3, 3)),
        nn.ReLULayer(),
        nn.MaxPool2dLayer(IndexList[2](2, 2)),
    )
    var features = nn.build_graph(model_def, g, x)
    var flat = g.op(OP.FLATTEN, features)
    var logits = g.op(OP.DOT, flat, g.param(TensorShape(2704, 10)))
    g.out(logits)
    g.loss(nn.CrossEntropyLoss(g, logits, y_true))
    return g^


# ===----------------------------------------------------------------------===#
# main
# ===----------------------------------------------------------------------===#

def main():
    comptime batch_size = 32

    print("=" * 60)
    print("1)  3-Layer MLP")
    print("=" * 60)
    comptime mlp = build_mlp(batch_size, 13, 1)
    print(" execution ")
    mlp.visualize("execution")
    print()
    print(" architecture ")
    mlp.visualize("architecture")
    print()

    print("=" * 60)
    print("2)  Multi-branch (fork + merge)")
    print("=" * 60)
    comptime mb = build_multi_branch(batch_size, 13)
    print(" execution ")
    mb.visualize("execution")
    print()
    print(" architecture ")
    mb.visualize("architecture")
    print()

    print("=" * 60)
    print("3)  Skip connection (residual)")
    print("=" * 60)
    comptime skip = build_skip(batch_size, 13)
    print(" execution ")
    skip.visualize("execution")
    print()
    print(" architecture ")
    skip.visualize("architecture")
    print()

    print("=" * 60)
    print("4)  Tiny ConvNet")
    print("=" * 60)
    comptime conv = build_conv(batch_size)
    print(" execution ")
    conv.visualize("execution")
    print()
    print(" architecture ")
    conv.visualize("architecture")
    print()

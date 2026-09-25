"""Regression test for custom modules built without graph arguments."""
from std.testing import assert_true

import mantle.nn as nn
from mantle import Graph, Tensor, TensorShape, f32


@fieldwise_init
struct ResidualMLP(nn.Module, Movable):
    var hidden: nn.LinearLayer
    var output: nn.LinearLayer

    def forward(mut self, input: nn.Expr) -> nn.Expr:
        return self.output(self.hidden(input).relu()) + input


@fieldwise_init
struct ResidualConv(nn.Module, Movable):
    var conv1: nn.Conv2dLayer
    var conv2: nn.Conv2dLayer

    def forward(mut self, input: nn.Expr) -> nn.Expr:
        return self.conv2(self.conv1(input).relu()) + input


@fieldwise_init
struct TinyTransformer(nn.Module, Movable):
    var block: nn.TransformerBlockLayer

    def forward(mut self, input: nn.Expr) -> nn.Expr:
        return self.block(input)


@fieldwise_init
struct TinyTokenTransformer(nn.Module, Movable):
    var tokens: nn.EmbeddingLayer
    var positions: nn.PositionalEmbeddingLayer
    var block: nn.TransformerBlockLayer
    var norm: nn.LayerNormLayer

    def forward(mut self, input: nn.Expr) -> nn.Expr:
        var x = self.tokens(input)
        x = self.positions(x)
        x = self.block(x)
        return self.norm(x)


struct ReshapeModule(nn.Module, Copyable, Movable):
    def __init__(out self):
        pass

    def forward(mut self, input: nn.Expr) -> nn.Expr:
        return input.transpose(TensorShape(0, 2, 1)).reshape(TensorShape(2, 4))


def make_graph(batch_size: Int) -> Graph:
    var architecture = ResidualMLP(nn.Linear(2), nn.Linear(2))
    return nn.classification_graph(
        architecture, TensorShape(batch_size, 2)
    )


def make_conv_graph(batch_size: Int) -> Graph:
    var architecture = ResidualConv(
        nn.Conv2d(1, kernel_size=1), nn.Conv2d(1, kernel_size=1)
    )
    return nn.build_module_graph(
        architecture, TensorShape(batch_size, 1, 2, 2)
    )


def make_transformer_graph(batch_size: Int) -> Graph:
    var architecture = TinyTransformer(
        nn.TransformerBlock(
            num_heads=2, d_ff=8, dropout_p=0.0, causal=True
        )
    )
    return nn.build_module_graph(
        architecture, TensorShape(batch_size, 2, 4)
    )


def make_token_transformer_graph(batch_size: Int) -> Graph:
    var architecture = TinyTokenTransformer(
        nn.Embedding(8, 4),
        nn.PositionalEmbedding(2),
        nn.TransformerBlock(num_heads=2, d_ff=8, dropout_p=0.0, causal=True),
        nn.LayerNorm(4),
    )
    return nn.build_module_graph(architecture, TensorShape(batch_size, 2))


def make_reshape_graph() -> Graph:
    var architecture = ReshapeModule()
    return nn.build_module_graph(architecture, TensorShape(2, 2, 2))


def test_custom_module_builds_static_graph() raises:
    comptime graph = make_graph(2)
    comptime logits = graph.outputs[0]
    assert_true(
        logits.shape == TensorShape(2, 2),
        "residual module should preserve the feature shape",
    )

    var model = nn.Model[graph]()
    var inputs = Tensor[f32](TensorShape(2, 2))
    var labels = Tensor[f32](TensorShape(2, 2))
    var loss = model.forward(inputs, labels)
    assert_true(loss[0] == loss[0], "module graph should execute finitely")
    print("test_custom_module_builds_static_graph: PASSED")


def test_custom_conv_module_builds_static_graph() raises:
    comptime graph = make_conv_graph(2)
    comptime output = graph.outputs[0]
    assert_true(
        output.shape == TensorShape(2, 1, 2, 2),
        "residual convolution should preserve its input shape",
    )

    var model = nn.Model[graph](inference_only=True)
    var inputs = Tensor[f32](TensorShape(2, 1, 2, 2))
    var result = model.inference(inputs)[0].copy()
    assert_true(result[0] == result[0], "module graph should execute finitely")
    print("test_custom_conv_module_builds_static_graph: PASSED")


def test_custom_transformer_module_builds_static_graph() raises:
    comptime graph = make_transformer_graph(2)
    comptime output = graph.outputs[0]
    assert_true(
        output.shape == TensorShape(2, 2, 4),
        "transformer block should preserve the sequence shape",
    )

    var model = nn.Model[graph](inference_only=True)
    var inputs = Tensor[f32](TensorShape(2, 2, 4))
    var result = model.inference(inputs)[0].copy()
    assert_true(result[0] == result[0], "transformer graph should execute finitely")
    print("test_custom_transformer_module_builds_static_graph: PASSED")


def test_custom_token_transformer_builds_static_graph() raises:
    comptime graph = make_token_transformer_graph(2)
    comptime output = graph.outputs[0]
    assert_true(
        output.shape == TensorShape(2, 2, 4),
        "token transformer should map tokens to sequence embeddings",
    )

    var model = nn.Model[graph](inference_only=True)
    var tokens = Tensor[f32](TensorShape(2, 2))
    var result = model.inference(tokens)[0].copy()
    assert_true(result[0] == result[0], "token transformer should execute finitely")
    print("test_custom_token_transformer_builds_static_graph: PASSED")


def test_expr_shape_helpers_build_static_graph() raises:
    comptime graph = make_reshape_graph()
    comptime output = graph.outputs[0]
    assert_true(
        output.shape == TensorShape(2, 4),
        "Expr transpose and reshape should compose in a module",
    )
    print("test_expr_shape_helpers_build_static_graph: PASSED")


def main() raises:
    test_custom_module_builds_static_graph()
    test_custom_conv_module_builds_static_graph()
    test_custom_transformer_module_builds_static_graph()
    test_custom_token_transformer_builds_static_graph()
    test_expr_shape_helpers_build_static_graph()

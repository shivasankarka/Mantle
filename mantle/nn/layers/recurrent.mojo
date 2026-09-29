# ===----------------------------------------------------------------------=== #
# Mantle: Recurrent Layers
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Recurrent (mantle.nn.layers.recurrent)
------------------------------------------------
LSTM and GRU layers, built by unrolling one shared-weight cell per timestep
over a static graph.
"""
from mantle import f32
from mantle.autograd.graph import Graph
from mantle.autograd.symbol import Symbol
from mantle.autograd.ops import OP
from mantle.core.tensor import Tensor, TensorShape
from mantle.core.math_util import q_sqrt
from mantle.autograd.params import Param
from mantle.autograd.attributes import AttributeVector, Attribute
from mantle.nn.module import Expr, Layer, Module
from mantle.nn.activations import Sigmoid, Tanh


# ===----------------------------------------------------------------------===#
# LSTM (functional)
# ===----------------------------------------------------------------------===#


def LSTM(mut g: Graph, inputs: Symbol, hidden_size: Int) -> Symbol:
    """
    A single-layer LSTM over `inputs` of shape `(B, T, D)`, returning the
    full output sequence `(B, T, hidden_size)`.

    Notes:
        The static graph unrolls one set of gate/state nodes per timestep,
        sharing one weight matrix across all of them, so graph size and
        build time scale with `T`.
    """
    var batch = inputs.shape[0]
    var seq_len = inputs.shape[1]
    var input_size = inputs.shape[2]

    var fan_in: Scalar[f32] = Scalar[f32](input_size + hidden_size)
    var bound = q_sqrt(fan_in)
    var weights = g.param(
        TensorShape(input_size + hidden_size, 4 * hidden_size),
        init=Param("random_uniform", -bound, bound),
    )
    var bias = g.param(
        TensorShape(4 * hidden_size),
        init=Param("random_uniform", -bound, bound),
    )

    # Zero-initialized, non-trainable initial state. The second `0.0` is
    # unused by `initialize_tensor`'s "constant" case but required because
    # `Model.allocate_tensor_memory` always reads two initializer args.
    var h = g.param(
        TensorShape(batch, hidden_size),
        init=Param("constant", 0.0, 0.0),
        trainable=False,
    )
    var c = g.param(
        TensorShape(batch, hidden_size),
        init=Param("constant", 0.0, 0.0),
        trainable=False,
    )

    def step(
        t: Int,
    ) {mut g, mut h, mut c, imm inputs, imm weights, imm bias, imm hidden_size} -> Symbol:
        var x_t_slice = g.op(
            OP.SLICE,
            inputs,
            attributes=AttributeVector(
                Attribute("starts", TensorShape(t)),
                Attribute("ends", TensorShape(t + 1)),
                Attribute("axes", TensorShape(1)),
            ),
        )
        var x_t = g.op(
            OP.SQUEEZE,
            x_t_slice,
            attributes=AttributeVector(Attribute("dims", TensorShape(1))),
        )
        var combined = g.concat(x_t, h, dim=1)
        var gates = g.op(OP.LINEAR, combined, weights, bias)
        var chunks = g.split(
            gates,
            sections=[hidden_size, hidden_size, hidden_size, hidden_size],
            dim=1,
        )
        var i_t = Sigmoid(g, chunks[0])
        var f_t = Sigmoid(g, chunks[1])
        var g_t = Tanh(g, chunks[2])
        var o_t = Sigmoid(g, chunks[3])
        c = g.op(OP.ADD, g.op(OP.MUL, f_t, c), g.op(OP.MUL, i_t, g_t))
        h = g.op(OP.MUL, o_t, Tanh(g, c))
        return g.op(
            OP.UNSQUEEZE,
            h,
            attributes=AttributeVector(Attribute("dims", TensorShape(1))),
        )

    var outputs_seq = step(0)
    for t in range(1, seq_len):
        var h_unsq = step(t)
        outputs_seq = g.concat(outputs_seq, h_unsq, dim=1)

    return outputs_seq


def LSTM(hidden_size: Int) -> LSTMLayer:
    """Create an LSTM layer for `Sequential` or a module."""
    return LSTMLayer(hidden_size)


# ===----------------------------------------------------------------------===#
# LSTMLayer
# ===----------------------------------------------------------------------===#


@fieldwise_init
struct LSTMLayer(Copyable, Layer, Module, Movable):
    """
    `Layer`-conforming wrapper around `LSTM`, for use in a reflection-based
    Module struct.
    """

    var hidden_size: Int

    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        return LSTM(g, input, self.hidden_size)

    def forward(mut self, input: Expr) -> Expr:
        return Expr(
            input.graph, LSTM(input.graph[], input.symbol, self.hidden_size)
        )

    def __call__(mut self, input: Expr) -> Expr:
        return self.forward(input)


# ===----------------------------------------------------------------------===#
# GRU (functional)
# ===----------------------------------------------------------------------===#


def GRU(mut g: Graph, inputs: Symbol, hidden_size: Int) -> Symbol:
    """
    A single-layer GRU over `inputs` of shape `(B, T, D)`, returning the
    full output sequence `(B, T, hidden_size)`.

    Notes:
        Input and hidden contributions use separate weight matrices (rather
        than one shared matrix over a concatenated `[x_t, h]`, as `LSTM`
        uses) because the reset gate scales only the hidden contribution to
        the candidate state, matching PyTorch's `GRU` weight layout
        (`weight_ih`/`weight_hh`, each producing 3*hidden_size).
    """
    var batch = inputs.shape[0]
    var seq_len = inputs.shape[1]
    var input_size = inputs.shape[2]

    var bound_i = q_sqrt(Scalar[f32](input_size))
    var weights_i = g.param(
        TensorShape(input_size, 3 * hidden_size),
        init=Param("random_uniform", -bound_i, bound_i),
    )
    var bias_i = g.param(
        TensorShape(3 * hidden_size),
        init=Param("random_uniform", -bound_i, bound_i),
    )

    var bound_h = q_sqrt(Scalar[f32](hidden_size))
    var weights_h = g.param(
        TensorShape(hidden_size, 3 * hidden_size),
        init=Param("random_uniform", -bound_h, bound_h),
    )
    var bias_h = g.param(
        TensorShape(3 * hidden_size),
        init=Param("random_uniform", -bound_h, bound_h),
    )

    var h = g.param(
        TensorShape(batch, hidden_size),
        init=Param("constant", 0.0, 0.0),
        trainable=False,
    )

    def step(
        t: Int,
    ) {mut g, mut h, imm inputs, imm weights_i, imm bias_i, imm weights_h, imm bias_h, imm hidden_size} -> Symbol:
        var x_t_slice = g.op(
            OP.SLICE,
            inputs,
            attributes=AttributeVector(
                Attribute("starts", TensorShape(t)),
                Attribute("ends", TensorShape(t + 1)),
                Attribute("axes", TensorShape(1)),
            ),
        )
        var x_t = g.op(
            OP.SQUEEZE,
            x_t_slice,
            attributes=AttributeVector(Attribute("dims", TensorShape(1))),
        )
        var gates_i = g.op(OP.LINEAR, x_t, weights_i, bias_i)
        var gates_h = g.op(OP.LINEAR, h, weights_h, bias_h)
        var chunks_i = g.split(
            gates_i, sections=[hidden_size, hidden_size, hidden_size], dim=1
        )
        var chunks_h = g.split(
            gates_h, sections=[hidden_size, hidden_size, hidden_size], dim=1
        )
        var r_t = Sigmoid(g, g.op(OP.ADD, chunks_i[0], chunks_h[0]))
        var z_t = Sigmoid(g, g.op(OP.ADD, chunks_i[1], chunks_h[1]))
        var n_t = Tanh(
            g, g.op(OP.ADD, chunks_i[2], g.op(OP.MUL, r_t, chunks_h[2]))
        )
        var one_minus_z = g.op(OP.SUB, 1.0, z_t)
        h = g.op(
            OP.ADD,
            g.op(OP.MUL, one_minus_z, n_t),
            g.op(OP.MUL, z_t, h),
        )
        return g.op(
            OP.UNSQUEEZE,
            h,
            attributes=AttributeVector(Attribute("dims", TensorShape(1))),
        )

    var outputs_seq = step(0)
    for t in range(1, seq_len):
        var h_unsq = step(t)
        outputs_seq = g.concat(outputs_seq, h_unsq, dim=1)

    return outputs_seq


def GRU(hidden_size: Int) -> GRULayer:
    """Create a GRU layer for `Sequential` or a module."""
    return GRULayer(hidden_size)


# ===----------------------------------------------------------------------===#
# GRULayer
# ===----------------------------------------------------------------------===#


@fieldwise_init
struct GRULayer(Copyable, Layer, Module, Movable):
    """
    `Layer`-conforming wrapper around `GRU`, for use in a reflection-based
    Module struct.
    """

    var hidden_size: Int

    def forward(self, mut g: Graph, input: Symbol) -> Symbol:
        return GRU(g, input, self.hidden_size)

    def forward(mut self, input: Expr) -> Expr:
        return Expr(
            input.graph, GRU(input.graph[], input.symbol, self.hidden_size)
        )

    def __call__(mut self, input: Expr) -> Expr:
        return self.forward(input)

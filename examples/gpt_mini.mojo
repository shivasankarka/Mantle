"""GPT-mini: a char-level GPT trained on tiny-shakespeare.

Architecture: token embedding + positional embedding -> N causal
TransformerBlocks -> final LayerNorm -> output projection to vocab.
Trained with next-token-prediction (CrossEntropyLoss) and sampled
greedily during/after training.

The graph is fully comptime, so vocab size must be known at compile
time -- tiny-shakespeare has exactly 65 distinct characters.

Usage:
    pixi run mojo run -I . examples/gpt_mini.mojo
"""
from std.time import perf_counter_ns as now
from std.random import random_ui64, random_float64
from std.math import exp

import mantle.nn as nn
from mantle import Tensor, TensorShape
from mantle import Graph, Symbol, OP, f32
from mantle.core.device import Device
from mantle.autograd.attributes import AttributeVector, Attribute
from mantle.serialize.checkpoint import (
    save_checkpoint_with_optim,
    load_checkpoint_with_optim,
)
from mantle.autograd.ops.gpu_elementwise import gpu_write_from_host


comptime DATA_PATH: StaticString = "./examples/data/tinyshakespeare.txt"
comptime CHECKPOINT_PATH: StaticString = "./examples/data/gpt_mini.ckpt"
comptime VOCAB_SIZE = 65
comptime TRAIN_DEVICE = Device.gpu


def build_vocab(
    text: String, mut vocab: List[String], mut char_ids: List[Int]
) raises:
    """Fills `vocab` with sorted unique chars and `char_ids` with text
    mapped through that vocab."""
    for i in range(text.byte_length()):
        var ch = String(text[byte=i])
        var found = False
        for j in range(len(vocab)):
            if vocab[j] == ch:
                found = True
                break
        if not found:
            vocab.append(ch)

    # simple insertion sort for determinism
    for i in range(1, len(vocab)):
        var key = vocab[i]
        var j = i - 1
        while j >= 0 and vocab[j] > key:
            vocab[j + 1] = vocab[j]
            j -= 1
        vocab[j + 1] = key

    var char_to_id = Dict[String, Int]()
    for j in range(len(vocab)):
        char_to_id[vocab[j]] = j

    for i in range(text.byte_length()):
        char_ids.append(char_to_id[String(text[byte=i])])


def create_gpt_mini(
    batch_size: Int,
    seq_len: Int,
    vocab_size: Int,
    d_model: Int,
    num_heads: Int,
    d_ff: Int,
    num_blocks: Int,
) -> Graph:
    var g = Graph()
    var ids = g.input(TensorShape(batch_size, seq_len))

    var tok_emb = nn.Embedding(g, ids, vocab_size, d_model)
    var x = nn.PositionalEmbedding(g, tok_emb, seq_len)

    for _ in range(num_blocks):
        x = nn.TransformerBlock(
            g, x, num_heads, d_ff, dropout_p=0.1, causal=True
        )

    x = nn.LayerNorm(g, x, d_model)

    var logits = nn.Linear(g, x, vocab_size)  # (B, T, vocab)
    var logits_flat = g.op(
        OP.RESHAPE,
        logits,
        attributes=AttributeVector(
            Attribute("shape", TensorShape(batch_size * seq_len, vocab_size))
        ),
    )
    g.out(logits_flat)

    var y_true = g.input(TensorShape(batch_size * seq_len, vocab_size))
    var loss = nn.CrossEntropyLoss(g, logits_flat, y_true)
    g.loss(loss)

    return g^


def make_random_batch(
    char_ids: List[Int],
    batch_size: Int,
    seq_len: Int,
    vocab_size: Int,
    mut x: Tensor[f32],
    mut y_onehot: Tensor[f32],
):
    var n = len(char_ids)
    for b in range(batch_size):
        var start = Int(random_ui64(0, UInt64(n - seq_len - 1)))
        for t in range(seq_len):
            x[b * seq_len + t] = Float32(char_ids[start + t])
            var target = char_ids[start + t + 1]
            y_onehot[(b * seq_len + t) * vocab_size + target] = 1.0


def sample_from_logits(
    logits_flat: Tensor[f32], row: Int, vocab_size: Int, temperature: Float32
) -> Int:
    """Samples an index from softmax(logits / temperature). temperature=0
    falls back to argmax (greedy)."""
    if temperature <= 0.0:
        var best = 0
        var best_val = logits_flat[row * vocab_size]
        for v in range(1, vocab_size):
            var val = logits_flat[row * vocab_size + v]
            if val > best_val:
                best_val = val
                best = v
        return best

    var max_logit = logits_flat[row * vocab_size]
    for v in range(1, vocab_size):
        var val = logits_flat[row * vocab_size + v]
        if val > max_logit:
            max_logit = val

    var probs = List[Float32]()
    var total: Float32 = 0.0
    for v in range(vocab_size):
        var p = exp(
            (logits_flat[row * vocab_size + v] - max_logit) / temperature
        )
        probs.append(p)
        total += p

    var r = Float32(random_float64(0.0, 1.0)) * total
    var cumulative: Float32 = 0.0
    for v in range(vocab_size):
        cumulative += probs[v]
        if r <= cumulative:
            return v
    return vocab_size - 1


def sample[
    g: Graph
](
    mut model: nn.Model[g, device=Device.gpu],
    vocab: List[String],
    char_to_id: Dict[String, Int],
    prompt: String,
    seq_len: Int,
    batch_size: Int,
    vocab_size: Int,
    num_chars: Int,
    temperature: Float32 = 0.8,
) raises -> String:
    var context = List[Int]()
    for i in range(prompt.byte_length()):
        context.append(char_to_id[String(prompt[byte=i])])
    while len(context) < seq_len:
        context.insert(0, 0)

    var generated = String(prompt)

    # Allocated once and reused across every generated char instead of
    # re-uploading a fresh GPU tensor per token: `dummy_y` is never read
    # (inference stops before the loss node), so it needs no per-iteration
    # write at all, and `x` only needs its content refreshed in place via
    # `gpu_write_from_host` (no realloc/zero-fill, unlike `.to_gpu()`).
    var x_cpu = Tensor[f32](TensorShape(batch_size, seq_len))
    var x_gpu = x_cpu.to_gpu()
    var dummy_y = Tensor[f32](
        TensorShape(batch_size * seq_len, vocab_size)
    ).to_gpu()

    for _ in range(num_chars):
        for t in range(seq_len):
            x_cpu[t] = Float32(context[len(context) - seq_len + t])
        gpu_write_from_host(x_gpu, x_cpu)

        var out = model.inference(x_gpu.share(), dummy_y.share())
        var logits_flat = out[0].to_host()

        # last real token's logits are at row (seq_len - 1) for batch 0
        var row = seq_len - 1
        var next_id = sample_from_logits(
            logits_flat, row, vocab_size, temperature
        )

        generated += vocab[next_id]
        context.append(next_id)

    return generated^


def main() raises:
    # Keep the first GPU run intentionally small: Mojo specializes the whole
    # static graph before `main()` can print. Once this smoke configuration
    # trains and the GPU fallback audit is clean, scale these to 64/16/128.
    comptime seq_len = 16
    comptime batch_size = 4
    comptime d_model = 32
    comptime num_heads = 4
    comptime d_ff = 128
    comptime num_blocks = 1
    comptime learning_rate = 3e-4
    comptime weight_decay = 0.01
    comptime warmup_steps = 200
    comptime num_steps = 1000
    comptime sample_every = 1000

    print("Loading data from", DATA_PATH, "...")
    var text = open(String(DATA_PATH), "r").read()

    var vocab = List[String]()
    var char_ids = List[Int]()
    build_vocab(text, vocab, char_ids)

    var char_to_id = Dict[String, Int]()
    for j in range(len(vocab)):
        char_to_id[vocab[j]] = j

    print("vocab size:", len(vocab))
    print("text length:", len(char_ids))
    if len(vocab) != VOCAB_SIZE:
        print(
            "[ERROR] VOCAB_SIZE constant (",
            VOCAB_SIZE,
            ") doesn't match actual vocab size (",
            len(vocab),
            "). Update VOCAB_SIZE and rerun.",
        )
        return

    comptime graph = create_gpt_mini(
        batch_size,
        seq_len,
        VOCAB_SIZE,
        d_model,
        num_heads,
        d_ff,
        num_blocks,
    )
    var model = nn.Model[graph, device=TRAIN_DEVICE]()
    var optim = nn.optim.AdamW[graph, device=TRAIN_DEVICE](
        model.parameters, lr=learning_rate, weight_decay=weight_decay
    )
    var lr_schedule = nn.optim.WarmupCosineSchedule(
        base_lr=learning_rate,
        warmup_steps=warmup_steps,
        total_steps=num_steps,
        min_lr=learning_rate * 0.1,
    )

    print(
        "Training started. (",
        num_steps,
        "steps, batch size",
        batch_size,
        ", seq len",
        seq_len,
        ")",
    )
    var start = now()

    for step in range(num_steps):
        optim.lr = lr_schedule.get_lr(step)

        var x_host = Tensor[f32](TensorShape(batch_size, seq_len))
        var y_onehot_host = Tensor[f32](
            TensorShape(batch_size * seq_len, VOCAB_SIZE)
        )
        make_random_batch(
            char_ids, batch_size, seq_len, VOCAB_SIZE, x_host, y_onehot_host
        )

        var loss = model.forward(x_host.to_gpu(), y_onehot_host.to_gpu())

        optim.zero_grad()
        model.backward()
        optim.step()

        if step % 50 == 0 or step == num_steps - 1:
            var loss_host = loss.to_host()
            print(
                "step",
                step,
                "/",
                num_steps,
                "\tloss:",
                loss_host[0],
                "\tlr:",
                optim.lr,
            )

        if step > 0 and step % sample_every == 0:
            print(
                "--- sample @ step",
                step,
                "---\n",
                sample(
                    model,
                    vocab,
                    char_to_id,
                    "\n",
                    seq_len,
                    batch_size,
                    VOCAB_SIZE,
                    80,
                ),
                "\n---",
            )

    print("Training finished:", Float64(now() - start) / 1e9, "seconds")
    print("GPU smoke training completed.")

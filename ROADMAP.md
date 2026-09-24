# Mantle Roadmap

Goal: grow Mantle from an MLP/convnet library into one that can train
transformers (GPT/BERT-style) and ResNet-class vision models end-to-end,
then optimize. Features first, performance later (Phase 5).

Design constraint we are keeping: the graph stays fully comptime/static.
Transformers use a fixed `max_seq_len` with padding + masking; no dynamic
shapes.

## Current state (audit, 2026-07)

What exists and works:

- **Autograd**: comptime static graph (`Graph`, `Symbol`, `Node`), 35+ ops with
  forward/backward, attribute system (`AttributeVector`), scoped naming,
  graph visualization. `Graph.compile()` performs dead-node elimination; the
  runtime pre-sizes its tensor arenas from the static graph, but does not yet
  reuse activation storage.
- **Ops**: ADD/SUB/MUL/DIV (with broadcasting), EXP/LOG/POW/SQRT/NEG/ABS,
  DOT (rank-generic batched matmul), GATHER, GELU, SUM/MEAN/MAX (global or
  single-axis), TRANSPOSE (arbitrary permutation),
  RESHAPE/FLATTEN/SQUEEZE/UNSQUEEZE, CONCAT/SPLIT/SLICE,
  SIGMOID/RELU/LEAKYRELU/TANH, CLIP, FMA, CONV2D, MAXPOOL2D, DROPOUT,
  BATCHNORM2D.
- **NN**: Linear (rank-generic), Conv2d, MaxPool2d, Dropout, BatchNorm2d,
  Embedding, LayerNorm layers (+ Layer trait, Sequential, FlattenLayer);
  Softmax/LogSoftmax/GELU composites; MSELoss/CrossEntropyLoss/L1Loss; Adam,
  SGD (+momentum), `clip_grad_norm`.
- **Data/serialize**: MNIST loader, DataLoader, checkpointing, ONNX export.

Key limitations found in the audit (Phase 1 below closes the first two):

- No LR schedulers, no AdamW, no ARGMAX/accuracy metric.
- Kaiming init is commented-out stub in `mantle/nn/initializers.mojo`.
- No PAD op and no AvgPool — blocks "same"-padding convs and ResNet heads.

---

## Phase 1 — Core ops for transformers — DONE

All four landed, tested, and committed on `ops`.

### 1.1 Batched / rank-generic matmul — done

- [x] Extended `OP.DOT` in place (didn't add a separate `OP.BMM`): last two
      dims matmul, leading dims either match exactly or one operand is rank-2
      and broadcasts across the other's batch dims. Covers `(B,T,D)@(D,K)`
      and `(B,H,T,d)@(B,H,d,T)`.
- **Files touched**: `mantle/autograd/ops/basics.mojo` (`DOT` struct +
  `dot_batch_broadcast_shape` helper), `mantle/autograd/ops/matmul.mojo`
  (`batched_dot`, `batched_dot_transpose_t1/t2` — loop the existing 2D tiled
  kernel over flattened batch dims via pointer offsets),
  `mantle/autograd/ops/ops.mojo` (DOT's gradient reduction now goes through
  `accumulate_grad` with `dot_batch_broadcast_shape`, mirroring how
  ADD/SUB/MUL/DIV reduce broadcast gradients).
- [x] `Linear` reads fan-in from `shape[-1]` instead of `shape[1]` — works on
      rank-3 input unchanged.
- **Tests**: `tests/mojo/test_batched_dot.mojo` — forward + gradient check
  for both the shared-t2 and matching-batch-dims cases.

### 1.2 GATHER op + Embedding layer — done

- [x] `OP.GATHER` in `mantle/autograd/ops/mlops.mojo`: `table (V,D)` +
      `indices (any shape)` → `indices.shape + (D,)`. Backward scatter-adds
      `ug` rows into a zeroed `(V,D)` gradient at the index positions; no
      gradient to indices (they're non-trainable, so the dispatcher never
      calls that branch).
- [x] `Embedding`/`EmbeddingLayer` in `mantle/nn/layers/embedding.mojo`
      (`random_normal` init, std 0.02).
- **Tests**: `tests/mojo/test_gather_embedding.mojo` — forward correctness,
  repeated-index gradient accumulation, and an end-to-end training
  convergence check.

### 1.3 GELU — done

- [x] `OP.GELU` in `mlops.mojo`, tanh approximation, closed-form backward.
- [x] `GELU`/`GELULayer` in `mantle/nn/activations.mojo`.
- **Tests**: `tests/mojo/test_gelu.mojo` — forward against reference values,
  backward against the analytic derivative.

### 1.4 LayerNorm — done

- [x] Composite in `mantle/nn/layers/layernorm.mojo`: MEAN(axis=-1) → SUB →
      POW(2) → MEAN(axis=-1) → ADD eps → SQRT → DIV → MUL gamma → ADD beta.
      No new OP needed — pure composition of existing ops.
- [x] `LayerNormLayer` wrapper.
- **Tests**: `tests/mojo/test_layernorm.mojo` — per-row mean≈0/std≈1 check,
  and a Linear+LayerNorm training-convergence check.
- **Found and fixed a real heap-corruption bug along the way**: in
  `DIV`'s backward broadcast branch (`mantle/autograd/ops/basics.mojo`),
  the vectorized closure's SIMD-width parameter was named `netls` (typo)
  but the body used the unrelated outer `nelts` constant for
  `store`/`load` widths. Since the closure is invoked via `vectorize[1]`,
  every call wrote `nelts`-wide (not 1-wide) past its allotted slot,
  corrupting the heap — invisible until a *later*, unrelated allocation
  crashed. Only surfaced once a value fed both a second reduction and a
  broadcasting `DIV` against that reduction's output (exactly what
  `diff / std` in LayerNorm does). Also fixed a missing `return` in
  `accumulate_op` (`mantle/core/tensorutils.mojo`) that let it silently
  double-apply when `res_shape == t1_shape`. Both fixes are in the same
  commit as LayerNorm.

---

## Phase 2 — Imperative authoring API

Make the common model-authoring path PyTorch-like. The static graph remains
Mantle's execution implementation, but ordinary users should not construct,
thread, or manage a `Graph` while defining a model. `compile()`/`fit()` may be
offered later as optional convenience methods, not the primary API.

- [ ] **Stateful imperative `Module` API** — layers own their parameters and
      expose `forward(x)` (and, where appropriate, `__call__(x)`). A model
      definition can compose ordinary values/layers without `Graph` arguments
      at each operation. Retain an explicit graph-level escape hatch for
      custom ops and research use.
- [ ] **`Sequential` as a first-class container** — construct a standard
      feed-forward model from an ordered list/tuple of layers, with a single
      forward call. Make the MNIST CNN the acceptance example.
- [ ] **User-defined modules** — a lightweight way to declare reusable
      models with named child layers, so residual blocks and transformer
      blocks do not require exposing `AttributeVector` bookkeeping.
- [ ] **Model lifecycle conventions** — `model.train()`, `model.eval()`,
      `model.parameters()`, `model.to(device)`, and an inference/no-gradient
      mode. Make stateful layers such as Dropout and BatchNorm honor it.
- [ ] **Model introspection** — `model.summary(input_shape)` reporting child
      layers, input/output shapes, parameter counts, and estimated activation
      memory; clear shape/device diagnostics at the layer boundary.
- [ ] **Data pipeline ergonomics** — finish the existing `DataLoader` around
      batching, deterministic shuffle/seeding, train/validation splitting,
      transforms, and device-ready batches. Keep loading separate from model
      execution.
- [ ] **Training/evaluation helpers** — small composable helpers for one
      train step, evaluation, `predict`, accuracy/loss aggregation, timing,
      callbacks, and non-finite-value checks. Avoid making a high-level
      `fit()` loop mandatory.
- [ ] **Checkpoint ergonomics** — save/load a model's named parameters,
      optimizer state, and metadata with a round-trip test; do not require
      every example to manually wire serialization.
- [ ] **Progressive migration** — adapt MNIST and a compact MLP example to
      the new surface while preserving the current graph API and tests. Do
      not perform a library-wide rewrite until those two examples are clear.

## Phase 3 — Transformer building blocks

All composites — no new OP enum entries. Pattern: graph-builder function +
Layer-conforming struct, like `Softmax`/`SoftmaxLayer`.

- [ ] **Causal mask helper** — comptime-build a `(T, T)` constant with 0 on/
      below the diagonal and -1e9 above (use `g.constant`). Additive mask,
      applied before Softmax.
- [ ] **MultiHeadAttention** builder in `mantle/nn/layers/attention.mojo`:
      qkv = 3 Linears → RESHAPE to `(B, T, H, d)` → TRANSPOSE to
      `(B, H, T, d)` → `scores = q @ k^T / sqrt(d)` (DOT + TRANSPOSE on last
      two axes) → ADD mask → Softmax(axis=-1) → Dropout → `@ v` → TRANSPOSE
      back → RESHAPE `(B, T, D)` → output Linear. Verify Softmax composite
      handles axis=-1 on rank-4 (it uses MAX/SUM with axis, which are
      single-axis — should work; test first).
- [ ] **FeedForward** builder: Linear(4·D) → GELU → Linear(D) → Dropout.
- [ ] **TransformerBlock** builder: pre-norm residual wiring —
      `x + MHA(LN(x))` then `x + FF(LN(x))`. Residuals are just `OP.ADD`.
- [ ] **Learned positional embeddings** — `(max_seq_len, d_model)` param,
      broadcast-added to token embeddings (or GATHER with arange ids).
- [ ] **GPT-mini example** in `examples/`: char-level tiny dataset, vocab
      ~65, 2 blocks, d_model 64, 4 heads, seq len 64. Train with
      CrossEntropyLoss on next-token prediction + greedy sampling loop using
      `model.inference()`. This is the acceptance test for Phases 1–2.
- **Watch out**: CrossEntropyLoss currently expects one-hot `(B, C)` and
  softmaxes axis=1 — for LM training, RESHAPE logits to `(B·T, vocab)` and
  one-hot the targets in the data pipeline.

---

## Phase 4 — Training infrastructure

- [ ] **AdamW** in `mantle/nn/optim.mojo` — copy Adam, apply decoupled
      weight decay (`p -= lr·wd·p` before the Adam update). Optionally skip
      decay for rank-1 params (biases, norms) via a shape check.
- [ ] **LR schedulers** — small structs holding step count with a
      `get_lr() -> Scalar[f32]` and optimizers gaining a settable `lr`:
      warmup+cosine (transformers), step decay (vision).
- [ ] **ARGMAX op + accuracy metric** — `OP.ARGMAX` forward-only (backward
      returns zeros); `accuracy(logits, targets)` helper in a new
      `mantle/nn/metrics.mojo`.
- [ ] **MIN op** — mirror of MAX in `basics.mojo`; trivial while in there.
- [ ] **BCELoss** — composite: `-mean(y·log(p) + (1-y)·log(1-p))` with CLIP
      on p for stability. All ops exist.
- [ ] **Label smoothing** — optional param on CrossEntropyLoss; smooth the
      one-hot inside the builder: `y·(1-ε) + ε/C`.

---

## Phase 5 — Vision (ResNet-class)

- [ ] **PAD op** — `OP.PAD`, constant mode first (attributes: per-axis
      before/after). Backward = SLICE of `ug` (op exists). Reflect/replicate
      later. Unblocks "same" padding.
- [ ] **AvgPool2d** — mirror MAXPOOL2D in `mantle/autograd/ops/pool.mojo`;
      backward distributes `ug/(kh·kw)` uniformly (simpler than max's argmax
      tracking). Add `AvgPool2d`/`AvgPool2dLayer`; a GlobalAvgPool is the
      same layer with kernel = spatial dims.
- [ ] **Kaiming/He init** — uncomment and finish
      `mantle/nn/initializers.mojo` (`calculate_fan` already works); gains
      for relu (√2) and leaky_relu; wire `"kaiming_uniform"`/`"kaiming_normal"`
      into `initialize_tensor`; make Conv2d default to kaiming_uniform.
- [ ] **ResidualBlock builder** — Conv→BN→ReLU→Conv→BN + skip ADD (1×1 conv
      downsample when shapes differ) in `mantle/nn/layers/residual.mojo`.
- [ ] **CIFAR-10 loader** in `mantle/data/datasets.mojo` (binary format is
      trivial: 1 label byte + 3072 pixel bytes per record).
- [ ] **ResNet-mini example** in `examples/`: ResNet-8-ish on CIFAR-10 with
      SGD+momentum, step decay, accuracy metric. Acceptance test for Phase 4.
- [ ] **Data augmentation** — random horizontal flip, random crop with pad,
      normalize; as `Tensor -> Tensor` transforms applied in the DataLoader.

---

## Phase 6 — GPU transformer training

Acceptance criterion: GPT-mini trains for 100 steps on GPU with decreasing
loss and no host round-trips inside the forward/backward/optimizer loop.

- [ ] **Batched GPU matmul** — native strided `(B,H,M,K) @ (B,H,K,N)` and
      transpose variants for attention. This is the primary GPT throughput
      blocker; do not submit one host-side matmul launch per attention head.
- [ ] **Native GPU transformer ops** — GATHER and scatter-add, arbitrary-axis
      reductions, broadcast elementwise ops, arbitrary transpose, SQRT, GELU,
      Softmax/LogSoftmax, and deterministic Dropout, each with backward.
      Remove the current CPU round-trip fallback from the GPT hot path.
- [ ] **Fused transformer kernels** — LayerNorm forward/backward,
      Softmax/LogSoftmax + CrossEntropy, and common Linear epilogues
      (bias/activation/residual) to cut intermediate buffers and kernel
      launches.
- [ ] **GPU optimizer completion** — validate GPU AdamW against CPU AdamW,
      add fused multi-tensor updates, and later mixed precision with gradient
      scaling.
- [ ] **GPU model ergonomics** — reusable device batch buffers, asynchronous
      input upload, GPU checkpoint save/load, eval mode, deterministic seeds,
      and a small Trainer API.
- [ ] **GPU correctness/perf suite** — CPU/GPU forward and gradient parity
      tests against Torch; shape benchmarks for MLP, attention, LayerNorm,
      and GPT-mini; regression limits for host-device copies and allocations.

## Phase 7 — Graph runtime and performance

Profile the GPT-mini and ResNet-mini examples first; optimize what's hot.

- [ ] **Graph.compile()** — in order of expected payoff:
      1. [x] Dead-node elimination (nodes not reachable from outputs/loss).
      2. [x] Pre-size runtime symbol arenas from static graph counts.
      3. [ ] Activation liveness planner: reuse intermediate buffers whose
         forward/backward live ranges do not overlap. This must describe each
         op's saved tensors before aliasing any storage.
      4. [ ] Block-level activation checkpointing: retain transformer block
         inputs and recompute block internals in backward, trading compute for
         a large reduction in persistent activation memory.
      5. [ ] Op fusion: start with elementwise chains (e.g. ADD+RELU) via a
         fused-kernel dispatch path. LayerNorm is already a fused op.
- [ ] **Parallelize batched matmul** over batch·heads (`parallelize` over
      slices; kernel per slice unchanged).
- [ ] **BLAS coverage audit** — confirm `dot_transpose_t1/t2` paths are hit
      in backward; route batched matmul through `mantle/core/blas.mojo`.
- [ ] **Fused OP.LAYERNORM / OP.GELU backward** if profiling justifies it.

---

## Phase 8 — Quality / stretch

- [ ] Replace `from mantle.core.tensorutils import *` in
      `autograd/ops/basics.mojo` with explicit imports.
- [ ] Wire `examples/data/` into example scripts (auto-download or clear
      instructions).
- [ ] FashionMNIST loader (same format as MNIST — near-free).
- [ ] Checkpoint save/load round-trip test for a transformer model.
- [ ] **Dynamic shapes exploration** (explicitly deferred): a runtime-shape
      graph path would be a major rework of `Graph`/`Model`; revisit only
      after Phases 1–5 land.

---

## Done

- [x] Dropout, BatchNorm2d, SGD (+momentum)
- [x] Gradient clipping (`clip_grad_norm`)
- [x] Softmax / SoftmaxLayer / LogSoftmax
- [x] NEG, ABS, SQRT ops
- [x] L1Loss
- [x] Phase 1: batched DOT, GATHER + Embedding, GELU, LayerNorm (see above)

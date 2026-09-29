# Mantle changelog

This is a summary of notable changes to Mantle. There are no tagged releases
yet, so everything below is grouped under **Unreleased** and ordered roughly
newest-first within each section. Commit hashes are short refs into `main`.

## Unreleased

### ⭐️ Added

- GPU backend for training: elementwise ops, Adam/AdamW, matmul, convolution,
  attention, and native math kernels, alongside the existing CPU path
  (`99b2bf1`, `e8afcfa`, `0df5487`, `432812c`).
- Reflection-based and `Sequential`-style model definition APIs, so a model
  can be built either as a plain struct with reflected fields or by chaining
  layers (`3ef267c`, custom `Module` API in `dcb002e`).
- Transformer building blocks: multi-head attention with optional causal
  masking, LayerNorm, GELU, embeddings/gather, softmax, and a `gpt_mini`
  example (`bb884c4`, `3028daa`, `afa446b`, `13b7de2`, `db52ea2`).
- `DataLoader` with batching, reproducible shuffling, and iteration
  improvements (`2effc70`, `ed49def`, `2027578`).
- BatchNorm2d, SGD with gradient clipping, and common module constructors
  (`766cf65`, `c3bc15e`, `468ffc6`, `6834d72`).
- GPU vs. CPU and matched GPU benchmark scripts for tracking performance
  over time (`aa4cfe8`, `db38381`).

### 🦋 Changed

- Ported the codebase to Mojo v1.1 and MAX, fixing the resulting API breaks
  (`0214309`, `f388c07`, `39132cc`).
- Reworked `Tensor` to carry an explicit `Device` (CPU/GPU) parameter through
  the pipeline instead of assuming CPU everywhere (`6fbd1a1`, `36b4d0c`,
  `73794cc`).
- Switched CPU matmul to MAX's `linalg` and fused matmul+bias into a single
  linear op (`7b5246f`, `d347dd8`).
- Cached compiled GPU kernels and dropped redundant device syncs so repeated
  calls avoid recompilation/sync overhead (`1511791`).
- Split Pixi environments so the default install only pulls in Mojo/MAX, with
  PyTorch, ONNX, and visualization extras opt-in (`e730efd`).
- Fused Conv2d's parameter-gradient backward into direct GPU writes and
  removed redundant gradient clears, avoiding a separate accumulate pass for
  single-consumer gradients (`5d7ea20`, `0797aba`, `4d59c81`, `1db305d`,
  `db645cd`).
- Adam/AdamW gradient resolution moved to compile time and CPU chunking was
  rebalanced to fix an imbalance in the parallel split (`432812c`).

### 🛠️ Fixed

- Fixed a heap corruption bug in LayerNorm's div backward (`3028daa`).
- Fixed a trait downcast warning and an unsafe-pointer-access issue
  surfaced by the Mojo v1.1 port (`582add2`, `5cf6ec0`).
- Fixed CPU chunking imbalance and the bias-path overwrite-gradient bug
  uncovered while adding AdamW GPU support (`432812c`).

## Older history

The project predates this changelog; for the full commit-by-commit history
(including the original [Basalt](https://github.com/basalt-org/basalt)
lineage this repository continues), see `git log`.

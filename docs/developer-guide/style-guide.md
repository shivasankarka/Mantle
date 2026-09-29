# Style guide

## Formatting

Run `pixi run format` before committing. It calls `mojo format` over
`mantle/`.

## Docstrings

Keep docstrings short and focused on what the function does:

1. A 1-3 sentence description of the function's behavior.
2. An `Args:` section for non-trivial parameters, one line each.
3. A `Notes:` section for anything else worth keeping — invariants,
   performance tradeoffs, gotchas — kept to the essential point rather than
   a full narrative.

```mojo
fn gpu_conv2d_kernel_backward[...](...):
    """Computes the kernel-weight gradient for a Conv2d backward pass.

    Args:
        upper_grad: Gradient flowing in from the next layer, NCHW layout.

    Notes:
        Splits the batch axis into chunks so each (weight, chunk) pair gets
        its own GPU thread, then reduces chunks in a second pass. Shape
        dimensions are comptime parameters: GPU integer division by a
        runtime value is markedly slower than by a compile-time constant.
    """
```

Do **not** put in a docstring:

- Benchmark numbers or specific timings — these belong in commit messages
  or PR descriptions, where they can be tied to the hardware and Mojo
  version they were measured on.
- The history of how the code got to its current form ("originally we did
  X, then switched to Y because..."). Describe the current design only;
  history lives in `git log`.

## Comments

Default to no comments. Add one only when the *why* is non-obvious: a
hidden constraint, a subtle invariant, or a workaround for a specific
compiler/runtime quirk. If removing the comment wouldn't confuse a future
reader, don't write it.

## GPU kernels

- Prefer comptime (parameter) shape dimensions over runtime arguments in
  hot inner loops — see `mantle/autograd/ops/gpu_conv.mojo` for the
  pattern of caching one compiled kernel per distinct shape via `_Global`.
- New kernels should have a matching CPU-reference correctness test (see
  `tests/mojo/test_gpu_cnn.mojo` for an example that checks a GPU kernel
  against a CPU implementation of the same op).

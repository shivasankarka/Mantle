# Quick start

Mantle is built and run through [Pixi](https://pixi.sh). The default
environment installs only Mojo and MAX, so a plain `pixi run` does not pull
in any Python ML packages.

## Run an example

```bash
pixi run mojo -I . examples/mnist.mojo
```

Other bundled examples:

```bash
pixi run mojo -I . examples/housing.mojo
pixi run mojo -I . examples/sin_estimate.mojo
```

Each example also has alternate model-definition styles that produce
equivalent training results while demonstrating different ways to build a
model:

```bash
pixi run mojo -I . examples/housing_module.mojo       # Module (PyTorch-like) API
pixi run mojo -I . examples/housing_sequential.mojo    # Sequential API
```

The same `_module`/`_sequential` variants exist for `sin_estimate` and
`mnist`.

## Optional environments

Extra Pixi environments install only the dependencies each one needs:

```bash
# PyTorch comparison scripts, plus pandas/matplotlib and ONNX export
pixi run -e examples python examples/mnist.py

# Mojo and Python-backed test suite
pixi run -e test test

# ONNX graph rendering with Netron
pixi run -e visualize python mantle/serialize/graph_render.py
```

## Running tests

```bash
pixi run -e test test
```

To run a single Mojo test file directly (useful while iterating):

```bash
pixi run mojo run -I . tests/mojo/test_gpu_cnn.mojo
```

## Next steps

- [Architecture overview](../developer-guide/architecture.md) — how Mantle's
  autograd, tensor, and op dispatch fit together.
- [Roadmap](../../ROADMAP.md) — what's built and what's planned.
- [Contributing](../developer-guide/contributing.md) — how to open a PR.

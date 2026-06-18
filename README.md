<br/>
<p align="center">
  <a href="https://github.com/shivasankarka/mantle#">
    <img src="./assets/mantle.png" alt="Logo" width="300" height="300">
  </a>

  <h1 align="center">Mantle</h1>

  <p align="center">
    A Mojo🔥-native machine learning framework built from the ground up for performance and flexibility.
  </p>
</p>

<div align="center">
  <a href="https://github.com/shivasankarka/mantle/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/shivasankarka/mantle?color=dark-green" />
  </a>

  <a href="https://github.com/shivasankarka/mantle/issues">
    <img src="https://img.shields.io/github/issues/shivasankarka/mantle?color=dark-green" />
  </a>

  <a href="https://github.com/shivasankarka/mantle/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/shivasankarka/mantle?color=dark-green" />
  </a>
</div>

---

> [!NOTE]
> This repository is an actively maintained continuation of the original [Basalt](https://github.com/basalt-org/basalt/tree/main) project.
>
> The original Basalt repository introduced one of the earliest machine learning frameworks built entirely in Mojo. This repository preserves that work and its history while updating the codebase for latets Mojo releases, improving the framework architecture, and introducing new features and functionality.
>
> We would like to thank the original authors and contributors whose work made this project possible.

## About The Project

Mantle is a machine learning framework built entirely in Mojo.

Designed specifically for the Mojo ecosystem, Mantle provides the core building blocks needed to develop, train, and deploy machine learning models while taking advantage of Mojo's performance-oriented design. Rather than wrapping existing frameworks, Mantle is implemented natively in Mojo from the ground up, giving developers direct access to the language's strengths in performance, compile-time specialization, and low-level control.

Originally derived from the Basalt project, Mantle continues development with support for modern Mojo releases, improved APIs, expanded operator coverage, and a stronger foundation for future machine learning workloads.

Current areas of focus include:

* Native Mojo-first machine learning APIs
* High-performance tensor operations
* Neural network layers and training utilities
* Improved model-building ergonomics
* Expanded operator and activation support
* Long-term maintainability and active development

As Mojo continues to evolve, Mantle aims to become a comprehensive machine learning toolkit for the ecosystem-providing a familiar developer experience while exploring new opportunities enabled by a language designed specifically for AI and high-performance computing.

### Benchmarks

Mantle is capable of achieving performance comparable to established frameworks such as PyTorch on a number of workloads, and there is still significant room for further optimization as Mojo continues to evolve.

![mantle\_benchmark](./assets/benchmark.png)

## Quick Start

Run the example models:

```bash
mojo -I . examples/housing.mojo
```

```bash
mojo -I . examples/sin_estimate.mojo
```

```bash
mojo -I . examples/mnist.mojo
```

Compare against the equivalent PyTorch implementations.

Install the dependencies and run:

```bash
python examples/housing.py
python examples/sin_estimate.py
python examples/mnist.py
```

Each example also includes alternate model-definition styles inspired by PyTorch and scikit-learn:

```bash
mojo -I . examples/housing_module.mojo
mojo -I . examples/housing_sequential.mojo
```

Likewise:

* `sin_estimate_module.mojo`
* `sin_estimate_sequential.mojo`
* `mnist_module.mojo`
* `mnist_sequential.mojo`

These variants produce equivalent training results while demonstrating different approaches to model construction.

## Roadmap

See `ROADMAP.md` for the current detailed roadmap.

### Current Focus

* [x] Reflection-based model building
* [x] Sequential model construction API
* [x] Custom Tensor and TensorShape implementations
* [x] Kernel and operator performance improvements
* [x] Profiling and benchmarking infrastructure

### In Progress

* [ ] High-level `fit()` training API
* [ ] Additional tensor operators
* [ ] Expanded layer library
* [ ] Additional activation functions
* [ ] Graph submodules and composition
* [ ] Computer vision benchmarks

### Long-Term Goals

* [ ] Better parallelization support
* [ ] GPU acceleration
* [ ] Reworked dataloading pipeline
* [ ] Autotuning
* [ ] Graph compilation optimizations
* [ ] Operator fusion
* [ ] ONNX interoperability
* [ ] MAX ecosystem compatibility

## Contributing

Mantle is a community-driven project and contributions of all sizes are welcome.

If you discover a bug, have an idea for a feature, or would like to contribute code, please open an issue or discussion first for larger changes.

Before opening a new issue:

* Check whether the issue has already been reported.
* Provide steps to reproduce bugs whenever possible.
* Include sufficient context for feature requests.

### Creating A Pull Request

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push your branch
5. Open a pull request

Before submitting:

* Ensure existing tests pass.
* Add tests for significant new functionality.
* Provide a clear explanation of the changes.
* Link any relevant issues or discussions.
* Include any special testing instructions if applicable.

Example test command:

```bash
mojo run -I . test/test_ops.mojo
```

## Origins

This project originated from the original [Basalt](https://github.com/basalt-org/basalt/tree/main) repository and preserves its commit history.

The original project demonstrated the potential of machine learning frameworks written entirely in Mojo and helped establish many of the ideas that continue to guide development today.

Since the original repository became inactive, development has continued here with support for newer Mojo versions, architectural improvements, bug fixes, expanded functionality, and ongoing maintenance.

We are grateful to all original Basalt contributors for their work and contributions.

## License

Distributed under the Apache 2.0 License with LLVM Exceptions.

See:

* `LICENSE`
* LLVM License: https://llvm.org/LICENSE.txt

for additional details.

## Acknowledgements

* Built with Mojo by Modular
* Thanks to the original Basalt authors and contributors
* Thanks to everyone who continues to contribute to the project

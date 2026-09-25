# Training benchmark suite

This suite regenerates the Housing / sine / MNIST small–medium–large chart
with matched Mantle and PyTorch CPU workloads. Every sample is 100 training
steps after five warm-up steps; a step includes forward, MSE backward, and
Adam. Inputs are intentionally synthetic and resident in memory, so dataset
loading and language-specific data-loader overhead do not affect the result.

Run current Mantle and PyTorch from the project root:

```bash
pixi run mojo -I ./ benchmarks/training_workloads.mojo | tee /tmp/mantle-current.csv
pixi run -e test python benchmarks/training_workloads_torch.py | tee /tmp/pytorch-current.csv
```

For the matched Apple Metal/MPS sweep, inputs and targets stay on the device;
only the model's forward pass, MSE backward pass, and Adam update are timed:

```bash
pixi run mojo -I ./ benchmarks/training_workloads_gpu.mojo | tee /tmp/mantle-gpu-current.csv
PYTHONUNBUFFERED=1 pixi run -e test python benchmarks/training_workloads_gpu_torch.py | tee /tmp/pytorch-mps-current.csv
```

To diagnose the GPU MNIST-medium path, this profiler synchronizes after each
phase and reports the total for 100 steps. It is intentionally not a throughput
measurement:

```bash
pixi run mojo -I ./ benchmarks/profile_mnist_gpu.mojo
```

The isolated pre-optimization checkout is at
`/Users/shivasankar/.codex/worktrees/mantle-pre-optimizations/basalt-main`.
Run the same source against its historical Mantle implementation:

```bash
cd /Users/shivasankar/.codex/worktrees/mantle-pre-optimizations/basalt-main
pixi run mojo -I ./ /Users/shivasankar/Documents/Codes/basalt-main/benchmarks/training_workloads.mojo | tee /tmp/mantle-baseline.csv
```

Each runner prints machine-readable lines in this form:

```text
RESULT,mnist-large,3,1.23456789
```

Keep the five samples per label. The chart will use their mean and standard
deviation, matching the error bars in the original image.

After updating `results.json`, regenerate the tracked image with:

```bash
pixi run -e test python benchmarks/plot_training_workloads.py
```

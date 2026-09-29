# Basalt Architecture & Data Flow

## Overview

Basalt is a compile-time autograd library. The key design principle: **the computation graph is fully known at compile time**. The compiler specializes `Model[g]` for each unique graph, unrolling the entire forward and backward pass into inlined operations with no runtime dispatch overhead.

---

## Comptime vs Runtime Split

This is the most important thing to understand about the codebase:

| Thing | When | Where |
|-------|------|-------|
| Graph structure (nodes, shapes, operators) | Comptime | `Graph` parameter `g` of `Model[g]` |
| `Symbol` values (IDs, shapes, trainable flags) | Comptime | Accessed via `Self.g.nodes[i].inputs[j]` |
| `OP`, `AttributeVector` per node | Comptime | Extracted with `comptime op = Self.g.nodes[i].operator` |
| Tensor data (weights, activations, gradients) | Runtime | Stored in `Collection` inside `Parameters` |
| `Collection` lookups by `Symbol.name` (UInt32) | Runtime | `self.parameters.tensors[symbol]` |

**Critical rule**: `materialize[Self.g]()` creates a runtime copy of the comptime graph. This works for the `List[Symbol]` fields (`inputs`, `outputs`) because `Symbol` is `TrivialRegisterPassable`. It **crashes** if the resulting graph is freed, because its internal List allocations were made by the comptime interpreter (not tcmalloc) and cannot be freed at runtime. Therefore `materialize[Self.g]()` must only be used for short-lived local variables that don't get destructed through tcmalloc — or avoided entirely by using `comptime for` + `Self.g` directly.

> **2026-06 update**: the codebase now compiles cleanly and the full `tests/mojo/` suite passes on Mojo nightly `1.0.0b3.dev2026061606`. Most of what's described below as "Open Issues" turned out to be real, fixable bugs (a SIMD-width inference bug in `tensorutils.reduce()`, stdlib `exp`/`log`/`sqrt`/`max` signature drift) rather than fundamental architecture problems. See `ROADMAP.md` for what's been built since (the `Collection` rewrite, the reflection-based `Module`/`Sequential` builders) and what's still planned.

---

## Phase 1: Graph Construction (pure comptime, user code)

```
user calls linear_regression() → returns Graph
comptime graph = linear_regression(32, 13, 1)
```

**`Graph`** is built by the user calling methods on it:
- `g.input(shape)` → creates a `Symbol`, appends to `g.inputs`, returns `Symbol`
- `g.param(shape, init=Param(...))` → creates a `Symbol`, stores in `g.params` (a `ParamDict`), returns `Symbol`
- `g.op(OP.DOT, sym1, sym2)` → infers output shape, creates output `Symbol`, appends a `Node` to `g.nodes`, returns output `Symbol`
- `g.out(sym)` → appends to `g.outputs` (marks inference output)
- `g.loss(sym)` → sets `g.loss_out` (marks loss output)

**`Symbol`**: A thin, trivially copyable handle (20 bytes). Fields:
- `name: UInt32` — unique integer ID, auto-incremented by `graph.symbol_count`
- `dtype: DType` — always `DType.float32` globally
- `shape: TensorShape` — statically known tensor shape
- `trainable: Bool` — whether this symbol produces a gradient

**`Node`**: Represents one operation. Fields:
- `operator: OP` — which operation (ADD, DOT, RELU, etc.)
- `inputs: List[Symbol]` — input symbols
- `outputs: List[Symbol]` — output symbols (usually just one)
- `attributes: AttributeVector` — extra config (axis, padding, stride, etc.)

**`Param`**: Stores initialization spec for a parameter tensor. Fields:
- `initializer: Optional[Attribute]` — e.g. `Attribute("initializer", "random_uniform")`
- `data: Optional[List[Scalar[dtype]]]` — init args (e.g. `[-bound, bound]`) or literal values

**`ParamDict`**: Parallel arrays `symbols: List[Symbol]` + `values: List[Param]`. Indexed by position, not by symbol ID. Symbols here are the weight/bias graph parameters (not inputs or node outputs).

**`AttributeVector`**: Fixed-size array of up to 10 `Attribute` entries. Each `Attribute` stores name (16-byte `Bytes`), data (32-byte `Bytes`), and a type tag. Completely stack-allocated and `TrivialRegisterPassable`.

After construction, the graph is passed as a comptime parameter:
```mojo
comptime graph = linear_regression(32, 13, 1)
var model = nn.Model[graph]()
```

---

## Phase 2: Model Initialization (runtime)

`Model[g]()` construction:

### `allocate_tensor_memory()`

Uses `comptime for` over `Self.g` to build `parameters.tensors: Collection` — a flat array of all `Tensor[dtype]` values keyed by `Symbol.name`.

Order of allocation (matters for Collection lookup):
1. **Inputs**: for each `Symbol` in `Self.g.inputs` → allocate zero tensor of `sym.shape`
2. **Params**: for each symbol in `Self.g.params` → run initializer or zero-init tensor
3. **Node outputs**: for each node, for each output symbol → allocate zero tensor of output shape

For params with an initializer (e.g. `random_uniform`): calls `initialize_tensor(shape, type, data)` which calls `rand_uniform` or `rand_normal` to fill.

### `allocate_grad_memory()`

Same structure but only allocates for trainable symbols:
1. Trainable inputs (rare)
2. Trainable params (weights/biases)
3. Trainable node outputs (any intermediate that is `trainable=True`)

**`Collection`** (rewritten 2026-06): tensors are stored densely in insertion order (`data_ref: UnsafePointer[Tensor[dtype]]`), with a separate `index_map: UnsafePointer[Int]` array sized by the largest symbol id seen so far, mapping `Symbol.name -> dense slot` for **O(1)** lookup — no scanning. This keeps memory proportional to however many tensors were actually inserted (important for sparse subsets like Adam's per-trainable-parameter `rms_grads`/`momentum_grads`, which only cover a fraction of the graph's total symbol id space) while still supporting direct id-based lookup. `__getitem__(symbol)` returns a `Tensor[dtype]` via `.share()` (refcounted alias of the same buffer, not a deep copy) rather than a `ref` — this is safe because `Tensor` itself is refcounted (see `basalt/nn/tensor.mojo`), so writes through the shared alias land in the same storage.

---

## Phase 3: Forward Pass

```mojo
var loss = model.forward(x, y_true)
```

`forward()` calls `execute[len(Self.g.nodes)](t_inputs)`.

### `execute[num_nodes]()`:

1. **Write inputs** (comptime loop, each `sym` is comptime):
   ```mojo
   comptime for i in range(len(Self.g.inputs)):
       comptime sym = Self.g.inputs[i]
       self.parameters.tensors[sym] = t_input[i].copy()
   ```

2. **Unrolled node loop** (`comptime for i in range(num_nodes)`):
   - `comptime op = Self.g.nodes[i].operator` — which OP
   - `comptime attrs = Self.g.nodes[i].attributes` — comptime AttributeVector
   - Branch on `op.dynamic` (CONCAT/SPLIT) vs static
   - For static: extract comptime `t1`, `t2`, `t3`, `out` Symbols; call typed `forward_op[op, t1.shape, ...]`

Each `forward_op[op, shapes, attrs](mut res, t1, ...)` is a **fully specialized function** — the operator, shapes, and attributes are all comptime params, so the compiler inlines the exact math kernel.

**How `parameters.tensors[sym]` works**: `Collection.__getitem__(sym)` looks up `sym.name` directly in the `index_map` array (O(1)), then returns a `.share()`'d alias of the `Tensor` at that dense slot. The `Symbol` values extracted at comptime (`t1`, `out`, etc.) are `TrivialRegisterPassable` so they're embedded in the generated code as constants.

`forward()` returns `ref[origin_of(self)] Tensor[dtype]` pointing directly into `parameters.tensors[loss_out]` — no copy of the loss tensor.

---

## Phase 4: Backward Pass

```mojo
model.backward()
```

1. **Seed gradient**: `fill(parameters.grads[loss_out], 1.0)` — sets the output gradient to 1.

2. **Reverse unrolled loop** (`comptime for i in range(len(Self.g.nodes))`):
   - `comptime reverse_i = len(Self.g.nodes) - i - 1`
   - Extract comptime `op`, `attrs`, `out`, `t1`, `t2`, `t3`
   - For each trainable input, call `backward_op[tensor_id, op, out.shape, t1.shape, ...]`
   - `backward_op` returns `res_grad: Tensor`, then `accumulate_grad(grad, res_grad)` adds it

**Gradient accumulation**: `accumulate_grad[grad_shape, res_grad_shape](grad, res_grad)` handles broadcasting. For same-shape: element-wise add. For broadcast: sums across expanded dims.

**Which grads are kept**: Only symbols where `sym.trainable == True` have gradient tensors in `parameters.grads`. Input symbols are not trainable by default. Param symbols (weights/biases) are trainable by default. Intermediate node outputs are trainable if any input was trainable (`result_trainable` logic).

---

## Phase 5: Optimizer Step

```mojo
optim.zero_grad()   # set all grads to zero
model.backward()    # compute grads
optim.step()        # update params
```

**`Adam[g, trainable_parameters]`**:
- `trainable_parameters: List[Symbol]` — comptime list of trainable param symbols (computed from `g.params`)
- `parameters: Pointer[Parameters, MutAnyOrigin]` — pointer to model's Parameters (not owned)
- `rms_grads: Collection` — second moment estimates
- `momentum_grads: Collection` — first moment estimates

**`step()`**:
- `materialize[Self.trainable_parameters]()` → runtime `List[Symbol]`
- `parallelize[p_step](len(tr))` — one thread per param
- Inside: vectorized SIMD Adam update per param element

The Adam update uses `vectorize[1](n, v_step)` — width 1, meaning scalar fallback. This could be width `nelts` for speedup but currently plays it safe.

---

## Key Invariants

1. **Symbol name = Collection index key**: The `UInt32` name is the only thing used at runtime to look up tensors. Shape/dtype/trainable are only used at comptime.

2. **Graph is comptime-only**: Never use `graph` (a comptime value) for runtime operations like printing or passing to Python. `materialize[Self.g]()` works for short-lived reads but the result must not be destructed through tcmalloc (causes free-invalid-pointer crash).

3. **`comptime for` + `Self.g.nodes[i].inputs[j]`**: Works for extracting `Symbol` values (trivial). Fails if the compiler tries to materialize intermediate `List[Node]` or `List[Symbol]` for runtime use — use comptime extraction only.

4. **`TensorShape`** is stack-only (`TrivialRegisterPassable`) — safe to use as comptime param. Max rank 8, stored as `IndexList[8]`.

5. **`AttributeVector`** is stack-only (`TrivialRegisterPassable`) — safe as comptime param. Max 10 attrs, each with fixed 16-byte name + 32-byte data.

6. **`Collection` double-free risk**: `Collection.__init__(out self, *, deinit take: Self)` shallow-moves pointers (including the `index_map`). Only one instance should own the data. `Collection.__init__(out self, *, copy: Self)` deep-copies (allocates new memory for `data_ref`, `symbols_ref`, and `index_map_ref`). Be careful when copying `Parameters`.

---

## Resolved Issues (were "Open Issues" as of the original port; fixed 2026-06)

- **`model.backward()` dynamic op path**: works as written — `test_dynamic_ops.mojo` passes. The dynamic path (CONCAT/SPLIT) is exercised by that test even though housing/mnist/sin_estimate don't use it.

- **`allocate_tensor_memory` param init**: works as written — params get their configured `random_uniform`/`random_normal` init, verified by training losses matching across the original and reflection-based example variants bit-for-bit at the same checkpoints.

- **`Adam.step()` materialize**: works as written.

- **The actual root causes that were blocking things**: not the comptime/materialize machinery described above, but (a) a SIMD-width parameter-inference bug in `basalt/utils/tensorutils.mojo`'s `reduce()` (calling `op(m[0], ...)` with scalar `SIMD[dtype,1]` args inside a `comptime if _nelts == 1` branch couldn't infer the higher-order function's `simd_width` parameter — fixed by explicitly binding `op[dtype, 1](...)`), and (b) `std.math`'s `exp`/`log`/`sqrt`/`max` signatures changed (now `width: SIMDSize, //` positional-only, not the `thin def[dtype, nelts](...)` shape basalt's `elwise_transform`/`reduce` expect as higher-order parameters) — fixed with thin wrappers in `basalt/utils/math_util.mojo`. The original `Collection` was also mid-rewrite (now replaced, see above) but was not itself the source of the crashes encountered.

---

## Model-Building Styles (added 2026-06, on `graph_building` branch)

Three coexisting ways to build a `Graph` now exist — none removed the others:

1. **Free-function style** (original): write a function `(mut g: Graph, ...) -> Symbol` per layer/model, thread `Symbol`s through by hand. See `examples/sin_estimate.mojo`, `examples/housing.mojo`, `examples/mnist.mojo`.

2. **Reflection-based `build_graph`** (`basalt/nn/module.mojo`): define a plain struct whose fields are `Layer`-conforming layer structs (e.g. `var fc1: LinearLayer`, `var act1: ReLULayer`), then call `nn.build_graph(model_def, g, x)`. It reflects over the struct's fields in declaration order and chains each `Layer`-conforming field's `forward(g, x) -> x`. Non-`Layer` fields are silently skipped. See `examples/*_module.mojo`.

3. **`Sequential`** (`basalt/nn/module.mojo`): `nn.Sequential(LinearLayer(32), ReLULayer(), LinearLayer(10))` — a variadic-generic struct (`Sequential[*Ts: Layer & Movable]`) storing layers in a `Tuple[*Ts]`, no struct field declarations needed. `Sequential` itself conforms to `Layer`, so it nests inside style 2's structs too. See `examples/*_sequential.mojo`.

**The `Layer` trait**: `def forward(self, mut g: Graph, input: Symbol) -> Symbol`. All "layer" structs (`LinearLayer`, `ReLULayer`, `Conv2dLayer`, `MaxPool2dLayer`, `FlattenLayer`, `Sequential`) wrap the existing free-function builders (`Linear`, `ReLU`, `Conv2d`, `MaxPool2d`) and just call them from `forward`. Loss functions (`MSELoss`, `CrossEntropyLoss`) are **not** `Layer`s — they take two graph inputs (`y_pred`, `y_true`), not one, so they're still called manually after the chain, same in all three styles.

**Reflection mechanics that make style 2 work** — verified against Mojo nightly `1.0.0b3.dev2026061606`, not assumed:
- `from std.reflection import reflect`; `reflect[T]` (no parens — comptime alias, not a function call) gives a `Reflected[T]` handle with `.field_count()`, `.field_names()`, `.field_types()`, `.field_type[<StringLiteral>]`, `.field_ref[idx](instance)`.
- The accessor that lets you recover a per-field *concrete* type generically (needed to satisfy a trait bound) is **not** `rebind` on `field_types()[idx]` (gives an `AnyType`-erased value, no `.T`) and **not** `field_type[name]` fed a runtime/loop-derived name (`field_names()[idx]` is a `StringSlice`, won't convert to the required `StringLiteral`even though both are comptime). The working pattern is `conforms_to(field_type, Trait)` as a comptime filter + `trait_downcast[Trait](field_val)` to get the trait-bound call — this is what made `build_graph` possible.
- `r.field_ref[idx](instance)` returns a true mutable reference into the original struct; mutations through a trait method on it persist back to the source field (verified).

**Variadic-generics mechanics that make style 3 work**:
- `struct Sequential[*Ts: Layer & Movable]` — note the explicit `& Movable`: `Tuple[*element_types: Movable]` requires it, `Layer` alone doesn't imply it.
- Constructor: `def __init__(out self, var *layers: *Self.Ts): self.layers = Tuple(*layers^)` — the variadic argument arrives as a `VariadicPack`, not directly a `Tuple`; it must be re-spread with `*layers^` into `Tuple(...)`'s own variadic constructor, not passed as a single argument.
- Loop bound: `Self.Ts.__len__()` is comptime-valid for `comptime for`; `self.layers.__len__()` (instance method call) is not — Mojo treats it as a "dynamic value" even though the length is statically known, so iterate over the parameter pack length, not an instance-method call.

See `ROADMAP.md` for what's planned next on top of this (a `fit()`-style training loop, more layer wrappers, etc).

---

## File Map

```
basalt/
├── __init__.mojo              # comptime dtype=float32, nelts, seed, epsilon
├── autograd/
│   ├── symbol.mojo            # Symbol (TrivialRegisterPassable, 20B)
│   ├── graph.mojo             # Graph builder; holds inputs/params/nodes/outputs
│   ├── node.mojo              # Node: op + List[Symbol] in/out + AttributeVector
│   ├── params.mojo            # Param (init spec) + ParamDict (parallel arrays)
│   ├── attributes.mojo        # Attribute (name+data bytes) + AttributeVector (fixed array)
│   └── ops/
│       ├── ops.mojo           # OP enum; forward_op/backward_op dispatch
│       ├── basics.mojo        # ADD,SUB,MUL,DIV,EXP,LOG,POW,DOT,SUM,MEAN,MAX,FLATTEN,RESHAPE,TRANSPOSE,FMA
│       ├── mlops.mojo         # SIGMOID,RELU,LEAKYRELU,TANH,CLIP,SQUEEZE,UNSQUEEZE,SLICE
│       ├── matmul.mojo        # Optimized DOT kernel (tiled, parallelized)
│       ├── conv.mojo          # CONV2D (im2col + gemm)
│       ├── pool.mojo          # MAXPOOL2D
│       └── dynamics.mojo      # CONCAT, SPLIT (runtime List[Symbol] inputs/outputs)
├── nn/
│   ├── tensor.mojo            # Tensor[dtype]: heap array + TensorShape, refcounted
│   ├── model.mojo             # Model[g]: comptime specialization, forward/backward
│   ├── module.mojo            # Layer trait, build_graph (reflection), Sequential (variadic), FlattenLayer
│   ├── optim.mojo             # Adam[g, params]: vectorized parameter update
│   ├── activations.mojo       # ReLU/Sigmoid/Tanh/Softmax/LogSoftmax graph builders + ReLULayer
│   ├── loss.mojo              # MSELoss/CrossEntropyLoss graph builders
│   ├── initializers.mojo      # initialize_tensor: random_uniform/random_normal
│   └── layers/
│       ├── linear.mojo        # Linear(g, inputs, n_outputs) → DOT + ADD; LinearLayer
│       ├── conv.mojo          # Conv2d(g, ...) → CONV2D; Conv2dLayer
│       └── pool.mojo          # MaxPool2d(g, ...) → MAXPOOL2D; MaxPool2dLayer
└── utils/
    ├── collection.mojo        # Collection: dense Tensor storage + O(1) symbol-id -> slot index_map
    ├── tensorutils.mojo       # fill, elwise_op, broadcast, reduce, transpose, accumulate_grad
    ├── math_util.mojo         # add/sub/mul/div/exp/log/sqrt_simd/max_simd as thin SIMD def wrappers
    ├── rand_utils.mojo        # rand_uniform, rand_normal, MersenneTwister
    ├── bytes.mojo             # Bytes[N]: fixed stack array of UInt8 (for Attribute storage)
    ├── dataloader.mojo        # DataLoader + Batch: slice tensors into batches
    ├── datasets.mojo          # BostonHousing, MNIST: CSV loaders
    ├── perf_utils.mojo        # PerfMetrics: timing per node (used with DEBUG flag)
    ├── tensor_creation_utils.mojo  # to_numpy, to_tensor, copy_np_data (Python interop)
    └── onnx_utils.mojo        # load/export ONNX models
```

# MLIR: A Compiler Infrastructure for the End of Moore’s Law

MLIR is not a single IR, but a framework for building and composing multiple IRs (dialects). It provides reusable infrastructure for transformations, optimizations, and lowering, making it easier to connect high-level domain-specific operations to low-level hardware-specific code.

## Why MLIR exists

MLIR was created (by Google, now under LLVM umbrella) because traditional compilers (like LLVM alone) struggle with:
- High-level abstractions (e.g., TensorFlow graphs, domain-specific operations)
- Targeting heterogeneous hardware (CPUs, GPUs, TPUs, accelerators)
- Its multi-level representation solves the "impedance mismatch" between high-level ML frameworks and low-level hardware code
- Not generalizable (ex. either ONLY ML or NO NL)

## How it works:

MLIR allows users to make new IRs by giving them:
- A dialect mechanism for defining custom ops, types, and attributes.
- Declarative tools (ODS) to auto-generate boilerplate.
- A shared optimization and pass infrastructure.
- A way to compose dialects together and progressively lower.
- Backend connections to LLVM, GPUs, and accelerators.

### Dialect System

Creating a dialect is like a namespace for the IR

Each dialect defines:
- Operations: mydialect.matmul, etc.
- Types: tensor<4x4xf32> or custom types
- Attributes: Metadata on ops (transpose, padding size, etc.)

### Operation Definition Framework

- MLIR offers a declarative language (TableGen) to define ops, types, and constraints
- This drastically lowers the cost of defining a new IR.

#### Example:

Instead of writing boilerplate C++ to define a matmul op, you write a spec that MLIR uses to auto-generate verification, parsing, and printing code

### Infrastructure for Transformations

- Provides pass management and transformation tools so dialects can:
    - Optimize ops (e.g., fuse, tile, vectorize)
    - Lower into another dialect
    - Reuse existing generic passes (like dead-code elimination or inlining)
- You don’t need to re-implement the compiler "plumbing"

### Multi-Level IR Coexistence

- Your new dialect doesn’t live in isolation. It can coexist with others:
    - TensorFlow ops can sit next to your custom ops
    - You can lower your dialect into existing ones like linalg, affine, or LLVM
- Gradually moving from high-level abstractions to low-level code allows for progressive lowering

### Integration with Existing Ecosystem

- Built-in paths to LLVM IR, SPIR-V, GPU backends, etc.
- New dialect automatically gets a way to connect with real hardware

## Example:

### Step 1: High-Level TensorFlow Graph

In TensorFlow, you might write:
```python
C = tf.matmul(A, B)
```

At this point, it’s represented as a high-level graph op (something like tf.MatMul).

### Step 2: MLIR TensorFlow Dialect

MLIR has a TensorFlow dialect that captures this directly:

```python
%0 = "tf.MatMul"(%A, %B) {transpose_a = false, transpose_b = false} : (tensor<...>, tensor<...>) -> tensor<...>
```

This still looks very close to TensorFlow, but it’s now inside MLIR, which means it can use MLIR’s passes and infrastructure.

### Step 3: MLIR Linear Algebra Dialect (Linalg)

We lower the high-level op into a more structured linalg dialect:

```python
%0 = linalg.matmul ins(%A, %B : tensor<...>, tensor<...>) -> tensor<...>
```

This expresses the same computation, but in a form that can be systematically optimized (e.g., tiling, fusion, loop transformations).

### Step 4: MLIR Loop / Affine Dialect

The matmul is lowered into explicit loops over indices:
```python
affine.for %i = 0 to M {
  affine.for %j = 0 to N {
    affine.for %k = 0 to K {
      %c = load %C[%i, %j]
      %a = load %A[%i, %k]
      %b = load %B[%k, %j]
      %prod = mulf %a, %b
      %sum = addf %c, %prod
      store %sum, %C[%i, %j]
    }
  }
}
```

Now the computation is explicit, enabling optimizations like vectorization, parallelization, and loop tiling.

### Step 5: Lowering to LLVM or GPU Dialects

If targeting a CPU, this can be lowered into LLVM IR and compiled to machine code.

If targeting a GPU, MLIR can lower into GPU dialects (e.g., CUDA kernels, SPIR-V for Vulkan). Example GPU lowering might assign each (i, j) iteration to a GPU thread.
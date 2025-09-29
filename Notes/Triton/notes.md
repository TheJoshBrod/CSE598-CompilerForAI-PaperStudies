# Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations

Triton presents: 
- new C based language
- IR based on LLVM to represent tensor programs for deep learning
- Set of tile-level optimization passes


## Introduction

Triton achieves high performance by letting you write low-level GPU kernels in a high-level representation. Instead of hiding the kernel behind predefined operations, Triton offers "intent-revealing language" that gives the compiler enough structure to automatically optimize memory access, tiling, and parallelism.

Other existing TVM-style loop schedulers struggle with irregular or sparse workloads because they assume traditional "affine loop nests" with consistent regular memory access. Triton does not make this assumption by giving the developer more flexibility through that higher level representation to write your own kernels beyond static tensor layouts.

### Example:

#### Scenario:
In every group of 4 DNN weights, 2 are non zero.

The nonzero ones' position is stored in metadata

We need to load ONLY those 2 non zero values, apply masking, and skip zeros during compute

#### Triton method

```python
# offsets is dynamic / non-affine
idxs = base_idx + tl.load(metadata_ptr)
weights = tl.load(weight_ptr + idxs, mask=mask)
activations = tl.load(x_ptr + activation_idxs)
acc = tl.sum(weights * activations, axis=0)
tl.store(out_ptr + output_idxs, acc)
```

Kernel Steps:
1. Load id's of weights
2. Load masked weights (only keep nonzeros)
3. Load activations 
4. Calculate sum of nonzero weights * activations
5. Store output back in memory


#### Why TVM would struggle

Assumes:
- Loop bounds are statically known.
- Indices are affine expressions of loop variables.
- Tensors are dense and rectangular.
- Access patterns are regular enough for schedulers to tile and reorder.

While in this example:
- You need index lookups based on metadata, not known static like assumed
- Accesses are indirect (e.g., A[indices[i]])
- The iteration space is not rectangular, not a n-d array/loop
- Masking is needed to avoid out-of-bounds or zero loads
- Fusion of metadata handling + compute is required for high performance

Instead TVM would would likely fall back on non optimized code altogether

### Other solutions

Micro Kernels: Hand written tile level kernels
- Requires a lot of manual labor and lacks portability
- Compiler backends lack support for tile-level operations/optimizations

### Triton Solution:

Triton-C:
- Allows you to write GPU kernels without CUDA 
- Backend handles optimizations
- Stable frontend for compilers and low-level users

Triton IR:
- LLVM/Tile-oriented IR generated from Triton-C or more commonly Python DSL
- Encodes loads/stores, masks, indexing, and parallelism at the block level and is later lowered to LLVM IR

Triton-JIT: Backend compiler that
- Runs target-independent optimization passes on Triton-IR
- Performs target-specific lowering to GPU-ready LLVM IR
- Auto-tunes meta-parameters like tile sizes and warp counts in real time



# Operator Fusion in XLA: Analysis and Evaluation

XLA provides operator fusion on the graph level (does not work in Eager mode)

Contributions:
- Describes XLA fusion mechanism of XLA 
- Evaluate Performance on JAX-based RL task
- Low Level profiling of XLA programs to identify limitations

## Related Works

### Optimized ML Frameworks:
- Traditional Deep Learning Frameworks (PyTorch, TensorFlow w/o XLA, MXNet) use cuDNN/cuBLAS or prewritten CUDA kernels for flexibility, but this limits deep optimization
- Newer compilers try to generate specialized kernels rather than relying on vendor primitives
- XLA apply rule-based fusion for simple ops but depend on cuDNN/cuBLAS for heavy operators
    - Fast compile but limited optimization depth
- TVM generates custom kernels and apply auto-tuning
    - More flexible with higher potential but long compile

### Operation Fusion in DL
Search based fusion strategies (unlike XLA): 
- Ivanov et. al: Exhaustively explore data layouts and fusion in transformers through cutting data movement
- PET enables partially equivalent transformations (ex. merging tensors to do one convolution instead of two) and uses beam search to expand optimization possibilities
- DeepCuts uses a greedy, cost-model–guided approach to fuse operations and tune CUDA parameters, but only handles limited fusion types and is expensive to adapt to new ops

##  XLA’S MULTI-PASS OPTIMIZATION

XLA Computation Graph Optimization:
- JAX or TensorFlow make a computation graph
- Iteratively optimize:
    - SPMD Partitioner: Partitions tensors to operate in parallel across devices
    - Optimization: Simple rule based operator conversions
    - Simplification: Inlining, constant propagation, WhileLoopSimplifier removes dead tuple elems
    - Fusion: Performs vertical operation fusion, ex. (A = matmul(x,y) B = add(A, b) VS C = add(matmul(x,y),b))
    - Horizontal fusion. Performs the horizontal operation fusion, ex. instead of serial y1 = relu(x) y2 = sigmoid(x) do both in parallel on same input
    - Post fusion optimization. Combines small nondependent collective operations into larger combined operations
    - etc.

## Operation Fusion

Paper predominantly focuses on its contributions to operator fusion

Four fusion strategies

### Instruction fusion:

Simple vertical operation fusion

ex.

```python
# Bad
A = matmul(x,y)
B = add(A,b)
C = ReLU(A,B) 

# GOOD
C = ReLU(add(matmul(x,y),b))
```

Reverse post order traversal to find if functions should be fused:
- Expensive operations not fused (convolution, sort, all reduce, etc.)
- Checks if fused kernel would be too large for GPU hardware
- Will fused kernel cause nested loop?

### Fusion Merger
- Merges existing fusion kernels into larger fused kernels when safe (e.g., to reduce memory writes and kernel launches)

So combines the already merged kernels from instruction fusion even further


### Multi-Output Fusion
- With one fused kernel, load the input once into memory (save to reg) and compute both functions in parallel within the kernel
- Allows one fused kernel to produce multiple outputs — either from sibling ops (shared inputs) or producer-consumer chains — reducing global memory reads

### Horizontal Fusion
Fuses independent kernels (often with different shapes) into a single GPU launch to reduce kernel launch overhead
# Memory Safe Computations with XLA Compiler

DL frameworks optimize for speed but do not consider memory size

When XLA traditionally compiles it makes an internal representation of data flow, but doesn't consider the memory size of the program.

XLA compiler extension that adjusts data flow representation to user specified memory limit.

This extension rewrites or modifies the dataflow graph to use less memory based on memory requested

## Motivation

If a program runs out of memory it crashes, so its not just a performance issue

Memory is expensive so often a financial cost bottleneck

Some algorithms require NxN matrices which has quadratic memory usage


To help adjust the amount of memory the program uses is usually a programmer's task but this extension handles it by itself

## Related Work

### Distribute Work across Server

Typically to address mem issues throw more GPUs or computer clusters at problem

Sharding (like in TorchTitan) increases memory bandwidth across more hardware but is financially expensive

### Compilers

There are compilers to optimize human expressed code to run well on specific hardware but mostly for time complexity not memory.

## Memory Efficient Matrix/Linear Alg. Operations in XLA

Chose XLA because it is better integrated

### Match and Replace Pass

Searches for expressions in data flow graph which we know a more efficient version exists. 

ex. search for expressions with naive form of euclidean distance  between vectors of length n and m with a dimension d and replaces with optimal

Naive: 
- Broadcast over dimension D and creates temp tensor of size n x m x d `(x_nd - y_md)^2` while optimal is n x m `∑_d x^2_nd + y^2_md - 2x_nd*y_md`

### Reordering

Reorders commutative/associative functions to be more memory efficient


### Data flow graph splitting


Data-flow graph splitting restructures the computation graph into smaller subgraphs that are executed sequentially so memory usage stays within a set limit. Intermediate results may be temporarily stored, freed, or recomputed so that only a portion of the original graph is active at any time.

### XLA Limitations

- Weak linear algebra type system
- Default memory allocation manager is not aware of memory limits
- If multiple "smaller" tensors each individually fit on GPU, multiple may be activated at the same time so they become too large all at once. 
# Glow: Graph Lowering Compiler Techniques for Neural Networks

Glow is an ahead-of-time (AOT) compiler design that makes it easier to run neural networks efficiently across many types of hardware by breaking down computation into manageable, optimizable layers.

## Problem

Neural networks are written as abstract graphs of operations (e.g., matrix multiplications, convolutions, activations).
To run them fast on real hardware, you need specialized code that makes the best use of memory, parallelism, and hardware features.

Writing optimized implementations for every operator on every possible hardware target is too much work.


## Solution: Two-phase IR

Glow introduces a **two-phase intermediate representation (IR)** to bridge the gap between high-level neural network graphs and low-level machine instructions:

### High-Level IR
- Strongly typed representation of neural network operations
- Used for **domain-specific optimizations** (e.g., operator fusion, simplification, transformations)

### Low-Level IR
- Looks like assembly instructions working with explicit memory addresses
- Enables **machine-specific memory optimizations**:
  - Static memory allocation
  - Copy elimination
  - Instruction scheduling

The two-phase design **separates graph-level optimizations from hardware-level optimizations**, making compilation cleaner and more effective.


## Lowering Phase

- Glow **simplifies complex neural network operators** into a **small set of linear algebra primitives**
- Benefits:
  - Shrinks the operator space.
  - New hardware backends only need to optimize a few primitives instead of hundreds of operators.
  - Makes it much easier to add support for new hardware.


## Strong Focus on Memory Optimization

Unlike many compilers that prioritize kernel fusion or compute scheduling, Glow focuses on **explicit memory layout and addresses**

This enables:
- **Static memory allocation** → allocate once, reuse efficiently
- **Copy elimination** → avoid unnecessary data movement
- **Instruction scheduling** → reduce stalls in memory-bound workloads

These memory optimizations reduce overhead and naturally lead to **faster execution**.


## Ease of Hardware Support

- Thanks to lowering, Glow backends only need to implement a small set of primitives
- This reduces engineering effort compared to compilers like **TVM** or **XLA**, which often require larger operator coverage


## End Goal

Generate **machine-specific optimized code** that:
- Takes advantage of specialized hardware features
- Runs efficiently without requiring developers to hand-tune every operator
- Provides predictable performance since it is **ahead-of-time compiled**

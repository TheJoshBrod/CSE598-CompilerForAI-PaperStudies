Relay: A High-Level Compiler for Deep Learning (Aug 2019)

Scenario: Attempt to run NN on local machine (no api) with minimal hardware so you have to employ quantization on the model

Ideally should be straightforward to:
- Expressivity: Write models with control flow, first-class functions, and data structures (trees, graphs, etc.)
- Composability: Add new optimizations (quantization, operator fusion, partial evaluation)
- Portability: Add new hardware targets (GPU, TPU, etc.)

Previously difficult due to each IR treating components as disconnected set of programming tasks

ex. operators defined in c++, connected by dataflow graph, then scripted in python

Difficult to analyze cross lanuage boundaries between optimization and deployment

Relay seeks to solve this by:
- Create an IR to treat deep learning programs as if they were functional programs, so you can express any model in a clear, structured way
- Use compiler mindset to unify deep learning features so they're easier to extend and combine
- Seperate the model representation from the hardware so optimizations can be reused across platforms

Existing Technologies:
- Define-and-run: static, efficient, less flexible.
- Define-by-run: dynamic, easy to use, less efficient.
- Low-level tensor compilers: fast ops, not full models.
- Deep learning compilers: optimize full models across hardware.
- Languages for DL: push the boundaries by making DL a first-class citizen in a programming language.

Takes existing IRs designined for ML, and adds features such as tensor and tensor operations.


What does Relay take from existing IRs:
- Multiple Outputs: After a tensor operation, split tensor into seperate chunks
- Let: Handles variable/function scoping
- Control Flow: Adds if statements and First Class recursive functions
- First Class Functions: Adds support for it
- Data Abstraction: Adds support for custom types beyond default primitives

How does Relay handle Type System:
- Tensor Types: Forces tensors to ONLY contain primitive types (ints, doubles, etc.)
- Operaters and Type Relationships: Opaque operators enable backend to choose different lowering stats based on hardware (how operators work is up to lower level of toolchain)
- Type Inference: Implicitly infer typing based off existing algorithm

How does the compiler work?:
- Frontend: Produces a AST for Relay from "any source" and turn into Uniform I
- Compiler: Error checks, Optimizes, and domain specific optimizations on program through multi-pass pipeline
- Backend: Relay specific operators are turned into hardware specific executables

Finds operators that can be "fused" and marks them

```
ex. y = ReLU(mx+b)

1. temp = m * x
2. temp1 =  temp1 + b
3. y = ReLU(temp1)

Combine these into:

y = FusedKernel(x, m, b)
```

Accomplishes this by representing ALL operators in secondary IR, allows lower level tools on the compiler toolchain to choose what to fuse based on hardware. 

Also can generate fused kernels/operations by itself, doesnt have to rely on outside tools:
- Identifies fusion eligible subexpressions (labeled as primitives)
- Turns them into platform speicifc code, combines the ones that can be combined


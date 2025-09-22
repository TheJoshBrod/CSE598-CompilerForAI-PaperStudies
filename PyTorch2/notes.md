PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation

## Abstract:

Introduces two extensions to PyTorch:
- TorchDynamo: JIT compiler that "intercepts" PyTorch programs to create Functional Transformation graph IR of program 
- TorchInductor: Compiler backend that lowers TorchDynamo graphs down to triton or llvm

## Intro:

Eager Mode: 
- Ex. PyTorch (by default) and JAX
- Define-By-Run: The model defines itself as the program executes line by line 
- Each operation immediately dispatched to backend (GPU/CPU)
- Easy to use and debug, but prevents global graph-level optimizations

Graph Mode:
- EX. TensorFlow, Caffe, Theano, CNTK 
- Define-And-Run: Compile model into a graph and then later execute that graph
- The entire model’s graph is known ahead of time
- Allows for global optimizations (e.g., kernel fusion, constant folding).
- Less flexible for developers (no debuggers, dictionaries, lists, custom classes, 3rd party libraries, etc.)


Solution?

Torch Dynamo:
- Python-Level JIT compiler
- Allows for runtime Graph compilation
- Uses CPython to modify Python Bytecode before executed

TorchInductor:
- New compiler backend for TorchDynamo
- Pytorch -> Triton or C++/OpenMP
- Uses similar extractions to Eager mode


## Relate Works

### Graph Capture Methods

Torch JIT Trace:
- Record/Replay: Runs program once with example input to create static graph
- Cannot handle data-dependent control flow
- Produces a TorchScript IR graph that runs independently of Python

Torch JIT Script:
- Compiles models via static analysis of Python code (AST)
- Attempts to make PyTorch code behave like a static language
- Only supports subset of python

Lazy Tensors:
- Instead of running ops immediately (Eager mode), records ops into a computation graph
- Execution is deferred until the graph is materialized (often after forward + backward).
- Once all of the passes occur the graph is sent as a single batch to use XLA compiler
- Issue: Compiling graph is expensive (new graphs can take 10s-100s of ms)
- Hashes graphs to not repeat compilation

Torch FX Symbolic Trace:
- Similar to JIT Trace but traces at Python level not PyTorch c++ level
- Makes FX Graph of tensor operations to transform and lower into compilers/backends
- Records more operations than JIT Trace bc python level, but no dynamic data dependent ctrl flow
- Limited to PyTorch visible operations (ex. NO math.random)

torch.onnx.export:
- Combines JIT Script w/ JIT Trace
- Same issues both have

### Alternative to PyTorch

JAX:
- Doesn't have the same challenges TorchDynamo solves
- Directly tied to XLA, whole stack is optimized around graph capture + compilation
- Eager by default, but compiler/XLA (graph mode) is the central design goal
- Does NOT support data-dependent control flow
- Functions must be pure and functional (no mutation of Python state)


PyTorch:
- Eager by default, primary user experience
- TorchDynamo was built to alleviate issues with eager-only through FX graphs
- Not tied to specific compiler backend: FX Graphs can then be lowered to backends
- Supports dynamic Python features (control flow, mutable state, external libraries) making capture more complex but flexible

## TorchDynamo

Instead of removing/replacing python, uses CPython's JIT to compile python to bytecode

Once in bytecode, replaces bytecode with compiled "artifacts" that fuse pytorch operations


## TorchInductor

Deep Learning compiler backend that lowers Dynamo's FX graphs into more efficient fused kernels

Combines higher level operations into "loop-level" IR then generates optimized code through LLVM or Triton.

Focuses on operator fusion, dynamic shape support, and backend portability


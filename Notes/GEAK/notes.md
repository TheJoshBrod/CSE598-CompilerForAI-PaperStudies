# GEAK: INTRODUCING TRITON KERNEL AI AGENT & EVALUATION BENCHMARKS

## Provides two innovations:

Evaluation Suite:
- Benchmark suite for Triton based GPU Kernels

GEAK (Generating Efficient AI-centric GPU Kernels)
- Uses LLMs to generation AMD kernels


## Introduction 

GEAK:
- Agentic LLM framework for Triton kernel generation
- Multiple agents for generation, evaluation, reflection, and optimization for iterative refinement
- Runs as a final rewrite/recompile before execution, NOT AT RUNTIME


TritonBench:
- Each benchmark contains short task descriptions and correctness harness (unit tests, shape checks, etc.)
- Throughput performance is measured after correctness is verified

ROCm Triton Benchmark:
- Extracted sections of AMD ROCm turned into Triton compatible tasks
- Useful because sample of real-world patterns (fused ops, ML workloads, tensor transforms)

## Related Works

### Benchmarks

There are existing program synthesis benchmarks that focus on correctness using curated or crowd sourced test cases. 

There have been efforts to put automated LLM test generation for expanded coverage and scalability but these typically focus on sequential or CPU tasks.

GPU-Oriented benchmarks, such as TritonBench or KernelBench, focus on both correctness AND performance of existing GPU kernels.

None of these examples offer minimal short tasks for LLM generated kernels. Thats where this new benchmark suite fills in.

### LLMs for Code generation

KernelLLM: PyTorch -> Triton code
- Trained LLM
- NOT a reasoning model
- Only trained for PyTorch to Triton

GEAK: Natural Language AND Reference Code -> Triton code
- LLM agnostic as it is an agentic framework
- Encourages more compute for LLMs by using longer reasoning chains, multiple candidate generations, and larger context windows
- Also uses parallel scaling and multiple generations

## Benchmarks

Triton Benchmark:
- Took TritonBench-G kernels and made them AMD GPU compatible

ROCm Triton Benchmark:
- Publicly available kernels written by Triton engineers

### Evaluation

- Call accuracy: % of AI-generated kernels that can compile and run without error
- Evaluation accuracy: % of AI-generated kernels that satisfy all unit tests (only for those that compile)
- Speedup: Relative execution time improvements over ground truth (only done for kernels who passed unit tests)

## GEAK

### Pipeline

GEAK agents:
- Generation Agent: Produces multiple candidates using extra inference compute

- Evaluation & Reflection Agents: Leverages previous failures or performance metrics to guide the next generation round (a Reflexion-style feedback loop)

- Optimization Agent: Uses compute-intensive reasoning to suggest performance tweaks or kernel improvements

### Modules

1-shot prompting: Pull most similar triton code from dataset. Unclear how they did this, possibly code-embeddings or AST?

Knowledge Injection: Enhance prompt with domain specific efficient triton kernel including hardware specs 

Reflexion: When an agent fails, store "reflection" in memory buffer that they recall for future tasks 

LLM selection for kernel agent: Underling LLM significantly impacts performance

LLM as Optimizer: Iteratively generate and refine solutions to optimize problems by generating candidates which are evaluated and included in next prompt

Debugging trap: Impose limit on number of debug attempts so doesnt infinite loop

Parallel Scaling: Run GEAK indepently in parallel to introduce diversity helping yield correct and faster kernels

## Experiment

### Naive approach

Direct zeroshot prompting LLMs to generate kernels:
- Very poor results
- Even SOTA LLMs struggled

Oneshot prompt LLMs to generate kernels:
- Gave a single relevant example
- Slightly better in overall accuracy and performance
- Gemini 2.5 Pro saw higher accuracy but lower speedup than direct prompting in some examples

Overall clear platform-specific challenges for naive approaches

### GEAK

Had significant correctness and performance gains relative to the other methods




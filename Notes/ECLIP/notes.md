ECLIP: Energy-efficient and Practical Co-Location of ML Inference on Spatially Partitioned GPUs

## Definitions:

CU masking: Restricting a kernel to ONLY run on a specific subsection of the AMD GPU (only give the kernel a slice)

Stream: Place to submit kernels, typically stream doesn't say WHICH CU, but we're changing that

CU mask streams: Initially create pools of reusable partitions of GPU, then dynamically allocate kernel groups to them

GEMM: General Matrix Multiply

## Abstract

Problem: Power waste and inefficiency in hardware utilization

Solution? Utilize ONE gpu for running multi-model systems in an effort to shrink power waste


## Intro

GPUs can be broken into smaller "compute resources" (ex. Compute Units (CU) AMD, Streaming Multiprocessors (SM) NVIDIA)

These compute resources are not fully utilized when they are idle.

When running multiple requests on the same GPU latency and throughput are throttled due to shared compute and memory resources


Spatial Partitions:
- Hardware default: No true partitions of workloads 
- Inference servers (software layer): Attempt this on top of the GPUs

Prior work:
- Attempts to partition ENTIRE models, not subsections due to high overhead

ECLIP: Scheduling and Resource Allocation to achieve kernel wise spatial partitions with minimal repartition overhead

Scheduler:
- Restrict which CUs a kernel can use, similarly how multiple models share a GPU interfere
- Create pools of reusable partitions of GPU to dynamically allocate kernel groups to them

## Existing Energy Efficiency Techniques:

DVFS: Dynamic Voltage and Frequency Scaling
- Dynamically adjust clock freq and supply voltage depending on workload or constraints
- Weakness: limited impact because underutilization of idle resources causes power leakage

Raw CU Masking:
- Limiting work to subset of CUs can save power by power-gating
- Weakness: limited to clock gating or entire GPU gating


## Limitations of GPU partitioning for ML inference

### Model-grain right sizing:
- Model must be right size to use minimal amount of resources and maintain performance
- ML models are sized to use minimal resources while meeting QoS target
- Resources allocated to a model is the "knee" in resource vs latency curve
- 

###  Kernel-grain Right-Sizing
- Even if ML model is the right size, if we frequently repartition which kernels can be called on which partition, there is high overhead
- Pre-execution CU masking require modifying PyTorch code (before program is ran/compiled if we want to manually assign CUs to kernels we would have to modify pytorch code)
- Existing solution 1: Elastic Room attempts to compile GEMM kernel and selects appropriate kernel size. Only works with GEMM kernels, lots of runtime modifications, AND not PyTorch friendly
- Existing Solution 2: KRISP proposes architecture changes. But not compatible with modern GPUs
- Existing Solution 3: libsmctrl uses undocumented CUDA api callbacks to enforce SM mask globally. Not widely supported and impractical


### Challenges towards practical kernel-level spatial partitioning of ML inference
- Slow to partition kernels to CUs

## Initial Experiment

### Observation 1: CU masking IO control calls have high and unpredictable timing overheads

Calling CU mask IO control too often is costly and unpredictable

### Observation 2. The GPU can accommodate a finite number of CU masked streams before experiencing slowdown due to queue oversubscription

GPU can only handle 7 CU masked streams efficiently. Each stream creates software command queue which is limited by the number of physical hardware queues. (similar to having more threads than CPU cores). 

## ECLIP Framework:

ECLIP has a trade off for spatial repartitioning granularity with repartitioning overheads

1. Creates pools of streams with pre-defined CU masks
2. Runtime scheduler to intercept and redirect kernels with respect to inter-kernel data
3. Resource allocation optimizer to assign optimal CU assignments

### Runtime Scheduler

#### Pre-allocated CU Masked Streams

Pre-allocates 7 CU mask streams of incremental sizes of 15 CU (size of a larger compute component of GPU)

Each ML model is handled by a designated worker thread for inference requests. Tasks must be broken into CU bundles of increments of 15.


#### Redirecting Kernel Dispatch to Pre-allocated CU Masked Streams

Intercepts kernel calls and uses lookup table to determine which CU it should send kernel to

#### Enforcing Data Dependencies

Since we are changing which kernels go where, the normal "in-order" assumption of calls is serial

So if kernel b is redirected to a different CU-masked stream than kernel a, they may run out of order (dependency issue)

ECLIP fixes this by adding a HSA Barrier packet to say "don't start this kernel until X finishes" 

Uses 2D vector of user streams x complete signals:
- When kernel is dispatched checks predecessor
- If yes, launch immediately. Otherwise insert barrier packet tied to A's completion

Barriers are costly though because:
- Tracking overhead
- Barrier is just a stall so causes delays
- Overhead may be larger than execution time

ECLIPS mitigation:
- Models cost of barriers
- Budget how many barriers to use using a resource allocation model
- Optimizes paritioning decisions to reduce # of dependencies (less dependencies, less barriers)


### Resource Allocation Optimizer

Assigns resources to groups of kernels


#### Minimum CU Threshold

Read the minimum CUs per kernel before noticeable slowdown

#### CU Switching overhead

Instead of adding barrier packets after every kernel, use it for GROUPS of packets

#### Fairness Among Workers for Energy Efficiency

Create one objective function per worker
- Optimization objective aims to minimize worker execution time

Instead of squeezing maximum performance out of just one model, it balances CU allocations so all workers finish closer together. This avoids idle CUs, which are wasteful on GPUs without fine-grained power gating.

#### Optimization Formulation

Our optimization goal is to minimize execution time for all workers while maintaining fairness

This is done by allocating a predefined set of configs align with SE org (15 CUs)

What influences execution time:
- Number of CUs allocated
- Slowdown from co-location with other workers’ kernels (even if two workers dont share CU they share memory, cache, and command processor)
- Overhead of CU config switches
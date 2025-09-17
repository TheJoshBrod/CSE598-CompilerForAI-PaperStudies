Existing ML frameworks (pytorch, tensorflow, etc.) map operators to vendor provided kernels (cuDNN, MKL-DNN).

This is bad because it requires signifigant engineering effort to manually tune for each hardware platform.

Limits development and operation of new operators and specialized accelerators

Existing compilation technique for tensors consist of "search-based compilation"

Users declare operator or "sub-graph" of multiple operators in high level language, compiler searches for program tailored towards hardware platform.

To do this in an effective way, the search space is quite large.

The existing search approaches fail to capture effective optimizations combinations due to relying on predefined/manually written templates (TVM, FlexTensor) or aggressive pruning by evaluating incomplete programs, preventing to cover entire search space

Ansor's Solution? Automatically generate a large search space with comprehensive coverage of optimizations and gives every tensor program in the space a chance to be chosen

Challenges:

1.) Implement automatic generation of potential instructions for a tensor operation

2.) Efficiently search space, throw out incomplete/malformed programs

3.) Focus optimization efforts on operations that deserve the effort

Ansor tries to solve this by using a hierarchical representation of the search space and uses evolutionary search to help decouple highlevel structure with low level details. It also focuses on those more "complex" tasks that require more optimzation.

```
In summary, this paper makes the following contributions:
• A mechanism to generate a large hierarchical search space of tensor programs for a computational graph.
• An evolutionary strategy with a learned cost model to fine-tune the performance of tensor programs.
• A scheduling algorithm based on gradient descent to prioritize important subgraphs when optimizing the endto-end performance of DNNs.
• An implementation and comprehensive evaluation of the Ansor system demonstrating that the above techniques outperform state-of-the-art systems on a variety of DNNs and hardware platforms.
```

How I'm organizing presentation:

1.) Existing ecosystem 
	- hardware (CPU, GPU, FPGA, ASICs, etc.)
	- Each with their own ISA
2.) Explain the issue
	- There are a few common high-level operations (matmul, conv2d)
	- New ones always being made
3.) Prior attempts at solving this:
	- TVM (Higher level IR to represent operations)
		o Good as an IR, but doesn't actual optimized compile to hardware 
		o Difficult to search "large and complicated" search space for optimizations
	- Template Guided Search (Compiler searches through list of templates)
		o Good performance boost
		o Developing templates requires serious effort
		o only cover limited program structures
		o Can NOT break down fixed templates and recompose them during search
	- Sequential construction based search
		o Sequentially make optimizations to program (ex. tile loop by factor 8, parallelize across threads, etc.)
		o Doing it incrementally prevents invalid or incomplete scheduling
4.) What makes Ansor Different?
	- Combines the Templated Guided Search AND Sequential construction based search
	- "De couples high-level structures from low-level details"
	- Constructs search space for graph removing need for developing manual templates
5.) How does Ansor work?
	a.) Takes operation fusion from Relay, converts DNN from popular formats (tensorflow, etc.) to partitioned small subgraph
	b.) Program Sampler: Generating large search space for compute graph
		- Organizes search space as two layer
        - Sketch layer: high level representations (similar to template guided search)
        - Annotation layer: Lower level choices (tile size, parallel, etc.) (similar to Sequential Construction)
        - Benefit: Sample a wide variety of both high level and lower level iterations
    c.) Performance tuner
        - Starts from sampled programs and fine-tunes them iteratively with an evolutionary algorithm
        - At each iteration, promise programs and resample new candidates via mutation and crossover
        - Crossover: combines parts of two good schedules.
        - Mutation: tweaks specific schedule decisions (e.g., tile size, loop order).
        - Unlike sequential construction (which builds step-by-step and can’t revise earlier decisions), evolutionary search can rewrite schedules out of order, helping Ansor escape local performance optima.
		- Queries a cost model to "predict" runtime of a canidate without having to run
	d.) Task Scheduler
		- Exponetial search space
		- Partitions graph of DNN into subgraphs
		- Negligible effect on performance as DNN is typically a layer by layer model
		- Uses scheduling algorithm based on gradient descent to allocate resources to sub graphs that are more likely to improve
6.) Program Sampler
	- Fixes issues of manual enumeration being impractical, auto expand search by set of flexible derivations rules
	- Avoids aggresive pruning based on evaluating incomplete programs, by sampling randomly we have a large search space
	- Sampling is not the ONLY thing relied upon, its just a starting point because we fine tune later
	- Sketches:
		o High level optimizations
		o Uses "derivation lists" or condition -> application rules to make changes
		o Input: subgraphs created by Relay algorithm represented as equal but different forms:
			- Mathematical expression
			- naive program from loops
			- Directed Acyclic Graph (DAG)**** This is the one we care about
		o Algorithm: Visit all nodes in topological order to build iterative structure
			- For compute heavy nodes with reuse opportunities (conv2d, etc.): build tile and fusion struct
			- For simple element-wise nodes (ReLU, etc.): inline the operation
			- NOTE: new nodes can be added during sketch generation 
		o Output: DAG -> List of Sketches
	- 

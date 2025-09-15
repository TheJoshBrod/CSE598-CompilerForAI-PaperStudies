GEPA: REFLECTIVE PROMPT EVOLUTION CAN OUTPERFORM REINFORCEMENT LEARNING

How should LLMs be optimized for best results given the tools provided?
- Reinforcement learning with verifiable rewards (RLVR) algorithms such as Group Relative Policy Optimization

Issue with RL, often takes thousands of trials to learn policy. Both time and monetary cost.

With agentic/reasonuing models we have a "thought" process of how the LLM accomplishes its task. This allows for a new method of RL instead of having to optimize a scalar policy.

This is accomplished through GEPA (Genetic-Pareto) that implements contextual reflection with multi-objective evolutionary search. In simpler terms mutates prompt based on feedback from each LLM output.

To avoid local optima, uses Pareto front: stochastically explore the top performing prompts 

Uses both mutation and crossover to propose increasingly effective canidates.

Has "Reflective credit assignments" to map where on its trajectory it assigns blame to. (ie tool calls, bad information, etc.)

Grades each "hop" seperately to produce per hop feedback.

Their setup
- gpt4.1 mini and Qwen3 8b
- MIPROv2
- GRPO
- GEPA


Observations:
- Reflective prompt evolution is highly efficient can be better than tweaking weights
- Clearer Instructions sometimes beat few shot prompts
- Optimization next-canidate strategy is critically important
- Instruction optimized prompts are compute cheaper and generalize better than fewshot
- Crossover strategy was succesful
- Merging prompts needs future research on when to invoke prompt merges


They tested this algorithm at inference time (deployment NOT training time)

Included the target task itself in both the training set and Pareto set. They did this to "overfit" to the specific task to iteratively propose a better implementation for each task. OR lessons from one task can transfer to the other.


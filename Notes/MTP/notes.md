# Meaning Typed Programming (MTP)

## Summary:

Programming is headed towards a bigger focus on AI, specifically LLMs and Generative AI, but integrating these tools can be challenging. Meaning-Typed Programming (MTP) seeks to solve this be taking the "semantic richness of code" and giving that directly to the LLM. 

This is accomplished through:
- by operator: Jac-lang operator to indicate desire of MTP 
- MT-IR: Meaning-Typed Itermediate representation generator to give to LLM
- MT-Runtime: Execution Engine to automate LLM interactions under the hood

In other words, MTP offers an abstraction to simplify programatic interactions with LLMs through creating an intemediary representation of the user's code to superceed the need to prompt engineer.

This doesn't replace code generation, instead it focuses on automating "functionality" within the code itself. LLMs should be able to do these tasks because Programs are designed to be readable (from programmer to programmer) AND with recent LLM advancements, they can better figure out developer intention

### Example:

```python
def calc_age(birth: date, today: date) -> int by gpt4()
```

The above is a function that doesn't "generate" code, instead you pass in the necessary context, and the LLM will respond with the result directly through its own internal "calculations"

###

Other attempts have been made to solve this issue (LangChain, Language Model Query Language, DSPy, SGLang, etc.) but they offer "too much" and require more effort for the programmer.

Issues with existing paradigms:
- Prompt Complexity
- Learning Curve
- Input & Output Conversion back to Programming Types can be difficult

Challenges MTP seeks to tackle:
- Abstraction must be simple 
    - by keyword operator
- Guarenteeing LLM is provided necessary context
    - Programs are designed to be readable by humans, and therefore, LLMs. Provide same context a developer would use
- Prompts must be generated at runtime, therefore must be lightweight
    - MT-IR is generated at compile time, MT-Runtime is done in real time to get variable values
- LLMs are non-deterministic, therefore output must be processed carefully
    - Generate a new prompt if object is malformed

Overall the paper discusses that the MTP integration reduces lines of code added to existing code; reduces token count, runtime, and cost; maintained performance even with poorly written code; and programmers seeemed to prefer it over its alternatives (DSPy and LMQL)

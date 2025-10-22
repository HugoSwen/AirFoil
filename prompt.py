Initialize_prompt_A = """
Given one input variable x, please design functions of x.
Provide the Python function code, using numpy (np) if necessary. 
Only output the right-hand side expression of the function (after 'return'), with no code block, no explanations, no imports, and no function definition.
"""
Initialize_prompt_B = Initialize_prompt_A + """
The function must be continuous, differentiable, monotonically increasing and positive for x in the interval [0,1].
When x is 1.0 or 0.0, the corresponding value calculated by the designed function should also be 1.0 or 0.0.
"""

Initialize_prompt_C = Initialize_prompt_B + """
Here are some well-performing functions for reference:
{functions}
"""

Evolve_prompt_1 = """
Given one input variable x, please design functions of x.
I have 2 pairs of functions as follows:
<Function 1> {function_1}
<Function 2> {function_2}
Please help me create a new function that has a totally different form from the given ones.
The function must be continuous, differentiable, monotonically increasing and positive for x in the interval [0,1]. 
When x is 1.0 or 0.0, the corresponding value calculated by the designed function should also be 1.0 or 0.0.

Provide the Python function code, using numpy (np) if necessary. 
Only output the right-hand side expression of the function (after 'return'), with no code block, no explanations, no imports, and no function definition.
"""

Evolve_prompt_2 = """
Given one input variable x, please design functions of x.
I have 2 pairs of functions as follows:
<Function 1> {function_1}
<Function 2> {function_2}
Please help me create a new function that is motivated by the given ones.
The function must be continuous, differentiable, monotonically increasing and positive for x in the interval [0,1]. 
When x is 1.0 or 0.0, the corresponding value calculated by the designed function should also be 1.0 or 0.0.

Provide the Python function code, using numpy (np) if necessary. 
Only output the right-hand side expression of the function (after 'return'), with no code block, no explanations, no imports, and no function definition.
"""

Evolve_prompt_3 = """
Given one input variable x, please design functions of x.
I have one function as follows:
<Function 1> {function}
Please help me create a new function that is a revision of the given one.
The function must be continuous, differentiable, monotonically increasing and positive for x in the interval [0,1]. 
When x is 1.0 or 0.0, the corresponding value calculated by the designed function should also be 1.0 or 0.0.

Provide the Python function code, using numpy (np) if necessary. 
Only output the right-hand side expression of the function (after 'return'), with no code block, no explanations, no imports, and no function definition.
"""

Evolve_prompt_4 = """
Given one input variable x, please design functions of x.
I have one function as follows:
<Function 1> {function}
Please help me create a new function that has different parameter settings of the given one.
The function must be continuous, differentiable, monotonically increasing and positive for x in the interval [0,1]. 
When x is 1.0 or 0.0, the corresponding value calculated by the designed function should also be 1.0 or 0.0.

Provide the Python function code, using numpy (np) if necessary. 
Only output the right-hand side expression of the function (after 'return'), with no code block, no explanations, no imports, and no function definition.
"""
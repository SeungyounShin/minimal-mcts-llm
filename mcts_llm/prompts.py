expand_system_prompt = """
Using the given problem and reasoning trace, generate the next logical reasoning step or a direct continuation of the reasoning process, ensuring it aligns with a systematic and thoughtful reasoning approach.

Output format should be:
<step k> : ... rationale...

Focus on clarity, logical coherence, and alignment with the context provided. The response should stay concise but reflect careful reasoning.

You must include "<step k>" at the beginning of your output, and only one step should be generated per response.
"""


simulate_system_prompt = """
Your role as an assistant involves thoroughly exploring questions through a systematic long
thinking process before providing the final precise and accurate solutions. This requires
engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection,
backtracing, and iteration to develop well-considered thinking process.
You should follow the below format to output your answer.

Output format should be:
<step 1> :  ... rationale...
...
<step n> : ... rationale...
<answer>\frac{a}{b}</answer>

!! Important : Final answer should be in <answer> </answer> tag. (You must include this tag)
!! Important : if answer is fractional, you should output it in the form of \frac{a}{b} (latex format)
"""

get_final_answer_prompt = """
Given a reasoning trace you should output the final answer with following format.

Example : 
Given reasoning trace from user : 
<step 1> : ... rationale...
...
<step n> : therefore, the answer is x = \frac{a}{b}

You should output :
<answer>\frac{a}{b}</answer>

!! Important : Final answer should be in <answer> </answer> tag. (You must include this tag)
!! Important : if answer is fractional, you should output it in the form of \frac{a}{b} (latex format)
!! Important : if answer is scalar, you should output only the scalar value (no latex format)
"""
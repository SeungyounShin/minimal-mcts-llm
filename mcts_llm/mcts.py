import os
import random
import re
from mcts_llm.node import Node
from mcts_llm.prompts import expand_system_prompt, simulate_system_prompt, get_final_answer_prompt
from mcts_llm.utils import snapshot_mcts
from mcts_llm.math_grader import extract_answer, math_equal
from vllm import LLM, SamplingParams
from tqdm import tqdm

class MCTS:
    def __init__(self, task, llm : LLM, max_children : int = 3):
        self.task = task
        self.llm = llm
        self.tokenizer = self.llm.get_tokenizer()
        self.root = Node(content = task['problem'])
        self.max_children = max_children

        # in general 
        self.end_of_turn_token = "<|eot_id|>"

        # for expand
        self.expand_system_prompt = expand_system_prompt
        self.temperature = 1.0
        self.max_tokens = 1024
        self.top_p = 0.9

        # for simulate
        self.simulate_system_prompt = simulate_system_prompt
        self.simulate_temperature = 0.1
        self.get_final_answer_prompt = get_final_answer_prompt

    def run(self):
        for i in tqdm(range(self.task['iteration']), desc="MCTS Iteration"):
            node = self.select()
            expanded_nodes = self.expand(node)
            terminal_node = self.simulate(expanded_nodes)
            node = self.backpropagate(terminal_node) # this node should be the root
            snapshot_mcts(self, f"./visualization/mcts_tree_snapshot_{i}")

    def _get_problem_and_reasoning_trace(self, node):
        problem = self.root.content
        nodes = []
        # going up to root (gather reasoning trace)
        count_step = 0
        while not node.is_root():
            nodes.append(node)
            node = node.parent
            count_step += 1
        reasoning_trace :str = "\n".join([node.content for node in reversed(nodes)])
        return problem, reasoning_trace
    
    def select(self):
        """
        Select the most promising node based on UCB1.
        """
        node = self.root
        while not node.is_leaf() and not node.is_terminal:
            # Select the child with the highest UCB1 score
            candidates = node.get_children_and_not_terminal()
            if len(candidates) == 0 and len(node.children) < self.max_children:
                return node
            node = max(candidates, key=lambda child: child._ucb1())

        if node.is_terminal:
            import pdb; pdb.set_trace()
        return node

    def expand(self, node):
        if len(node.children) >= self.max_children:
            return [node]

        # get (problem, reasoning_trace)
        problem, reasoning_trace = self._get_problem_and_reasoning_trace(node)

        # expand
        num_expand = self.max_children - len(node.children)
        convs = [
            [
                {"role": "system", "content": self.expand_system_prompt},
                {"role": "user", "content": f"problem: {problem}\nreasoning_trace: {reasoning_trace}"},
            ]
            for _ in range(num_expand)
        ]
        templated_convs = self.tokenizer.apply_chat_template(
            convs, tokenize=False, add_generation_prompt=True
        ) 
        templated_convs = [f"{conv}<step" for i, conv in enumerate(templated_convs)] # Force LLM to output <step k> ...
        # generate
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            n=1,  # Since we've already duplicated the prompt_token_ids, we only need to generate 1 completion per prompt
        )
        responses : list = self.llm.generate(
            templated_convs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        output_texts : list[str] = [response.outputs[0].text for response in responses]
        refined_output_texts : list[str] = ["<step" + output_text.split("<step")[0].strip() for output_text in output_texts]

        # attach as children
        for refined_output_text in refined_output_texts:
            child = Node(content = refined_output_text, parent = node)
            node.children.append(child)
        return node.children

    def simulate(self, expanded_nodes):
        expanded_nodes = [node for node in expanded_nodes if not node.is_terminal]
        explore_node = random.choice(expanded_nodes)
    
        problem, reasoning_trace = self._get_problem_and_reasoning_trace(explore_node)
        convs = [
            [
                {"role": "system", "content": self.simulate_system_prompt},
                {"role": "user", "content": f"{problem}"},
                {"role": "assistant", "content": f"{reasoning_trace}"},
            ]
        ]
        templated_convs = self.tokenizer.apply_chat_template(
            convs, tokenize=False, add_generation_prompt=False
        ) 
        templated_convs = [conv.rstrip(self.end_of_turn_token) if conv.endswith(self.end_of_turn_token) else conv for conv in templated_convs] # delete end_of_turn at the end of string
        templated_convs = [f"{conv}\n<step" for i, conv in enumerate(templated_convs)] # Force LLM to output <step k> ...
        sampling_params = SamplingParams(
            temperature=self.simulate_temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            n=1,  
        )
        responses : list = self.llm.generate(
            templated_convs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        output_texts : list[str] = [response.outputs[0].text for response in responses]
        completion = [f"{reasoning_trace}<step{o}" for o in output_texts]
        answers = [re.search(r'<answer>(.*?)</answer>', text).group(1) if re.search(r'<answer>(.*?)</answer>', text) else None for text in output_texts]
        completion_without_answer = [completion[i].split("<answer>")[0] for i, answer in enumerate(answers) if answer is None]

        rest_reasoning_trace: list[list[str]] = [
            [f"<step{trace}" for trace in output_text.split("<step")] for output_text in output_texts
        ]
        
        # dont need to get final answer
        if len(completion_without_answer) == 0:
            final_answer_output_texts : list[str] = [f"<answer>{answer}</answer>" for answer in answers]

        else:
            # get final answer
            _convs = [
                [
                    {"role": "system", "content": self.get_final_answer_prompt},
                    {"role": "user", "content": f"{problem}"},
                    {"role": "assistant", "content": f"{c}"},
                ] for c in completion_without_answer
            ]
            _templated_convs = self.tokenizer.apply_chat_template(
                _convs, tokenize=False, add_generation_prompt=False
            ) 
            _templated_convs = [conv.rstrip(self.end_of_turn_token) if conv.endswith(self.end_of_turn_token) else conv for conv in _templated_convs] # delete end_of_turn at the end of string
            _templated_convs = [f"{conv}\n<answer>" for i, conv in enumerate(_templated_convs)] # Force LLM to output <step k> ...
            # generate
            final_answer_responses : list = self.llm.generate(
                _templated_convs,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            final_answer_output_texts : list[str] = [f"<answer>{response.outputs[0].text}" for response in final_answer_responses]
            # rest_reasoning_trace , final_answer_output_texts

        # add simulation as children
        simulation = rest_reasoning_trace[0] + final_answer_output_texts
        # append to explore_node
        node = explore_node
        for i, step in enumerate(simulation):
            is_terminal = (step in final_answer_output_texts)
            child = Node(content = step, parent = node, is_terminal = is_terminal)
            node.children.append(child)
            node = child
        
        return node

    def backpropagate(self, terminal_node):
        correct = False
        # verify terminal_node is a leaf
        if not terminal_node.is_leaf():
            raise Exception("terminal_node is not a leaf")

        final_answer = terminal_node.content.split("<answer>")[1].split("</answer>")[0].strip()    
        is_correct = (
            extract_answer(final_answer, "math", use_last_number=False) == self.task['answer'] or 
            final_answer == self.task['answer'] or 
            math_equal(final_answer, self.task['answer'], timeout=True)
        )
        
        RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
        if is_correct:
            correct = True
            print(f"{GREEN}correct answer: {final_answer} {RESET}")
        else:
            correct = False
            print(f"{RED}wrong answer: {final_answer} the correct answer is {self.task['answer']} {RESET}")

        node = terminal_node
        while not node.is_root():
            node.visit_count += 1
            node.total_reward += 1 if correct else 0
            node.update_value()
            node = node.parent

        return node

if __name__ == "__main__":
    import os 
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

    # problem = {"problem": "Allen and Bethany each arrive at a party at a random time between 1:00 and 2:00.  Each stays for 15 minutes, then leaves.  What is the probability that Allen and Bethany see each other at the party?", "solution": "We let the $x$ axis represent the time Allen arrives, and the $y$ axis represent the time Bethany arrives.\n\n[asy]\ndraw((0,0)--(60,0), Arrow);\ndraw((0,0)--(0,60), Arrow);\nlabel(\"1:00\", (0,0), SW);\nlabel(\"1:15\", (0,15), W);\nlabel(\"1:45\", (60,45), E);\nlabel(\"1:15\", (15,0), S);\nlabel(\"2:00\", (60,0), S);\nlabel(\"2:00\", (0,60), W);\nfill((0,0)--(60,60)--(60,45)--(15,0)--cycle, gray(.7));\nfill((0,0)--(60,60)--(45,60)--(0,15)--cycle, gray(.7));\n[/asy]\n\nThe shaded region represents the times that Allen and Bethany would see each other at the party.  For example, if Allen arrived at 1:30, Bethany could arrive at any time between 1:15 and 1:45 and see Allen at the party.  Let one hour equal one unit.  Then, we can calculate the area of the shaded region as the area of the entire square minus the areas of the two unshaded triangles.  This will be equal to $2\\cdot \\frac{1}{2} \\cdot \\frac{3}{4} \\cdot \\frac{3}{4}=\\frac{9}{16}$.  So, the area of the shaded region is $1-\\frac{9}{16}=\\boxed{\\frac{7}{16}}$.  Since the area of the square is 1, this is the probability that Allen and Bethany see each other at the party.", "answer": "\\frac{7}{16}", "subject": "Counting & Probability", "level": 5, "unique_id": "train/counting_and_probability/314.json"}
    # problem = {"problem": "When the base-12 integer $1531_{12}$ is divided by $8$, what is the remainder?", "solution": "We have $$1531_{12} = 12^3 + 5\\cdot 12^2 + 3\\cdot 12 + 1.$$Note that $12^2$ is divisible by $8$, so $$1531_{12} = (\\text{a multiple of 8}) + 3\\cdot 12 + 1.$$Therefore, the remainder upon dividing $1531_{12}$ by $8$ is the same as the remainder upon dividing $3\\cdot 12+1$ by $8$. This remainder is $\\boxed{5}$.", "answer": "5", "subject": "Number Theory", "level": 3, "unique_id": "train/number_theory/93.json"}
    problem = {"problem": "At a certain university, the division of mathematical sciences consists of the departments of mathematics, statistics, and computer science. There are two male and two female professors in each department. A committee of six professors is to contain three men and three women and must also contain two professors from each of the three departments. Find the number of possible committees that can be formed subject to these requirements.", "solution": "There are two cases:\nCase 1: One man and one woman is chosen from each department.\nCase 2: Two men are chosen from one department, two women are chosen from another department, and one man and one woman are chosen from the third department.\nFor the first case, in each department there are ${{2}\\choose{1}} \\times {{2}\\choose{1}} = 4$ ways to choose one man and one woman. Thus there are $4^3 = 64$ total possibilities conforming to case 1.\nFor the second case, there is only ${{2}\\choose{2}} = 1$ way to choose two professors of the same gender from a department, and again there are $4$ ways to choose one man and one woman. Thus there are $1 \\cdot 1 \\cdot 4 = 4$ ways to choose two men from one department, two women from another department, and one man and one woman from the third department. However, there are $3! = 6$ different department orders, so the total number of possibilities conforming to case 2 is $4 \\cdot 6 = 24$.\nSumming these two values yields the final answer: $64 + 24 = \\boxed{88}$.", "answer": "88", "subject": "Counting & Probability", "level": 5, "unique_id": "train/counting_and_probability/5093.json"}
    # problem = {"problem": "When the digits in the number $2005$ are reversed we obtain the number $5002,$ and $5002 = a \\cdot b \\cdot c$, such that $a$, $b$ and $c$ are three distinct primes. How many other positive integers are the products of exactly three distinct primes $p_1$, $p_2$ and $p_3$ such that $p_1 + p_2 + p_3 = a+b+c$?", "solution": "5002 factors to $2 \\cdot 41 \\cdot 61$, which sums to 104.  Since 2 is the only even prime number, and we need the sum of these 3 distinct primes to be even, 2 must be one of these primes, meaning we need to look at pairs of primes that sum to 102.  We start with 3, subtract that from 102, and see if the resulting number is prime.  We need check only primes up to 51 in this manner because if the prime is greater than 51, its corresponding prime would be less than 51, meaning we would have found the pair already.  In this manner, we find the following 7 different pairs: $(5,97);(13,89);(19,83);(23,79);(29,73);(31,71);(43,59)$, and thus, there are $\\boxed{7 \\text{ distinct integers}}$.", "answer": "7 \\text{ distinct integers}", "subject": "Number Theory", "level": 5, "unique_id": "train/number_theory/336.json"}

    mcts_task = {
        'problem' : problem['problem'],
        'answer' : problem['answer'], # extract_answer(problem['solution'], "math", use_last_number=False),
        'iteration' : 8
    }

    llm = LLM(
        model='meta-llama/Llama-3.1-8B-Instruct',
        gpu_memory_utilization=0.8,
        enable_prefix_caching=True,
        # seed=42,
        tensor_parallel_size=4,
    )

    mcts = MCTS(
        task=mcts_task,
        llm=llm
    )
    
    mcts.run()
from typing import List, Union
from utils_code import refine_text
from execute import check_correctness
from sanitize import sanitize
from datasets import load_dataset
import datasets

import signal
import time

def _timeout_handler(signum, frame):
    raise TimeoutError("Execution took too long")
PYTHON_STOP = [ "\nif __name__",
                "\ndef main(",
                "\nprint("
                ]
    
PYTHON_IMPORTS = [  "import math",
                    "import re",
                    "import sys",
                    "import copy",
                    "import datetime",
                    "import itertools",
                    "import collections",
                    "import heapq",
                    "import functools",
                    "import hashlib",
                    "import numpy",
                    "import numpy as np",
                    "import string",
                    "from typing import *",
                    "from collections import *"
                    ]
class rm_code():
    def __init__(self):
        """
        use a dict to store the data, where the key is the 'seq_id'

        """
        dataset = load_dataset("OpenCoder-LLM/opc-sft-stage2","educational_instruct")["train"]
        dataset = dataset.select(range(4000))
        self.task_data = {data['seq_id']: data for data in dataset}
        self.imports_code = PYTHON_IMPORTS
    
    def format_prompt(self,
                      problem: str,
                      tests: Union[List[str],str],
                      code: str = None
                    ) -> str:
        problem = f"You are an expert Python programmer, and here is your task:\n{problem}"
        if isinstance(tests, List):
            test = "\n".join(tests)
        else:
            test = tests
        test = f"Your code should pass these tests:\n{test}\n"
        prompt = problem + test
        if code:
            code = refine_text(code)
            code = f"\n```python\n{code}\n```\n"
            prompt = prompt + code
        else:
            prompt = prompt + "\n```python\n"
        return prompt
    
    def get_prompt(self,task_id):
        prompt = self.format_prompt(self.task_data[task_id]['instruction'],self.task_data[task_id]['testcase'][0])
        return {
                    'task_id': task_id,
                    'prompt': prompt
                }
    def get_binary_reward(self,task_id, solution):
        """
        return a binary feedback (int type)
        """
        signal.signal(signal.SIGALRM, _timeout_handler)
        # 60-second alarm
        signal.alarm(60)

        try:
            # 1) Prepare the code
            solution = sanitize(solution, self.task_data[task_id]["entry_point"])
            code = (
                "\n".join(self.imports_code) + "\n"
                + solution + "\n"
                + "\n".join(self.task_data[task_id]['testcase'])
            )
            # 2) Run the checker
            feedback = check_correctness(
                task_id=task_id,
                completion_id=0,
                solution=code,
                time_out=999  # you can still keep an internal time_out if needed
            )
            return int(feedback["passed"])

        except TimeoutError:
            # If we hit the signal alarm, return 0
            return 0

        # solution = sanitize(solution,self.task_data[task_id]["entry_point"] )
        # code =  (
        #             "\n".join(self.imports_code)  + "\n"
        #             + solution + "\n"
        #             + "\n".join(self.task_data[task_id]['testcase'])
        #         )
        # feedback = check_correctness(task_id = task_id, completion_id=0,    solution=code,time_out=999)
        # return int(feedback["passed"])

if __name__ == "__main__":
    RM = rm_code()
    id = 660113403
    print(RM.get_prompt(id))
    #print(print(RM.get_prompts()['prompt'][0]))
    print(RM.get_binary_reward(id,"def is_palindrome(s):\nreturn"))
    print(RM.get_binary_reward(id,RM.task_data[id]["code"]))

"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    def __init__(self, verbose: bool = True):
        self._verbose = verbose
        # data_point["input"], data_point["output"]
        if self._verbose:
            print(
                f"Using prompt template PET/CT."
            )

    def generate_prompt(
        self,
        instruction: Union[None, str] = None,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        
        prefix = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request." 
        if label:
            res = prefix + f"\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{label}"
        else:
            res = prefix + f"\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}" # \n\n### Response:\n 

        if self._verbose:
            print(res)

        return res

    def get_response(self, output: str) -> str:
        return output.split("### Response:")[-1].strip()

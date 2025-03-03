import json
import jsonlines
import json_repair
import pandas as pd
import numpy as np
import torch
import shutil
import glob
import re

from typing import List
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset
from config.prompt import prompt_lbv1_cot,prompt_lbv1_nocot

def construct_cot_nocot_split(split: str):
    data_list=load_dataset('THUDM/LongBench',split, split='test')
    new_cot_data_list = []
    new_nocot_data_list = []
    for data in  data_list:
        context = data["context"]
        question = data["input"]
        answers = data["answers"]
        all_classes=data['all_classes']
        id = data["_id"]
        output = answers[0]
        instruction_cot = prompt_lbv1_cot.format(context=context, question=question)
        instruction_nocot = prompt_lbv1_nocot.format(context=context, question=question)
        new_cot_data_list.append({"id": id, "question":question,"instruction": instruction_cot, "answers": answers,"all_classes":all_classes ,"output": output, "system": "You are a helpful assistant."})
        new_nocot_data_list.append({"id": id,"question":question,"instruction": instruction_nocot, "answers": answers, "all_classes":all_classes,"output": output, "system": "You are a helpful assistant."})
    print(f"size of {split} new_cot_data_list: {len(new_cot_data_list)}")
    print(f"size of {split} new_nocot_data_list: {len(new_nocot_data_list)}")
    with jsonlines.open(f"/dataset/longbenchv1/{split}_cot.jsonl", 'w') as writer:
        writer.write_all(new_cot_data_list)
    with jsonlines.open(f"dataset/longbenchv1/{split}_nocot.jsonl", 'w') as writer:
        writer.write_all(new_nocot_data_list)
    print(f"Finished writing {split} dataset")
    return



from datasets import load_dataset
datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa","musique"]
for dataset in datasets:
    construct_cot_nocot_split(dataset)

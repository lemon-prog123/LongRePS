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
from config.prompt import prompt_lbv2_cot,prompt_lbv2_nocot

def construct_cot_nocot_split(filtered_data,split="MQA"):
    new_cot_data_list = []
    new_nocot_data_list = []
    for item in filtered_data:
        context = item["context"]
        question=item['question']
        _id=item['_id']
        difficulty=item['difficulty']
        instruction_cot=prompt_lbv2_cot.format(context=context, question=question,choice_A=item['choice_A'],choice_B=item['choice_B'],choice_C=item['choice_C'],choice_D=item['choice_D'])
        instruction_nocot=prompt_lbv2_nocot.format(context=context, question=question,choice_A=item['choice_A'],choice_B=item['choice_B'],choice_C=item['choice_C'],choice_D=item['choice_D'])
        new_cot_data_list.append({"id": id, "instruction": instruction_cot, "output": item['answer'], "id":_id,"difficulty":difficulty,"question":item['question'],"num_tokens":item['token_num'],"system": "You are a helpful assistant."})
        new_nocot_data_list.append({"id": id, "instruction": instruction_nocot, "output": item['answer'], "id":_id,"difficulty":difficulty,"question":item['question'],"num_tokens":item['token_num'],"system": "You are a helpful assistant."})

    print(f"size of new_cot_data_list: {len(new_cot_data_list)}")
    print(f"size of new_nocot_data_list: {len(new_nocot_data_list)}")
    with jsonlines.open(f"dataset/longbenchv2/{split}_cot.jsonl", 'w') as writer:
        writer.write_all(new_cot_data_list)
    with jsonlines.open(f"dataset/longbenchv2/{split}_nocot.jsonl", 'w') as writer:
        writer.write_all(new_nocot_data_list)



split_list=["Single-Document QA","Multi-Document QA"]
split_tag=["SQA","MQA"]
dataset=load_dataset('Lemon123prog/Longmix-LongRePS',split='test')

for (split,tag) in zip(split_list,split_tag):
    filter_data=[]
    for data in dataset:
        if data['domain']==split and data['token_num']<105*1024: #In case of the Bug for over-long data
            filter_data.append(data)
    construct_cot_nocot_split(filter_data,split=tag)


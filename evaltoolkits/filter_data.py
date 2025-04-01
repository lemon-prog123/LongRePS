import json
import jsonlines
import json_repair
import pandas as pd
import numpy as np
import torch
import shutil
import glob
import re
import string
import time
import os
import argparse
from typing import List, Tuple
from tqdm import tqdm
from pathlib import Path
from json_repair import repair_json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from openai import OpenAI

from utils import normalize_answer, preprocess_pred_for_json_repair, extract_fact_list, verify_fact_list, extract_number
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_src_file', type=str, default=None)
    parser.add_argument('--path_to_stage1_file', type=str, default=None)
    parser.add_argument('--sample_num', type=int, default=30)
    return parser.parse_args(args)
# 我们需要你帮忙评价模型推理过程的质量。模型的接收的输入是一段长文本，以及一个复杂的问题，它的任务是根据问题的需要，从长文本中检索出相关信息（以[Excerpt xxx]的形式开头，包含在``中），并给出正确的答案。现在，我们已经在上面给出了问题和模型的推理过程。模型最终得到的结果是正确的，但是我们需要你来评价模型的推理过程是否合理。请你根据以下几个方面来评价模型的推理过程：- 逻辑性：模型对问题的拆解应当合理。推理过程对于检索到的信息的使用应该符合逻辑，根据检索到的信息得出答案的逻辑链条应该合理。 - 完整性：推理过程应该主要使用从文中检索到的信息，即[Excerpts xxx]后内容，而非模型自身的知识。 - 简洁性：只应当检索回答问题相关的信息，不应罗列过多无关的信息。

EVAL_PROMPT = '''[Question]
{question}

[The Start of Assistant's Reasoning Path]
{reasoning}
[The End of Assistant's Reasoning Path]

[System]
We would like to request your feedback on the quality of the reasoning process in the given response. 
The model receives a long text input and a complex question. Its task is to retrieve relevant information from the long text (marked as [Excerpt xxx] and enclosed in ``) based on the question's requirements and provide the correct answer. Above, we have provided both the question and the model's reasoning process. While the model's final answer is correct, we need you to evaluate whether its reasoning process is sound.

Please assess the model's reasoning process based on the following aspects:

1. Logical Coherence:
- The model should break down the question appropriately
- The use of retrieved information should follow logical patterns
- The chain of reasoning from retrieved information to the final answer should be sound

2. Completeness:
- The reasoning process should primarily rely on information retrieved from the text ([Excerpts xxx])
- The model should not heavily depend on its own knowledge base

3. Conciseness:
- Only information relevant to answering the question should be retrieved
- The model should avoid listing excessive or irrelevant information

Please rate whether this reasoning path is suitable for the question. The assistant receives an overall score on a scale of 1 to 100, where a higher score indicates better overall performance.
Please note that if the assistant's reasoning process fully meets the above criteria, its overall rating should be full marks (100).
Please first provide a comprehensive explanation of your evaluation, avoiding any potential bias.
Then, output a line indicating the score of the Assistant.

PLEASE OUTPUT WITH THE FOLLOWING FORMAT, WHERE THE SCORE IS ON A SCALE OF 1 TO 100 BY STRICTLY FOLLOWING THIS FORMAT: "[[score]]", FOR EXAMPLE "Rating: [[100]]":
<start output>
Evaluation evidence: your evaluation explanation here, no more than 100 words
Rating: [[score]]
<end output>

Now, start your evaluation:'''

def process_single_data(example):
    """处理单条数据的函数"""
    checked_data = {**example}
    checked_data['new_pred'] = []
    checked_data['new_f1_score_list'] = []
    checked_data['new_extracted_pred_list'] = []
    fact_valid_flag = 0
    json_valid_flag = 0
    total_num=len(example['pred'])
    save_num=0
    for pred_idx, pred in enumerate(example['pred']):
        try:
            pred = preprocess_pred_for_json_repair(pred)
            content = json_repair.loads(pred)
            if len(content) >1:
                content=content[0]
                pred=json.dumps(content)
            if not isinstance(content, dict) or not content or 'reasoning' not in content or len(content) > 2 or type(content['reasoning']) != str:
                continue
            else:
                json_valid_flag = 1
            
            fact_list = extract_fact_list(content['reasoning'])
            if len(fact_list) >= 0:
                if verify_fact_list(fact_list, example['instruction']):
                    save_num+=1
                    fact_valid_flag = 1
            checked_data['new_pred'].append(pred)
            checked_data['new_f1_score_list'].append(example['f1_score_list'][pred_idx])
            checked_data['new_extracted_pred_list'].append(example['extracted_pred_list'][pred_idx])
        except:
            continue
    
    checked_data['fact_valid'] = fact_valid_flag
    checked_data['json_valid'] = json_valid_flag
    checked_data['save_rate']=save_num/total_num
    return checked_data

def filter_stage_1(path_to_src_file: str, path_to_stage1_file: str, f1_score_thresh: float = 1.0, sample: int =10):

    with jsonlines.open(path_to_src_file) as fin:
        data_list = list(fin)
    print("Sample Number ",sample)
    for data in data_list:
        data['pred']=data['pred'][:sample]
        data['f1_score_list']=data['f1_score_list'][:sample]
        data['extracted_pred_list']=data['extracted_pred_list'][:sample]
    
    dataset = Dataset.from_list(data_list)

    # 并行处理数据
    processed_dataset = dataset.map(
        process_single_data,
        num_proc=16,
        desc="Processing data"
    )

    # 统计结果
    no_valid_fact_cnt = sum(1 for x in processed_dataset if not x['fact_valid'])
    no_valid_json_cnt = sum(1 for x in processed_dataset if not x['json_valid'])
    avg_save_rate = np.mean([x['save_rate'] for x in processed_dataset])
    processed_dataset = processed_dataset.filter(lambda x: len(x['new_pred']) > 0)

    print(f"Avg Save Rate: {avg_save_rate}")
    print(f"No valid fact count: {no_valid_fact_cnt}")
    print(f"No valid JSON count: {no_valid_json_cnt}")
    print(f"Checked data count: {len(processed_dataset)}")

    # 处理最终结果
    result_data_list = []

    for item in processed_dataset:
        if not item['new_f1_score_list']:
            continue
        max_value = max(item['new_f1_score_list'])
        # max_index = item['new_f1_score_list'].index(max_value)
        # score = item['new_f1_score_list'][max_index]
        
        if max_value >= f1_score_thresh:

            remaining_idx_list = [idx for idx, score in enumerate(item['new_f1_score_list']) if score >= f1_score_thresh][:10]
            remaining_pred_list = [item['new_pred'][idx] for idx in remaining_idx_list][:10]
            remaining_f1_score_list = [item['new_f1_score_list'][idx] for idx in remaining_idx_list][:10]
            remaining_extracted_pred_list = [item['new_extracted_pred_list'][idx] for idx in remaining_idx_list][:10]
            output = remaining_pred_list[0]
            st_output = item['answers']
            f1_score = remaining_f1_score_list[0]

            result_data_list.append({
                "instruction": item['instruction'],
                "filtered_pred": remaining_pred_list,
                "filtered_f1_score_list": remaining_f1_score_list,
                "filtered_extracted_pred_list": remaining_extracted_pred_list,
                "answers": item['answers'],
                "output": output,
                "st_output": st_output,
                "f1_score": f1_score,
                "system": "You are a helpful assistant.",
                "id": item['id'],
            })

    print(f"selected_cnt: {len(result_data_list)}")

    # 保存结果
    with jsonlines.open(path_to_stage1_file, 'w') as writer:
        writer.write_all(result_data_list)

    return


def get_score_evidence(question:str, pred:str,ground_truth:str) -> Tuple[float, str]:
    prompt = EVAL_PROMPT.format(question=question, reasoning=pred)
    max_retries = 5
    API_KEY = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=API_KEY,base_url="")
    for _ in range(max_retries):
        content = ""
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user","content": prompt}],
                model="gpt-4o-mini",
                temperature=0,
                max_tokens=512,
            )
            content = response.choices[0].message.content
            if content == None:
                continue
            evidence = content
            rating = extract_number(content)
            return (rating, evidence)
        except Exception as e:
            content = "" if content == None else content
            print(e, content)
            time.sleep(30)
            return (0.0, content)

    return (0.0, "")


def get_score_evidence_list(question:str, predictions:List[str], answer:str) -> Tuple[List[float], List[str]]:
    score_list: List[float] = []
    evidence_list: List[str] = []
    for pred in predictions:
        # if there are more than 3 "100" in score_list, break
        if score_list.count(100) >= 3:
            break
        score, evidence = get_score_evidence(question=question, pred=pred, ground_truth=answer)
        score_list.append(float(score))
        evidence_list.append(evidence)
    return score_list, evidence_list


def get_llm_score_for_single_data(data):
    answer = data["answers"][0]
    question = data["instruction"].split("Question:")[-1].replace("\n\nAnswer:","").replace("*","").strip().replace("\n\nResponse:","").strip()
    reasoning_list = [json_repair.loads(pred)['reasoning'] for pred in data["filtered_pred"]]
    score_list, evidence_list = get_score_evidence_list(question, reasoning_list, answer)

    data["llm_score_list"] = score_list
    data["llm_evidence_list"] = evidence_list
    return data


def filter_stage_2(path_to_stage1_file: str, path_to_stage2_file: str, path_to_stage3_file: str, llm_score_thresh: float = 100):
    
    with jsonlines.open(path_to_stage1_file) as fin:
        data_list = list(fin)
    dataset = Dataset.from_list(data_list)

    preprocessed_dataset = dataset.map(
        get_llm_score_for_single_data,
        num_proc=8,
        desc="Processing data"
    )
    preprocessed_dataset = preprocessed_dataset.to_list()
    score_list=[]
    for data in preprocessed_dataset:
        score_list.append(np.mean(data['llm_score_list']))
        max_score = max(data["llm_score_list"])
        max_index = data["llm_score_list"].index(max_score)
        data["output"] = data["filtered_pred"][max_index]
        data["st_output"] = data["answers"]
        data["llm_score"] = max_score
    print(f'Avg Score in Stage2 is {np.mean(score_list)}')

    with jsonlines.open(path_to_stage2_file, 'w') as writer:
       writer.write_all(preprocessed_dataset)

    
    stage3_data_list = [data for data in preprocessed_dataset if data['llm_score'] >= llm_score_thresh]
    print(f"Stage3 selected_cnt: {len(stage3_data_list)}, avg score in stage3 is {np.mean([data['llm_score'] for data in stage3_data_list])}")

    with jsonlines.open(path_to_stage3_file, 'w') as writer:
        writer.write_all(stage3_data_list)
    
    return

def filter_stage_3(path_to_stage2_file: str, path_to_stage3_file: str, llm_score_thresh: float = 100):
    with jsonlines.open(path_to_stage2_file) as fin:
          preprocessed_dataset= list(fin)

    stage3_data_list = [data for data in preprocessed_dataset if data['llm_score'] >= llm_score_thresh]
    print(f"Stage3 selected_cnt: {len(stage3_data_list)}, avg score in stage3 is {np.mean([data['llm_score'] for data in stage3_data_list])}")

    with jsonlines.open(path_to_stage3_file, 'w') as writer:
        writer.write_all(stage3_data_list)
    
    return

args = parse_args()
path_to_src_file = args.path_to_src_file
path_to_stage1_file = args.path_to_stage1_file
sample=args.sample_num
path_to_stage2_file = path_to_stage1_file.replace("stage1", "stage2")
path_to_stage3_file = path_to_stage1_file.replace("stage1", "stage3")
f1_score_thresh = 1.0
llm_score_thresh = 0

assert "thresh" + str(f1_score_thresh) in path_to_stage1_file, "f1_score_thresh is not consistent with the one used in stage1"

filter_stage_1(path_to_src_file, path_to_stage1_file, f1_score_thresh,sample)
filter_stage_2(path_to_stage1_file, path_to_stage2_file, path_to_stage3_file, llm_score_thresh)

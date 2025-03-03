import re
import json
import string
import json_repair

from pathlib import Path
from typing import Union, List

# from metrics import normalize_answer
def extract_number(text):
    match = re.search(r'\[\[([0-9]*\.?[0-9]+)\]\]', text)
    if match:
        return float(match.group(1))
    match = re.search(r'\[([0-9]*\.?[0-9]+)\]', text)
    if match:
        return float(match.group(1))
    return 0.0

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def load_jsonl_file(path_to_file: Union[str, Path]):
    data_list = []
    error_cnt = 0
    with open(path_to_file) as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line)
                data_list.append(data)
            except Exception as e:
                error_cnt += 1
                print(f"Failed loading line {idx}, error: {e}")
                print(line)
    print(f"Failed loading {error_cnt} lines, total {len(data_list)} lines loaded")
    return data_list

def preprocess_pred_for_json_repair(pred: str):
    escaped_str = re.sub(
        r'(?<="reasoning": ")(.*?)(?="\s*,\s*\n\s*"answer":)',
        lambda match: re.sub(r'(?<!\\)"', r'\\"', match.group(0)),
        pred,
        flags=re.DOTALL
    )
    return escaped_str

def extract_fact_list(target_str: str):
    pattern = r'\[Excerpt \d+\] `([^`]*)`'
    fact_list = re.findall(pattern, target_str)
    return fact_list

def verify_fact_list(fact_list: List[str], instruction: str) -> bool:
    # 去除 instruction 中的标点符号
    instruction_cleaned = normalize_answer(instruction)
    
    for fact in fact_list:
        # 去除 fact 中的标点符号
        fact_cleaned = normalize_answer(fact)
        # 比较去除标点符号后的 fact 和 instruction
        if fact_cleaned not in instruction_cleaned:
            # print(fact)
            return False
    return True

def check_pred_fact_consistency(pred: str, instruction: str):

    processed_pred = preprocess_pred_for_json_repair(pred)
    content = json_repair.loads(processed_pred)
    if not isinstance(content, dict) or not content or 'reasoning' not in content or len(content) > 2 or type(content['reasoning']) != str:
        return False
    fact_list = extract_fact_list(content['reasoning'])
    if len(fact_list) > 0 and verify_fact_list(fact_list, instruction):
        return True
    return False
import json
import json_repair
import argparse
import numpy as np
import jsonlines

from pathlib import Path
from typing import List
from tqdm import tqdm

from utils import load_jsonl_file
from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
    qa_recall_score,
    babiq3_score,
    ruler_score,
    babi_score,
    accuracy_score
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "Ruler": ruler_score,
    "MQA-Medium": accuracy_score,
    "MQA-Medium-v2": accuracy_score,
    "babiq3": qa_f1_score,
    "Babi": babi_score,
    "Babiq3": babiq3_score,
    "MQA": accuracy_score,
    "ICL": accuracy_score,
    "LIL": accuracy_score,
    "LSDU": accuracy_score,
    "NIAH": accuracy_score,
    "BABILong": accuracy_score,
    "LHU": accuracy_score,
    "CRU": accuracy_score,
    "SQA": accuracy_score,
    "SQA-Medium-v2": accuracy_score,
    "multifieldqa": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "gov": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "multi": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_src_file', type=str, default=None)
    return parser.parse_args(args)

def get_score_list(dataset:str, predictions:List[str], answers:List[str],**kwargs) -> List[float]:
    score_list: List[float] = []
    for pred in predictions:
        if dataset=="Ruler":
            score=1
        else:
            score=0
        for answer in answers:
            if dataset=="Ruler":
                score = min(dataset2metric[dataset](pred, answer,**kwargs),score)
            else:
                score = max(dataset2metric[dataset](pred, answer,**kwargs),score)
        score_list.append(score)
    return score_list

def main():
    args = parse_args()
    data_list = load_jsonl_file(args.path_to_src_file)
    best_score_list = []
    file_name = Path(args.path_to_src_file).name
    dataset = file_name.split('.')[0].split("-")[0]
    print(f"Eval {dataset}")
    for data in tqdm(data_list, desc="Calculating F1 score"):
        extracted_pred_list:List[str] = data["extracted_pred_list"]
        answers = data["answers"]
        if type(answers) !=type(["!"]):
            answers=[answers]
        if "all_classes" in data.keys():
            all_classes=data['all_classes']
        else:
            all_classes=[]
        score_list = get_score_list(dataset, extracted_pred_list, answers,all_classes=all_classes)
        best_score_in_this_data = max(score_list)
        best_score_list.append(best_score_in_this_data)
        data["f1_score_list"] = score_list
    final_score = np.mean(best_score_list) # *100 and round to 2 decimal places
    final_score = round(final_score*100, 2)
    print(f"Final score: {final_score}")
    with jsonlines.open(args.path_to_src_file, mode='w') as writer:
        writer.write_all(data_list)
    data_list_noinstr = [{k:v for k,v in data.items() if k!="instruction"} for data in data_list]
    with jsonlines.open(args.path_to_src_file.replace(".jsonl","_noinstr.jsonl"), mode='w') as writer:
        writer.write_all(data_list_noinstr)

    # add to result.json, overwrite the value if it already exists
    # check if result.json exists
    result_path = Path(args.path_to_src_file).parent / "result.json"
    if result_path.exists():
        with open(result_path, 'r') as f:
            result = json.load(f)
    else:
        result = {}
    result[file_name] = final_score
    with open(result_path, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"Result saved in {result_path}")

if __name__ == '__main__':
    main()
    


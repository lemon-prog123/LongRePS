import re
import json
import json_repair
import argparse
import jsonlines

from typing import List
from tqdm import tqdm
from utils import load_jsonl_file

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_src_file', type=str, default=None)
    return parser.parse_args(args)

def extract_pred_list(raw_pred_list:List[str]):
    extracted_pred_list = []
    for pred in raw_pred_list:
        if pred.startswith("```json"):
            # remove the begining and ending ```json
            pred = pred.replace("```json", "").replace("```", "")
            pred = pred.strip()
        if pred.startswith("{"):
            pred = pred.strip()
            try:
                content = json_repair.loads(pred)
                if type(content)==list:
                    content=content[0]
                content = content["answer"]
                extracted_pred_list.append(str(content))
            except Exception as e:
                # print(e, pred)
                # try to extract the answer from the raw pred, if failed, append the raw pred
                # use re to extract the content after "answer: " and before "." (inclusive)
                try:
                    #content =re.findall(r'"answer": (.+?)(?=\n|$)', pred)[0].strip()
                    #print(content)
                    #content = content.strip('\'"[]')
                    pattern = r'"answer": "([^"]+)"'
                    match = re.search(pattern, pred)
                    content = match.group(1)
                    #print(content)
                    # print(f"Extracted re: {content}")
                    extracted_pred_list.append(content)
                except Exception as e2:
                    extracted_pred_list.append(pred)
        else:
            # extract plain text format response
            # print("extracting plain text format response")
            '''
            try:
                content = pred.split("Answer:")[1].split("Reasoning:")[0]
            except:
                try:
                    content = pred.split("Answer:")[1]
                except:
                    content = pred
                    try:
                        content = pred.split("Reasoning:")[0]
                    except:
                        content = pred
            '''
            try:
                content=pred.split("Answer:")[1]
                #content=pred.split("Reasoning:")[1]
            except:
                try:
                    content=pred.split("Reasoning:")[1]
                except:
                    try:
                        #print("Pred:",pred)
                        content=pred.split('\n')[0]
                        #print("Content:",content)
                    except:
                        content=pred
            try:
                content=content.split("\n")[0]
            except:
                content=content
            extracted_pred_list.append(content)
    return extracted_pred_list


def main():
    args = parse_args()
    data_list = load_jsonl_file(args.path_to_src_file)
    for data in tqdm(data_list, desc="Extracting preds"):
        extracted_pred_list = extract_pred_list(data["pred"])
        data["extracted_pred_list"] = extracted_pred_list
    path_to_tgt_file = args.path_to_src_file.replace(".jsonl", "_eval.jsonl")
    with jsonlines.open(path_to_tgt_file, mode='w') as writer:
        writer.write_all(data_list)

if __name__ == "__main__":
    main()
    


    
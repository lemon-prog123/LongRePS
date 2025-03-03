import os
import torch
import json
import argparse
import torch.distributed as dist
import numpy as np
import random
import torch.multiprocessing as mp
import time

from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
from pydantic import BaseModel
from typing import List

from utils import load_jsonl_file, check_pred_fact_consistency

class Response(BaseModel):
    reasoning: str
    answer: str


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--test', action='store_true', help="Evaluate on test mode")
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--sample_num', type=int, default=30)
    parser.add_argument('--max_gen', type=int, default=512)
    parser.add_argument('--gpt', action='store_true', help="Evaluate on test mode")
    parser.add_argument('--temperature', type=float, default=0)
    return parser.parse_args(args)

def get_api_results(model_name, prompt, gpu_id, sample_num, max_gen, temp,gpt=False):
    max_retries = 5
    response_list = []
    # json_schema = Response.model_json_schema() #TODO
    for i in range(max_retries):
        if gpt:
            api_key = os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=api_key, base_url="Your Online Model URL")
        else:
            client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:800{gpu_id}/v1")
        try: 
            response=client.chat.completions.create(
                #prompt=prompt,
                messages=[
                    {"role":"system", "content": "You are a helpful assistant."},
                    {"role": "user","content": prompt}
                ],
                model=model_name,
                temperature=temp,
                n=sample_num,
                max_tokens=max_gen,
                # extra_body={"guided_json": json_schema},
            )
            
            for choice in response.choices:
                response_list.append(choice.message.content)
                #response_list.append(choice.text)
            return response_list
        except Exception as e:
            print(e)
            time.sleep(50)
    return None

def get_pred_from_vllm(rank, data, max_gen, model_name, out_path, sample_num,lock, temp,gpt):
    # print("Temp: ",temp)
    if gpt:
        print("Eval On ",model_name)
    for json_obj in tqdm(data):
        prompt = json_obj['instruction']
        preds = get_api_results(model_name, prompt, rank, sample_num, max_gen=max_gen, temp=temp,gpt=gpt)
        def check_pred_validity(pred:str, prompt):
            if prompt.endswith("Answer:") or prompt.endswith("Type:") or prompt.endswith("Summary: ") or prompt.endswith("Answer:\n") or prompt.endswith("\".\n"):
                return True
            if "\"answer\"" not in pred:
                return False
            return True
        
        if preds==None:
            new_preds = get_api_results(model_name, prompt, rank, 5, max_gen=max_gen, temp=0.3,gpt=gpt)
            if new_preds==None:
                continue
            else:
                preds=new_preds
        
        check_flag=False
        if len(preds) == 1:
            if not check_pred_validity(preds[0], prompt):
                new_preds = get_api_results(model_name, prompt, rank, 5, max_gen=max_gen, temp=0.3)
                if new_preds!=None:
                    for pred in new_preds:
                        if check_pred_validity(pred, prompt):
                            preds = [pred]
                            check_flag=True
                            break
            else:
                check_flag=True
            
            if not check_pred_validity(preds[0], prompt):
                new_preds = get_api_results(model_name, prompt, rank, 10, max_gen=max_gen, temp=0.3)
                if new_preds!=None:
                    for pred in new_preds:
                        if check_pred_validity(pred, prompt):
                            preds = [pred]
                            check_flag=True
                            break
                else:
                    continue
            else:
                check_flag=True

        if "answers" in json_obj.keys():
            instruction, answers, _id = json_obj["instruction"], json_obj["answers"], json_obj["id"]
        else:
            instruction, answers, _id = json_obj["instruction"], [json_obj["output"]], json_obj["id"]
        if "all_classes" in json_obj.keys():
            all_classes=json_obj['all_classes']
        else:
            all_classes=[]
        
        try:
            question=json_obj['question']
        except:
            question = instruction.split("Question:")[-1].replace("\n\nAnswer:","").replace("*","").strip()
        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump({"pred": preds, "instruction": instruction, "question":question, "answers": answers, "id":_id,"check_flag":str(check_flag) ,"all_classes": all_classes, "length": 0}, f, ensure_ascii=False)
                f.write('\n')
    dist.destroy_process_group()
    return

if __name__ == '__main__':
    print(os.getpid())
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method('fork', force=True)
    model_name = args.model
    dataset_name = args.dataset_name

    sources = load_jsonl_file(args.data_path)
    data_all = [data_sample for data_sample in sources]
    data_subsets = [data_all[i::world_size] for i in range(world_size)]

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    processes = []
    lock = mp.RLock()
    for rank in range(world_size):
        p = mp.Process(target=get_pred_from_vllm, args=(rank, data_subsets[rank], args.max_gen, model_name, out_path, args.sample_num, lock, args.temperature,args.gpt))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
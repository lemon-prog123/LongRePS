<p align="center">
    <img src="llama.png" width="150" style="margin-bottom: 0.2;"/>
<p>

<h3 align="center"><a href="https://arxiv.org/pdf/2502.20790">
Chain-of-Thought Matters: Improving Long-Context Language Models with Reasoning Path Supervision</a></h3>

<p align="center">
    🤗 <a href="https://huggingface.co/collections/Lemon123prog/longreps-67c54f91b940623729f7ba9e" target="_blank">HF Repo</a> • 📃 <a href="https://arxiv.org/pdf/2502.20790" target="_blank">Paper</a>
</p>


## ⚙️ How to Prepare Data for Training

**Llama-3.1-8B**:
```python
from datasets import load_dataset
import jsonlines
model="Llama-3.1-8B"
dataset = load_dataset("Lemon123prog/Llama-3.1-8B-LongRePS")
warmup_data=dataset['warmup'].to_list()
orm_data=dataset['train_orm'].to_list()
prm_data=dataset['train_prm'].to_list()

with jsonlines.open(f"/data/{model}_warmup.jsonl", 'w') as writer:
    writer.write_all(warmup_data)

with jsonlines.open(f"/data/{model}_orm.jsonl", 'w') as writer:
    writer.write_all(orm_data)

with jsonlines.open(f"/data/{model}_prm.jsonl", 'w') as writer:
    writer.write_all(prm_data)
```

**Qwen-2.5-7B**:
```python
from datasets import load_dataset
import jsonlines
model="Qwen-2.5-7B"
dataset = load_dataset("Lemon123prog/Qwen-2.5-7B-LongRePS")
warmup_data=dataset['warmup'].to_list()
orm_data=dataset['train_orm'].to_list()
prm_data=dataset['train_prm'].to_list()

with jsonlines.open(f"/data/{model}_warmup.jsonl", 'w') as writer:
    writer.write_all(warmup_data)

with jsonlines.open(f"/data/{model}_orm.jsonl", 'w') as writer:
    writer.write_all(orm_data)

with jsonlines.open(f"/data/{model}_prm.jsonl", 'w') as writer:
    writer.write_all(prm_data)
```

Or you can simply run [preprocess.py](preprocess.py)
```bash
python preprocess_train.py
```

## 🖥️ How to Prepare Data for Evaluating

```bash
bash scripts/preprocess_lb.sh
```
Then you will obtain the processed evaluation data in the **dataset** directory.

## 🍧 Training

### Download base models

```python
from huggingface_hub import snapshot_download
from pathlib import Path
repo_id ="Qwen/Qwen2.5-7B"
root_dir = Path("Your own path for Qwen")
snapshot_download(repo_id=repo_id,local_dir=root_dir/repo_id,repo_type="model")

repo_id ="meta-llama/Llama-3.1-8B"
root_dir = Path("Your own path for Llama")
snapshot_download(repo_id=repo_id,local_dir=root_dir/repo_id,repo_type="model")
```

Set **Model_Path** in the scripts before training.

### Warm Up Stage

**Llama-3.1-8B**
```bash
bash scripts/llama_warmup.sh
```

**Qwen-2.5-7B**
```bash
bash scripts/qwen_warmup.sh
```

### Sample Data and Fine-tune Models

Set **Model-Name** & **Model-Path** & **File-Name** in the scripts before sampling.
```bash
cd evaltoolkits
bash loop_sample.sh
```

After the sampling process, you can use [filter_data.py](evaltoolkits/filter_data.py) to launch the filtering framework.

```bash
cd evaltoolkits
python filter_data.py \
--path_to_src_file [Sampling Data] \
--path_to_stage1_file [Output Data Path]
```

You can modify [dataset_info.json](data/dataset_info.json) to enable the added **filtered dataset** to be included in the file list.

Finally, by set the **warm-up model path** and **datset_name** in the scripts, you can launch the fine-tuning process.

**Llama-3.1-8B**
```bash
bash scripts/llama_sft.sh
```

**Qwen-2.5-7B**
```bash
bash scripts/qwen_sft.sh
```


## 📊 Evaluation

**LongBench v1**
```bash
cd evaltoolkits
bash launch_lbv1.sh
```

**LongBench v2**
```bash
cd evaltoolkits
bash launch_lbv2.sh
```

Note: Set **model_path** and **mode** to the desired target model.


## 👍 Acknowledgement
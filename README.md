<p align="center">
    <img src="pics/llama.png" width="150" style="margin-bottom: 0.2;"/>
<p>

# 📖 Chain-of-Thought Matters: Improving Long-Context Language Models with Reasoning Path Supervision

<p align="center">
    🤗 <a href="https://huggingface.co/collections/Lemon123prog/longreps-67c54f91b940623729f7ba9e" target="_blank">HF Repo</a> • 📃 <a href="https://arxiv.org/pdf/2502.20790" target="_blank">Paper</a>
</p>

**LongRePS** tackles quality bottlenecks in CoT reasoning for extended contexts by integrating process supervision. As shown in the figure, we have discovered that in complex task scenarios, using the chain of thought always enables the model performance to achieve a universal gain. Furthermore, we figure out that while vanilla CoT improves with context length, self-sampled reasoning paths exhibit significant inconsistency and hallucination risks, especially in multi-hop QA and complex scenarios.

The framework operates in two phases: (1) **​Self-sampling**​ generates diverse CoT candidates to capture reasoning variability, and (2) **​Context-aware assessment**​ enforces answer correctness, grounding via text matching, and intrinsic consistency via LLM-based scoring.

Evaluations on long-context tasks show LongRePS achieves 13.6/3.8-point gains on MuSiQue (LLaMA/Qwen) and cross-task robustness, outperforming outcome supervision. The results validate process supervision as pivotal for scalable long-context reasoning, with open-source code enabling community adoption.

![](pics/combined_plot.png)
***
| ​**Model**​            | ​**Short**​ | ​**Medium**​ | ​**Long**​ |
| :-------: | :--------------------: | :--------------------: | :------------------: |
| LLaMA3.1-70B-Instruct | +2.0      | +9.9       | +9.0     |
| Qwen2.5-72B-Instruct  | +1.0      | +5.0       | +6.7     |
| Claude-3.5-Sonnet     | +3.5      | +9.3       | +1.6     |
| GPT-4o               | -0.5      | +5.5       | +5.7     |
| GPT-4o-mini          | +4.0      | +1.3       | +6.2     |
| ​Avg.​             | +2.0      | +6.2       | +5.8     |

## 🔍 List of Contents
- [⚙️ How to Prepare Data for Training](#how-to-Prepare-Data-for-Training)
- [🖥️ How to Prepare Data for Evaluating](#how-to-Prepare-Data-for-Evaluating)
- [🍧 Training](#training)
- [📊 Evaluation](#evaluation)
- [📄 Acknowledgement](#acknowledgement)

<a name="how-to-Prepare-Data-for-Training"></a>

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

<a name="how-to-Prepare-Data-for-Evaluating"></a>

## 🖥️ How to Prepare Data for Evaluating

```bash
bash scripts/preprocess_lb.sh
```
Then you will obtain the processed evaluation data in the **dataset** directory.

<a name="training"></a>

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

<a name="evaluation"></a>

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

<a name="acknowledgement"></a>

## 📄 Acknowledgement
We are deeply thankful for the following projects that serve as the foundation for LongRePS:

* [**SEALONG**](https://github.com/SihengLi99/SEALONG)
* [**LongBench**](https://github.com/THUDM/LongBench)
* [**LLaMA-Factory**](https://github.com/hiyouga/LLaMA-Factory)


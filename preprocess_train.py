from datasets import load_dataset
import jsonlines


def preprocess(model:str):
    dataset = load_dataset(f"Lemon123prog/{model}-LongRePS")
    warmup_data=dataset['warmup'].to_list()
    orm_data=dataset['train_orm'].to_list()
    prm_data=dataset['train_prm'].to_list()

    with jsonlines.open(f"./data/{model}_warmup.jsonl", 'w') as writer:
        writer.write_all(warmup_data)

    with jsonlines.open(f"./data/musique-{model}_orm_train.jsonl", 'w') as writer:
        writer.write_all(orm_data)

    with jsonlines.open(f"./data/musique-{model}_prm_train.jsonl", 'w') as writer:
        writer.write_all(prm_data)


preprocess("Llama-3.1-8B")
preprocess("Qwen-2.5-7B")
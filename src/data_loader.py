from datasets import load_dataset
from transformers import AutoTokenizer
import json
import yaml
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from src.zero_shot_parsers import pyd_format

prompt_path = os.path.join(parent_dir, "Prompt.yml")


with open(prompt_path, "r") as f:
    prompt = yaml.safe_load(f)

def load_data():
    dataset = load_dataset("google-research-datasets/paws", "labeled_final")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    validation_dataset = dataset["validation"]
    return train_dataset, test_dataset, validation_dataset


def build_messages(batch):
    prompt_str = f"""
<role>{prompt['role'].strip()}</role>
<task_definition>{prompt['task_definition'].strip()}</task_definition>
<instructions>{prompt['instructions'].strip()}</instructions>
<output_format>{pyd_format.strip()}</output_format>
"""

    batch_size = len(batch['id'])
    
    # 使用列表推导式替代 for 循环
    all_messages = [
        [
            {"role": "system", "content": prompt_str},
            {"role": "user", "content": json.dumps(
                {
                    "id": batch['id'][i],
                    "sentence1": batch['sentence1'][i],
                    "sentence2": batch['sentence2'][i]
                }, 
                ensure_ascii=False
            ).strip()}
        ]
        for i in range(batch_size)
    ]

    return {"messages": all_messages}



def build_prompt_applier(tokenizer, enable_thinking=False):
    def apply_prompt(batch):
        messages = build_messages(batch)["messages"]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking  # 关闭思考模式以提高速度
        )
        return {"text": text}
    return apply_prompt


if __name__ == "__main__":
    pass

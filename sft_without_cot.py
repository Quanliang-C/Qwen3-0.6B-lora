import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from src.lora_data_loader import load_data, build_prompt_applier, build_full_prompt_applier
from src.zero_shot_parsers import pyd_parser, pyd_format

import argparse

parser = argparse.ArgumentParser()
## 不要启用thinking，这里bool类型会报错
parser.add_argument("--think", type=bool, default=False)
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--lora_dropout", type=float, default=0.1)
parser.add_argument("--load_batch_size", type=int, default=32)
parser.add_argument("--train_batch_size", type=int, default=16)
parser.add_argument("--eval_batch_size", type=int, default=16)
parser.add_argument("--num_train_epochs", type=int, default=3)
parser.add_argument("--learning_rate", type=float, default=2e-4)
parser.add_argument("--warmup_steps", type=int, default=100)
args = parser.parse_args()

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

"""load model, dataset, tokenizer, prompt applier"""
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

## 不使用device_map, 统一交给trainer处理
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
train_dataset, _, validation_dataset = load_data()
full_prompt_applier = build_full_prompt_applier(tokenizer, enable_thinking=args.think)
short_prompt_applier = build_prompt_applier(tokenizer, enable_thinking=args.think)



"""载入验证集，使用完整prompt（包含答案）"""
# 验证集需要完整的对话（prompt + answer）来计算token拟合度
validation_full_dataset = validation_dataset.map(
    full_prompt_applier,
    batched=True,
    batch_size=args.load_batch_size
)
validation_short_dataset = validation_dataset.map(
    short_prompt_applier,
    batched=True,
    batch_size=args.load_batch_size
)

# tokenize验证集 (结果是python list)
# 不使用padding, 统一交给data_collator处理
validation_full_tokenized = validation_full_dataset.map(
    lambda x: tokenizer(x['text'], truncation=True),
    batched=True,
    batch_size=args.load_batch_size
)
validation_short_tokenized = validation_short_dataset.map(
    lambda x: tokenizer(x['text'], truncation=True),
    batched=True,
    batch_size=args.load_batch_size
)

## 取消，统一交给trainer处理
# validation_full_tokenized = validation_full_tokenized.remove_columns(["sentence1", "sentence2", "text"])
# validation_short_tokenized = validation_short_tokenized.remove_columns(["sentence1", "sentence2", "text"])

# validation_full_tokenized = validation_full_tokenized.rename_column("label", "metric_label")
# validation_short_tokenized = validation_short_tokenized.rename_column("label", "metric_label")

# 创建验证集的masked labels (使用 python lists)
validation_masked_labels = []
for i in range(len(validation_full_tokenized)):
    # 使用 .copy()
    full_ids = validation_full_tokenized[i]["input_ids"].copy()
    prompt_len = len(validation_short_tokenized[i]["input_ids"])
    
    # 为列表切片分配一个可迭代对象
    full_ids[:prompt_len] = [-100] * prompt_len

    validation_masked_labels.append(full_ids)

validation_full_tokenized = validation_full_tokenized.add_column("labels", validation_masked_labels)

# 在所有操作完成后，最后转换为torch格式
# validation_full_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels", "metric_label"])
# 取消，统一交给trainer处理



"""prepare train dataset"""
train_full_dataset = train_dataset.map(
    full_prompt_applier,
    batched=True,
    batch_size=args.load_batch_size
)

train_short_dataset = train_dataset.map(
    short_prompt_applier,
    batched=True,
    batch_size=args.load_batch_size
)

# 不使用padding, 统一交给data_collator处理
input_full_tokenized_datasets = train_full_dataset.map(
    lambda x: tokenizer(x['text'], truncation=True),
    batched=True,
    batch_size=args.load_batch_size
)

input_short_tokenized_datasets = train_short_dataset.map(
    lambda x: tokenizer(x['text'], truncation=True),
    batched=True,
    batch_size=args.load_batch_size
)


"""make trianning data for sft, add masked labels"""
masked_train_labels = []

for i in range(len(input_full_tokenized_datasets)):
    # 直接操作 list，无需转换为 tensor 再转回
    full_ids = input_full_tokenized_datasets[i]["input_ids"]
    prompt_len = len(input_short_tokenized_datasets[i]["input_ids"])
    
    # 克隆一份 list 来创建 labels
    labels = list(full_ids)
    labels[:prompt_len] = [-100] * prompt_len
    masked_train_labels.append(labels)

## ready to sft
train_dataset_for_sft = input_full_tokenized_datasets.add_column("labels", masked_train_labels)





## 清除所有不干净的列
final_train_dataset = train_dataset_for_sft.remove_columns(
    [col for col in train_dataset_for_sft.column_names if col not in ["input_ids", "labels", "attention_mask"]]
)
final_validation_dataset = validation_full_tokenized.remove_columns(
    [col for col in validation_full_tokenized.column_names if col not in ["input_ids", "labels", "attention_mask"]]
)




lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

data_collator = DataCollatorForSeq2Seq(
    model=model,
    tokenizer=tokenizer,
    pad_to_multiple_of=8
)

training_args = TrainingArguments(
    output_dir="./sft_without_cot_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=10,
    logging_dir="./logs",
    logging_first_step=True,
    logging_nan_inf_filter=False,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    num_train_epochs=args.num_train_epochs,
    lr_scheduler_type="cosine",
    warmup_steps=args.warmup_steps,
    log_level="warning",
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    bf16=use_bf16,
    fp16=not use_bf16 and torch.cuda.is_available(),
    dataloader_num_workers=2
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=final_train_dataset,
    eval_dataset=final_validation_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# 开始训练（每个epoch会自动计算metrics）
print("🚀 开始训练...")
trainer.train()

# 保存模型
model.save_pretrained("./sft_without_cot_final")
tokenizer.save_pretrained("./sft_without_cot_final")

print("✅ 微调完成！模型已保存到 ./sft_without_cot_final")








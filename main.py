import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

# 환경 변수
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
dataset_path = "data/instruction_data_500.jsonl"
output_dir = "./outputs"

# 토크나이저 및 모델 로딩
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"": 0},  # 모든 연산을 GPU 0에서 수행
    torch_dtype=torch.float16
)

# LoRA 설정
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, peft_config)

# 데이터셋 로딩 및 전처리
dataset = load_dataset("json", data_files=dataset_path, split="train")

def generate_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Output:\n{example['output']}"

dataset = dataset.map(lambda x: {"text": generate_prompt(x)})
tokenized = dataset.map(
    lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512),
    batched=True,
    remove_columns=dataset.column_names
)
tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]})

# 학습 설정
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    warmup_steps=10,
    max_steps=45,
    logging_steps=5,
    save_steps=45,
    fp16=True,
    logging_dir=f"{output_dir}/logs",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
)

trainer.train()

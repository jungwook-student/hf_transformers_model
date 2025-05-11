import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import os

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
hf_token = os.environ.get("HF_TOKEN")

print("📦 모델 및 토크나이저 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=hf_token,
    device_map="auto",
    torch_dtype=torch.float16,
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_config)

print("✅ 모델 로딩 완료")

print("📚 데이터셋 로딩 및 전처리 중...")
dataset = load_dataset("json", data_files="train.jsonl")["train"]

def generate_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Output:\n{example['output']}"

dataset = dataset.map(lambda x: {"text": generate_prompt(x)})
dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512), batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

print("✅ 데이터셋 준비 완료")

training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    fp16=True,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

print("🚀 학습 시작...")
trainer.train()
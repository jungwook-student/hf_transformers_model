
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
hf_token = os.environ.get("HF_TOKEN")

print("📦 모델 및 토크나이저 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=hf_token,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

print("✅ 모델 로딩 완료")

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, peft_config)

print("📚 데이터셋 로딩 및 전처리 중...")
dataset = load_dataset("json", data_files="data/instruction_data_500.jsonl")["train"]

def generate_prompt(example):
    return f"### Instruction:\n{example['prompt']}\n\n### Output:\n{example['completion']}"

dataset = dataset.map(lambda x: {"text": generate_prompt(x)})

tokenized = dataset.map(
    lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512),
    batched=True,
    remove_columns=["prompt", "completion", "text"]
)

training_args = TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=1,
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=2,
    ddp_find_unused_parameters=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized,
)

print("🚀 학습 시작...")
trainer.train()

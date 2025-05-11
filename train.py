from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
import os

# 모델 정보
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
hf_token = os.environ["HF_TOKEN"]

# 토크나이저 및 모델 로딩
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token, device_map="auto", torch_dtype="auto")

# LoRA 설정
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    task_type=TaskType.CAUSAL_LM,
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, peft_config)

# 데이터 로딩 및 포맷
dataset = load_dataset("json", data_files={"train": "data/instruction_data_500.jsonl"})["train"]

def format(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    return {"text": prompt}

dataset = dataset.map(format)

tokenized = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512), batched=True)

# 학습 인자 설정
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="no",
    bf16=True,
    report_to="none"
)

# Trainer 구성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True)
)

trainer.train()

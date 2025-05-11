import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType

# 환경 변수 및 모델 정보
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
hf_token = os.environ["HF_TOKEN"]

# 모델 및 토크나이저 로딩
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=hf_token,
    device_map="auto",
    torch_dtype="auto"
)

# LoRA 설정
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    task_type=TaskType.CAUSAL_LM,
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, peft_config)

# 데이터셋 로딩 및 포맷팅
dataset = load_dataset("json", data_files={"train": "data/instruction_data_500.jsonl"})["train"]

def format(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    return {"text": prompt}

dataset = dataset.map(format)

# 텍스트 → 토큰 변환
tokenized = dataset.map(
    lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512),
    batched=True,
    remove_columns=["instruction", "input", "output", "text"]
)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="no",
    bf16=True,
    remove_unused_columns=False,
    report_to="none"
)

# Collator는 padding이 된 토큰들을 처리해주는 역할
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer 정의 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

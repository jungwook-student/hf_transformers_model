import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    pipeline,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType

# 환경 변수
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
hf_token = os.environ["HF_TOKEN"]

# 모델 및 토크나이저 로딩
print("📦 모델 및 토크나이저 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=hf_token,
    torch_dtype=torch.float16,
    device_map="auto"  # 멀티 GPU 자동 분산
)

model = get_peft_model(model, LoraConfig(
    r=8,
    lora_alpha=32,
    task_type=TaskType.CAUSAL_LM,
    lora_dropout=0.05,
    bias="none"
))
model.config.use_cache = False  # Trainer compatibility

# 데이터셋 로딩 및 전처리
print("📚 데이터셋 로딩 및 전처리 중...")
dataset = load_dataset("json", data_files={"train": "data/instruction_data_500.jsonl"})["train"]

def format(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    return {"text": prompt}

dataset = dataset.map(format)

tokenized = dataset.map(
    lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512),
    batched=True,
    remove_columns=["instruction", "input", "output", "text"]
)

# Trainer 설정
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="no",
    fp16=True,
    remove_unused_columns=False,
    report_to="none",
    ddp_find_unused_parameters=False  # 🔥 device_map="auto" 사용 시 필수
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 학습 시작
print("🚀 학습 시작...")
trainer.train()
print("✅ 학습 완료!")

# 간단한 추론 테스트
print("🔍 추론 테스트 시작...")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

examples = [
    "첫 등원에 읽기 좋은 놀이책을 찾고 있어요.",
    "동물을 배울 수 있는 책 있으면 알려주세요.",
    "놀이책 중에서 4-6세가 좋아할 만한 책 있을까요?",
    "2-4세 아이랑 읽기 좋은 자연 책 추천해 주세요.",
    "생일에 읽기 좋은 놀이책을 찾고 있어요."
]

for prompt in examples:
    formatted = f"### Instruction:\n{prompt}\n\n### Input:\n\n### Response:\n"
    output = pipe(formatted, max_new_tokens=64)[0]["generated_text"]
    print("🧾 입력:", prompt)
    print("📘 응답:", output.split('### Response:')[-1].strip())
    print("-" * 80)

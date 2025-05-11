import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import BitsAndBytesConfig
import os
import time

print("📦 모델 및 토크나이저 로딩 중...")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.float16)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

base_model = prepare_model_for_kbit_training(base_model)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, config)

print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} || all params: {sum(p.numel() for p in model.parameters()):,} || trainable%: {100 * sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(p.numel() for p in model.parameters()):.4f}")
print("✅ 모델 로딩 완료")

print("📚 데이터셋 로딩 및 전처리 중...")

dataset = load_dataset("json", data_files="data/instruction_data_500.jsonl")["train"]

def generate_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"

dataset = dataset.map(lambda x: {"text": generate_prompt(x)})
dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512), batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_steps=5,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="no",
    fp16=True,
    logging_dir="logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

print("🚀 학습 시작...")
trainer.train()

print("✅ 학습 완료, 예제 문장 테스트 중...")

peft_model_path = "output/final"
model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)

# 예제 테스트
model.eval()
inputs = tokenizer(
    [
        "### Instruction:\n다음 문장을 분석하여 도서 추천 조건을 추출하세요.\n\n### Input:\n4-5세 아이가 좋아할만한 놀이책을 찾고 있어요.\n\n### Response:\n",
        "### Instruction:\n다음 문장을 분석하여 도서 추천 조건을 추출하세요.\n\n### Input:\n첫 등원에 도움이 되는 책 알려줘.\n\n### Response:\n"
    ],
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        temperature=0.7
    )
    for i, output in enumerate(outputs):
        print(f"📘 예제 {i+1} 결과:")
        print(tokenizer.decode(output, skip_special_tokens=True))
        print("="*50)

print("🛑 작업 완료. SSH 연결로 접속 시 세션 유지를 위해 종료하지 않고 대기합니다.")
while True:
    time.sleep(60)
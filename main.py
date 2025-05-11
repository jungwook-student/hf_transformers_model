
import torch
from transformers import (
    AutoModelForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from transformers import BitsAndBytesConfig
import os

# ✅ 모델 이름
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ✅ 토크나이저 로딩 (tokenizer_config.json 이 없으므로 명시적으로 LlamaTokenizer 사용)
tokenizer = LlamaTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# ✅ QLoRA 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# ✅ PEFT: LoRA 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ✅ 모델 로딩 및 준비
print("📦 모델 및 토크나이저 로딩 중...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100 * trainable / total:.4f}")
print("✅ 모델 로딩 완료")

# ✅ 데이터셋 로딩 및 전처리
print("📚 데이터셋 로딩 및 전처리 중...")
dataset = load_dataset("json", data_files="data/instruction_data_500.jsonl")["train"]

def generate_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Output:\n{example['output']}"

dataset = dataset.map(lambda x: {"text": generate_prompt(x)})
dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512), batched=True)

# ✅ 학습 설정
training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    fp16=True,
    logging_steps=10,
    save_strategy="no",
    learning_rate=2e-4,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ✅ 트레이너 정의 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("🚀 학습 시작...")
trainer.train()

# ✅ 예제 문장 테스트
print("\n🧪 예제 문장 테스트:")
test_inputs = [
    "첫 등원에 읽기 좋은 놀이책을 찾고 있어요.",
    "동물을 배울 수 있는 책 있으면 알려주세요.",
    "놀이책 중에서 4-6세세가 좋아할 만한 책 있을까요?",
    "2-4세 아이랑 읽기 좋은 자연 책 추천해 주세요.",
    "자연과학 주제의 책 중에서 재미있는 이야기 형식이 있나요?"
]

for i, input_text in enumerate(test_inputs, 1):
    prompt = f"### Instruction:\n다음 문장을 분석하여 도서 추천 조건을 추출하세요.\n\n### Input:\n{input_text}\n\n### Output:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"예제 {i}:\n{decoded}\n")

# ✅ 세션 유지
print("✅ 학습 및 테스트 완료. 세션을 유지합니다. Ctrl+C 로 종료하세요.")
import time
while True:
    time.sleep(60)

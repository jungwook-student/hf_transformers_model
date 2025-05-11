import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os
import time

# ✅ 모델과 토크나이저 불러오기
print("📦 모델 및 토크나이저 로딩 중...")
model_name = "NousResearch/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True
)
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} || "
      f"all params: {sum(p.numel() for p in model.parameters()):,} || "
      f"trainable%: {100 * sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(p.numel() for p in model.parameters()):.4f}")

print("✅ 모델 로딩 완료")

# ✅ 데이터 로딩
print("📚 데이터셋 로딩 및 전처리 중...")
dataset = load_dataset("json", data_files="data/instruction_data_500.jsonl")["train"]

def generate_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Output:\n{example['output']}"

dataset = dataset.map(lambda x: {"text": generate_prompt(x)})
dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512), batched=True)

# ✅ 학습 설정
training_args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_steps=10,
    max_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    tokenizer=tokenizer
)

# ✅ 학습 시작
print("🚀 학습 시작...")
trainer.train()

# ✅ 예제 문장 테스트
print("\n📌 예제 문장 테스트 중...\n")
model.eval()
example_inputs = [
    "첫 등원에 읽기 좋은 놀이책을 찾고 있어요.",
    "동물을 배울 수 있는 책 있으면 알려주세요.",
    "놀이책 중에서 4-6세세가 좋아할 만한 책 있을까요?",
    "2-4세세 아이랑 읽기 좋은 자연 책 추천해 주세요.",
    "잠들기 전 읽기 좋은 동화책이 필요해요."
]

for input_text in example_inputs:
    prompt = f"### Instruction:\n다음 문장을 분석하여 도서 추천 조건을 추출하세요.\n\n### Input:\n{input_text}\n\n### Output:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"📝 입력: {input_text}")
    print(f"📤 출력: {decoded[len(prompt):].strip()}\n")

# ✅ 무한 대기
print("🕓 학습 종료 후 대기 중 (모델 확인용)...")
while True:
    time.sleep(60)

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from transformers.trainer_callback import EarlyStoppingCallback

# ✅ 단일 GPU만 사용하도록 제한
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ✅ 모델 로딩
print("📦 모델 및 토크나이저 로딩 중...")
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# ✅ LoRA 구성 적용
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
print("✅ 모델 로딩 완료")

# ✅ 데이터 로딩 및 전처리
print("📚 데이터셋 로딩 및 전처리 중...")
dataset = load_dataset("json", data_files="data/instruction_data_500.jsonl")["train"]

def generate_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Output:\n{example['output']}"

dataset = dataset.map(lambda x: {"text": generate_prompt(x)})
dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512), batched=True)

# ✅ 학습 설정
training_args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_dir="logs",
    logging_steps=10,
    save_total_limit=2,
    save_steps=50,
    evaluation_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
print("🚀 학습 시작...")
trainer.train()

# ✅ 예제 문장 테스트
print("\n🎯 학습 결과 테스트:")
example_prompts = [
    "첫 등원에 읽기 좋은 놀이책을 찾고 있어요.",
    "동물을 배울 수 있는 책 있으면 알려주세요.",
    "4-6세가 좋아할 놀이책 있을까요?",
    "자연 관련 책 추천해 주세요.",
    "소리나는 책이나 촉감책 같은 거 있어요?"
]
model.eval()
for i, prompt in enumerate(example_prompts):
    input_text = f"### Instruction:\n다음 문장을 분석하여 도서 추천 조건을 추출하세요.\n\n### Input:\n{prompt}\n\n### Output:\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"[{i+1}] {output_text}")

# ✅ 프로세스 종료 막기 (SSH 등 확인용)
print("\n⏳ 작업 완료. 세션을 유지 중입니다. Ctrl+C 또는 수동 종료 필요.")
import time
while True:
    time.sleep(60)

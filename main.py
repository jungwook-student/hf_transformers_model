import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from trl import SFTTrainer

# ✅ 모델 로딩
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("📦 모델 및 토크나이저 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # 다중 GPU 고려
    torch_dtype=torch.float16,
    quantization_config=bnb_config
)
model = prepare_model_for_kbit_training(model)

# ✅ LoRA 설정
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
      f"|| all params: {sum(p.numel() for p in model.parameters()):,} "
      f"|| trainable%: {100 * sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(p.numel() for p in model.parameters()):.4f}")
print("✅ 모델 로딩 완료")

# ✅ 데이터 전처리
print("📚 데이터셋 로딩 및 전처리 중...")
dataset = load_dataset("json", data_files="data/instruction_data_500.jsonl")["train"]

def generate_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Output:\n{example['output']}"

dataset = dataset.map(lambda x: {"text": generate_prompt(x)})
dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512), batched=True)

# ✅ Trainer 설정
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=1,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="no"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    tokenizer=tokenizer,
)

print("🚀 학습 시작...")
trainer.train()

# ✅ 예제 문장 테스트
print("✅ 학습 완료, 예제 문장 테스트 중...")
model.eval()
inputs = [
    "첫 등원에 읽기 좋은 놀이책을 찾고 있어요.",
    "동물을 배울 수 있는 책 있으면 알려주세요.",
    "놀이책 중에서 4-6세가 좋아할 만한 책 있을까요?",
    "2-4세 아이랑 읽기 좋은 자연 책 추천해 주세요.",
    "형제 갈등이 있는 아이에게 도움이 되는 책 있을까요?"
]
for i, sentence in enumerate(inputs, 1):
    prompt = f"### Instruction:\n다음 문장을 분석하여 도서 추천 조건을 추출하세요.\n\n### Input:\n{sentence}\n\n### Output:\n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, max_new_tokens=50, do_sample=False)
    print(f"[예제 {i}]")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print()

# ✅ 종료 방지
print("🕓 스크립트 종료 방지 중... Ctrl+C 로 종료 가능")
import time
while True:
    time.sleep(60)
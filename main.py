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
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("📦 모델 및 토크나이저 로딩 중...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    quantization_config=bnb_config
)
model.to("cuda")
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
      f"|| trainable%: {100 * sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(p.numel() for p in model.parameters()):.4f}", flush=True)
print("✅ 모델 로딩 완료", flush=True)

# ✅ 데이터 전처리
print("📚 데이터셋 로딩 및 전처리 중...", flush=True)
dataset = load_dataset("json", data_files="data/infant_book_recommendation_dataset.jsonl")["train"]

def generate_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Output:\n{example['output']}"

dataset = dataset.map(lambda x: {"text": generate_prompt(x)})
dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512), batched=True)

# ✅ Trainer 설정
training_args = TrainingArguments(
    output_dir="./output_mistral",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=1,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="epoch",
    save_total_limit=1
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args
)

print("🚀 학습 시작...", flush=True)
trainer.train()
trainer.save_model("./output_mistral")  # 학습 완료 후 직접 저장

# ✅ 예제 문장 테스트
print("✅ 학습 완료, 예제 문장 테스트 중...", flush=True)
model.eval()
inputs = [
    "형제가 생긴 아이에게 읽혀줄 책 추천해줘.",
    "동물을 배울 수 있는 동물그림책 을 추천해주세요",
    "배변훈련을 하고 있는 우리 아이에게 읽어줄만한 책이 있을까?",
    "우리 아이가 유치원 가기 무서워해요. 도움이 될만한 책 없을까요?",
    "숫자에 대해서 가르쳐줄수 있는 재미있는 그림책을 추천해줘. 3살 이야"
]
for i, sentence in enumerate(inputs, 1):
    prompt = f"### Instruction:\n다음 문장을 분석하여 도서 추천 조건을 추출하세요.\n\n### Input:\n{sentence}\n\n### Output:\n"
    device = model.device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    print(f"[예제 {i}]", flush=True)
    print(" ⏳ generating...", flush=True)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=30,
            do_sample=False,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=False
        )
    print(" ✅ generation complete.", flush=True)
    if outputs is None:
        print("❌ generate() returned None", flush=True)
        import sys; sys.exit(1)

    print(f"output type: {type(outputs)}", flush=True)
    try:
        if isinstance(outputs, torch.Tensor):
            tokens = outputs[0]
        elif isinstance(outputs, list) and isinstance(outputs[0], torch.Tensor):
            tokens = outputs[0]
        else:
            raise ValueError(f"Unexpected output type: {type(outputs)}")
    except Exception as e:
        print("❌ error while parsing outputs:", e, flush=True)
        import sys; sys.exit(1)

    print(f"output shape: {tokens.shape}", flush=True)
    print(f"raw token ids: {tokens.tolist()[:20]} ...", flush=True)
    print(" ⏳ decoding...", flush=True)
    try:
        decoded = tokenizer.decode(
            tokens.cpu(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        print(" ✅ decoded.", flush=True)
        print(decoded, flush=True)
    except Exception as e:
        print("❌ decoding failed:", e, flush=True)
    print()

# ✅ 종료 방지
print("🕓 스크립트 종료 방지 중... Ctrl+C 로 종료 가능", flush=True)
import time
while True:
    time.sleep(60)

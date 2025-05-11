
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from transformers import BitsAndBytesConfig
import os

# Logging
print("📦 모델 및 토크나이저 로딩 중...")

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model = model.to("cuda")

print("✅ 모델 로딩 완료")

# Load dataset
print("📚 데이터셋 로딩 및 전처리 중...")
dataset = load_dataset("json", data_files="data/instruction_data_500.jsonl")["train"]

def generate_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Output:\n{example['output']}"

dataset = dataset.map(lambda x: {"text": generate_prompt(x)})
dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512), batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Training
training_args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_total_limit=1,
    save_strategy="no",
    report_to="none",
    fp16=True,
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print("🚀 학습 시작...")
trainer.train()
print("✅ 학습 완료")

# 테스트용 예제
print("\n🧪 예제 문장 테스트:")
test_examples = [
    "동물을 배울 수 있는 책 있으면 알려주세요.",
    "4~6세가 좋아할만한 놀이책 추천해줘.",
    "자연을 주제로 한 책이 필요해.",
    "첫 등원에 읽기 좋은 책을 찾고 있어.",
    "2-4세 아이와 함께 볼 수 있는 책이 있을까?"
]

model.eval()
for prompt in test_examples:
    input_text = f"### Instruction:\n다음 문장을 분석하여 도서 추천 조건을 추출하세요.\n\n### Input:\n{prompt}\n\n### Output:\n"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=100)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"📝 입력: {prompt}\n🔍 출력: {decoded}\n")

# 종료 방지
input("🔒 학습이 완료되었습니다. 작업 공간을 종료하지 않으려면 이 창을 열어두세요. 종료하려면 Enter 키를 누르세요.")

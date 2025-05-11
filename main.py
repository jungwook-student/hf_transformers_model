import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset

print("📦 모델 및 토크나이저 로딩 중...")
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    load_in_4bit=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
tokenizer.pad_token = tokenizer.eos_token

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
print("✅ 모델 로딩 완료")

print("📚 데이터셋 로딩 및 전처리 중...")
dataset = load_dataset("json", data_files="data/instruction_data_500.jsonl")["train"]

def generate_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Output:\n{example['output']}"

dataset = dataset.map(lambda x: {"text": generate_prompt(x)})
dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512), batched=True)

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="no",
    fp16=True,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

print("🚀 학습 시작...")
trainer.train()

# 예제 추론
print("\n📌 학습 완료! 예제 문장 테스트 결과:")
example_prompts = [
    "처음 유치원 가는 날 아이가 볼만한 책이 있을까?",
    "동물에 대한 흥미를 높여줄 그림책이 필요해.",
    "3~5세가 좋아할 수 있는 자연 관련 책을 추천해줘.",
    "첫 등원에 긴장한 아이에게 도움이 되는 책이 있을까?",
    "요즘 아이가 숫자에 관심이 많아졌어. 좋은 책 있을까?"
]

model.eval()
for prompt in example_prompts:
    input_text = f"### Instruction:\n다음 문장을 분석하여 도서 추천 조건을 추출하세요.\n\n### Input:\n{prompt}\n\n### Output:\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"💬 입력: {prompt}")
    print(f"🧠 출력: {decoded.split('### Output:')[-1].strip()}\n")

input("🔚 프로그램 종료를 원하면 Enter 키를 누르세요.")

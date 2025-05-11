
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import BitsAndBytesConfig

print("📦 모델 및 토크나이저 로딩 중...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("✅ 모델 로딩 완료")
print("📚 데이터셋 로딩 및 전처리 중...")

dataset = load_dataset("json", data_files="data/instruction_data_500.jsonl")["train"]

def generate_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Output:\n{example['output']}"

dataset = dataset.map(lambda x: {"text": generate_prompt(x)})
def tokenize_function(examples):
    outputs = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

dataset = dataset.map(tokenize_function, batched=True)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir="outputs",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    logging_steps=1,
    save_strategy="no",
    learning_rate=2e-4,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

print("🚀 학습 시작...")
trainer.train()

print("✅ 학습 완료 - 예제 문장 추론 테스트")

# 사후 추론 예제 문장 테스트
example_prompts = [
    "첫 등원에 읽기 좋은 놀이책을 찾고 있어요.",
    "동물을 배울 수 있는 책 있으면 알려주세요.",
    "놀이책 중에서 4-6세세가 좋아할 만한 책 있을까요?",
    "2-4세세 아이랑 읽기 좋은 자연 책 추천해 주세요.",
    "감정을 배울 수 있는 책이 있으면 추천해주세요."
]

model.eval()
model.config.use_cache = True

for prompt in example_prompts:
    full_prompt = f"### Instruction:\n다음 문장을 분석하여 도서 추천 조건을 추출하세요.\n\n### Input:\n{prompt}\n\n### Output:"
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    print(f"🔹 입력 문장: {prompt}")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("------")

# SSH를 통한 접근을 위해 무한 루프로 종료 방지
print("⏳ 학습이 끝났습니다. SSH 접속을 위한 대기 상태입니다...")
while True:
    pass

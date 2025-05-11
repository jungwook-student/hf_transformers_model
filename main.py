
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

# ✅ 환경 변수
hf_token = os.environ["HF_TOKEN"]
model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# ✅ 모델 및 토크나이저 로딩
print("📦 모델 및 토크나이저 로딩 중...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=hf_token,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))

# ✅ 데이터셋 전처리
print("📚 데이터셋 로딩 및 전처리 중...")
dataset = load_dataset("json", data_files="data/instruction_data_500.jsonl", split="train")

def generate_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Output:\n{example['output']}"

dataset = dataset.map(lambda x: {"text": generate_prompt(x)})
tokenized = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512), batched=True)

# ✅ 학습 인자 설정
args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    warmup_steps=10,
    max_steps=375,
    learning_rate=2e-4,
    logging_steps=10,
    fp16=True,
    save_strategy="epoch",
    report_to="none"
)

# ✅ 학습 시작
print("🚀 학습 시작...")
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
)
trainer.train()

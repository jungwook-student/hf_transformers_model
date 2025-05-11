import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset
from transformers import BitsAndBytesConfig
from datetime import datetime

# ✅ 설정
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
hf_token = os.environ.get("HF_TOKEN")
dataset_path = "data/instruction_data_500.jsonl"

# ✅ 프롬프트 템플릿 함수
def generate_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Output:\n{example['output']}"

# ✅ Tokenizer 로딩
print("📦 모델 및 토크나이저 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token

# ✅ Model 로딩 및 준비
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             quantization_config=bnb_config,
                                             device_map={"": 0},
                                             token=hf_token)
model.config.use_cache = False
model.config.pretraining_tp = 1

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj"],
                         lora_dropout=0.05, bias="none")
model = get_peft_model(model, peft_config)

# ✅ 데이터셋 로딩
print("📚 데이터셋 로딩 및 전처리 중...")
dataset = load_dataset("json", data_files=dataset_path)["train"]
dataset = dataset.map(lambda x: {"text": generate_prompt(x)})
tokenized = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512), batched=True)

# ✅ 학습 인자 및 Trainer 설정
output_dir = f"outputs/mistral-tuned-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="no",
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    report_to="none"
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized,
    args=training_args,
    tokenizer=tokenizer
)

# ✅ 학습 시작
print("🚀 학습 시작...")
trainer.train()
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import subprocess
import time

print("🔧 시작: 환경 변수 확인")
model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
hf_token = os.getenv("HF_TOKEN")

if hf_token:
    print("✅ HF_TOKEN 환경 변수 확인됨")
else:
    print("❌ HF_TOKEN 환경 변수가 설정되지 않음 — 종료합니다.")
    exit(1)

print("📦 토크나이저 로딩 시작...")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
print("✅ 토크나이저 로딩 완료")

# 캐시 경로 설정
cache_dir = os.path.expanduser("~/.cache/huggingface")

print("📦 모델 로딩 시작 (device_map='auto')...")
print(f"📁 현재 캐시 디렉토리: {cache_dir}")

# 캐시 변화 추적
def show_cache_size():
    try:
        output = subprocess.check_output(["du", "-sh", cache_dir])
        print(f"📦 캐시 크기: {output.decode('utf-8').strip()}")
    except Exception as e:
        print(f"⚠️ 캐시 확인 실패: {e}")

# 캐시 상태 주기적으로 출력 (배경에서)
for i in range(3):
    show_cache_size()
    time.sleep(5)

# 모델 로딩 (이 단계에서 오래 걸릴 수 있음)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    token=hf_token
)
print("✅ 모델 로딩 완료")

# 프롬프트 구성 및 추론
user_input = "손흥민은 몇살이야?"
prompt = f"<|begin_of_text|><|user|>\n{user_input}<|end_of_text|>\n<|assistant|>\n"

print("📝 프롬프트 구성 완료")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("🧠 추론 시작...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id
    )
print("✅ 추론 완료")

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("🧾 모델 응답:")
print(response.replace(prompt, "").strip())
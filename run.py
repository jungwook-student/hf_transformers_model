from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

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

print("📦 모델 로딩 시작 (GPU 강제 할당)...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map={"": 0},  # GPU 0번 강제 지정
    token=hf_token
)
print("✅ 모델 로딩 완료")

print("📝 프롬프트 구성...")
user_input = "손흥민은 몇살이야?"
prompt = f"<|begin_of_text|><|user|>\n{user_input}<|end_of_text|>\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print("✅ 입력 토큰 변환 완료")

print("🧠 모델 추론 시작...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id
    )
print("✅ 모델 추론 완료")

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("🧾 모델 응답:")
print(response.replace(prompt, "").strip())
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# 모델 ID 및 토큰 불러오기
model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
hf_token = os.getenv("HF_TOKEN")

# 토크나이저 & 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    token=hf_token
)

# 테스트 입력 (LLaMA 3 채팅 스타일 프롬프트)
user_input = "손흥민은 몇살이야?"
prompt = f"<|begin_of_text|><|user|>\n{user_input}<|end_of_text|>\n<|assistant|>\n"

# 토큰화 및 모델 추론
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id
    )

# 출력 디코딩
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("🧠 모델 응답:")
print(response.replace(prompt, "").strip())
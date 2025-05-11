from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# 모델 및 adapter 경로 설정
base_model_id = "mistralai/Mistral-7B-Instruct-v0.3"
adapter_path = "./outputs"  # 학습된 LoRA adapter 경로

# 모델 로드
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_path)

# 테스트 문장 리스트
test_inputs = [
    "첫 등원에 읽기 좋은 놀이책을 찾고 있어요.",
    "동물을 배울 수 있는 책 있으면 알려주세요.",
    "놀이책 중에서 4-6세세가 좋아할 만한 책 있을까요?",
    "2-4세세 아이랑 읽기 좋은 자연 책 추천해 주세요.",
    "생일에 읽기 좋은 놀이책을 찾고 있어요."
]

# 추론 및 출력
for i, user_input in enumerate(test_inputs, 1):
    print(f"👤 입력 {i}: {user_input}")
    prompt = f"### Instruction:\n다음 문장을 분석하여 도서 추천 조건을 추출하세요.\n\n### Input:\n{user_input}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.replace(prompt, "").strip()
    print(f"🤖 출력 {i}:
{answer}
{'-'*50}")

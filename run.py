from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

model_id = "psymon/KoLlama2-7b"
hf_token = "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"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", use_auth_token=hf_token)

prompt = "안녕하세요, 반갑습니다"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(result)
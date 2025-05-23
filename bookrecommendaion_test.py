import json
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import requests

# 1. 튜닝된 모델 로딩
def load_model():
    base = AutoModelForCausalLM.from_pretrained(
        "davidkim205/komt-mistral-7b-v1",
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True)
    )
    model = PeftModel.from_pretrained(base, "JungWook0116/ko-mistral7b-v1-book-recommender")
    tokenizer = AutoTokenizer.from_pretrained("davidkim205/komt-mistral-7b-v1")
    return model.eval(), tokenizer

# 2. 조건 추출 함수
def extract_conditions(model, tokenizer, sentence: str):
    prompt = f"""### Instruction:
다음 문장을 분석하여 도서 추천 조건을 추출하세요.

### Input:
{sentence}

### Output:"""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=50, do_sample=False)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    parsed = decoded.split("### Output:")[-1].strip()
    return parse_extracted(parsed)

# 3. 추출된 문자열 파싱
def parse_extracted(text):
    result = {"theme": [], "type": None, "age": None}
    for part in text.split(","):
        part = part.strip()
        if part.startswith("theme="):
            result["theme"] = [t.strip() for t in part[len("theme="):].split()]  # 공백 기준 split
        elif part.startswith("type="):
            result["type"] = part[len("type="):].strip()
        elif part.startswith("age="):
            result["age"] = part[len("age="):].strip()
    return result

# 4. 도서 필터링
def filter_books(books, extracted):
    def match(book):
        return (
            any(t in book["theme"] for t in extracted["theme"])
            and extracted["type"] in book["types"]
            and any(a in book["age"] for a in extracted["age"].split("-"))
        )
    return [b for b in books if match(b)]

# 5. FAISS 유사도 검색
def build_faiss_index(texts, model):
    vectors = model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, vectors

def recommend_books(input_sentence, books, sbert_model, model, tokenizer, top_k=5):
    # 조건 추출
    extracted = extract_conditions(model, tokenizer, input_sentence)
    candidates = filter_books(books, extracted)
    
    if not candidates:
        return []

    candidate_texts = [
        f"theme={' '.join(b['theme'])}, type={' '.join(b['types'])}, age={b['age']}" for b in candidates
    ]
    index, _ = build_faiss_index(candidate_texts, sbert_model)

    query = f"theme={' '.join(extracted['theme'])}, type={extracted['type']}, age={extracted['age']}"
    query_vec = sbert_model.encode([query])
    _, idxs = index.search(query_vec, top_k)
    return [candidates[i] for i in idxs[0]]

if __name__ == "__main__":
    print("📦 모델 및 데이터 로딩 중...")
    url = "https://raw.githubusercontent.com/jungwook-student/hf_transformers_model/main/aladin_fully_enriched_upto_now_552.json"
    response = requests.get(url)
    books = json.loads(response.text)

    sbert_model = SentenceTransformer("jhgan/ko-sbert-nli")
    model, tokenizer = load_model()

    # 테스트 입력
    user_input = "3세 남자아이에게 읽어줄 창의력을 키워줄 그림책을 추천해줘"
    results = recommend_books(user_input, books, sbert_model, model, tokenizer, top_k=5)

    print("\n🔍 추천 도서 결과:")
    for b in results:
        print(f"- {b['title']} ({b['age']}, {b['types']})")

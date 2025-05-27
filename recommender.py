import json
import torch
import faiss
import requests
import difflib
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
from peft import PeftModel

# ✅ 모델 로딩
def load_models():
    print("📦 모델 로딩 중...")
    base = AutoModelForCausalLM.from_pretrained(
        "davidkim205/komt-mistral-7b-v1",
        device_map="auto",
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    model = PeftModel.from_pretrained(base, "JungWook0116/ko-mistral7b-v1-book-recommender")
    tokenizer = AutoTokenizer.from_pretrained("davidkim205/komt-mistral-7b-v1")
    sbert = SentenceTransformer("intfloat/multilingual-e5-base")
    return model.eval(), tokenizer, sbert

# ✅ 조건 추출
def extract_conditions(model, tokenizer, sentence: str):
    prompt = f"""### Instruction:
다음 문장을 분석하여 도서 추천 조건을 추출하세요.

### Input:
{sentence}

### Output:"""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=50, do_sample=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return parse_extracted(decoded.split("### Output:")[-1].strip())

# ✅ 파싱 함수
def parse_extracted(text):
    result = {"theme": [], "type": None, "age": None}
    for part in text.split(","):
        part = part.strip()
        if part.startswith("theme="):
            result["theme"] = [t.strip() for t in part[len("theme="):].split()]
        elif part.startswith("type="):
            result["type"] = part[len("type="):].strip()
        elif part.startswith("age="):
            result["age"] = part[len("age="):].strip()
    return result

# ✅ 추천 로직
def recommend_books(input_sentence, books, sbert_model, model, tokenizer, top_k=5):
    print(f"📨 추천 요청 수신: {input_sentence}")
    extracted = extract_conditions(model, tokenizer, input_sentence)
    print(f"🎯 추출된 조건: {extracted}")

    candidates = []
    for book in books:
        if "theme" not in book or "types" not in book or "age" not in book:
            continue

        theme_score = max(
            [difflib.SequenceMatcher(None, t, bt).ratio()
             for t in extracted["theme"] for bt in book["theme"]]
        ) if extracted["theme"] else 0.0

        type_score = max(
            [difflib.SequenceMatcher(None, extracted["type"], bt).ratio()
             for bt in book["types"]]
        ) if extracted["type"] else 0.0

        try:
            user_age = int(re.findall(r'\d+', extracted["age"])[0])
            book_ages = [int(x) for x in re.findall(r'\d+', book["age"])]
            age_score = 1.0 if user_age in book_ages else 0.0
        except:
            age_score = 0.0

        final_score = (theme_score + type_score + age_score) / 3
        if final_score > 0:
            candidates.append((final_score, book))

    candidates.sort(reverse=True, key=lambda x: x[0])
    filtered_books = [b for _, b in candidates]
    print(f"✅ 스코어링된 도서 수: {len(filtered_books)}")

    if not filtered_books:
        return []

    texts = [
        f"query: theme={' '.join(b['theme'])}, type={' '.join(b['types'])}, age={b['age']}"
        for b in filtered_books
    ]
    query = f"query: theme={' '.join(extracted['theme'])}, type={extracted['type']}, age={extracted['age']}"

    print("🔍 문장 임베딩 및 유사도 계산 중...")
    query_vec = sbert_model.encode([query], convert_to_tensor=True).cpu()
    corpus_embs = sbert_model.encode(texts, convert_to_tensor=True).cpu()

    scores = util.cos_sim(query_vec, corpus_embs)[0]
    top_indices = torch.topk(scores, k=min(top_k, len(filtered_books))).indices.tolist()
    top_books = [filtered_books[i] for i in top_indices]

    print(f"📚 추천 완료: {len(top_books)}권")
    return top_books

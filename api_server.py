from fastapi import FastAPI, Request
from pydantic import BaseModel
from recommender import load_models, recommend_books
import requests
import time

app = FastAPI()

# ✅ 모델, 도서 데이터 미리 로딩
print("📦 모델 및 도서 데이터 로딩 중...")
start_time = time.time()
model, tokenizer, sbert = load_models()
books_url = "https://raw.githubusercontent.com/jungwook-student/hf_transformers_model/main/aladin_fully_enriched_upto_now_552.json"
books = requests.get(books_url).json()
books = [b for b in books if "theme" in b and "types" in b and "age" in b]
print(f"✅ 초기화 완료: {len(books)}권 로드됨, ⏱️ 소요 시간: {time.time() - start_time:.2f}초")

class UserInput(BaseModel):
    user_input: str

@app.post("/recommend")
def recommend(user: UserInput):
    print("📨 추천 요청 수신:", user.user_input)
    results = recommend_books(user.user_input, books, sbert, model, tokenizer)
    print("🔚 추천 결과 수:", len(results))
    return {"recommendations": results}



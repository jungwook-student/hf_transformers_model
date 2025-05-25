import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch
import json
import requests

# ✅ 모델 로딩
@st.cache_resource
def load_models():
    adapter_model_id = "JungWook0116/ko-mistral7b-v1-book-recommender"
    base = AutoModelForCausalLM.from_pretrained(
        adapter_model_id, torch_dtype=torch.float16, device_map="auto", use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_model_id)
    sbert = SentenceTransformer("intfloat/multilingual-e5-base")
    return base, tokenizer, sbert

# ✅ 도서 데이터 로딩
@st.cache_data
def load_books():
    url = "https://raw.githubusercontent.com/jungwook-student/hf_transformers_model/main/aladin_fully_enriched_upto_now_552.json"
    data = requests.get(url).json()
    return [book for book in data if "theme" in book and "type" in book and "age" in book]

# ✅ 조건 추출 함수
def extract_conditions(prompt, model, tokenizer):
    input_text = f"### Instruction:\n다음 문장을 분석하여 도서 추천 조건을 추출하세요.\n\n### Input:\n{prompt}\n\n### Output:\n"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=50)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    extracted = decoded.split("### Output:")[-1].strip()
    return extracted

# ✅ 유사도 기반 추천
def recommend_books(prompt, model, tokenizer, sbert, books):
    extracted = extract_conditions(prompt, model, tokenizer)
    st.markdown(f"🎯 **추출된 조건**: `{extracted}`")

    query_emb = sbert.encode(extracted, convert_to_tensor=True)
    texts = [
        f"theme={' '.join(book['theme'])}, type={' '.join(book['type'])}, age={book['age']}"
        for book in books
    ]
    corpus_embs = sbert.encode(texts, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, corpus_embs)[0]
    top_k = torch.topk(scores, k=5)

    st.markdown("🔍 **추천 도서 결과:**")
    for idx in top_k.indices.tolist():
        b = books[idx]
        st.write(f"- **{b['title']}** ({b['age']}, {b['type']})")

# ✅ Streamlit UI 구성
st.title("📚 유아 도서 추천기")

with st.spinner("초기 로딩 중입니다. 모델과 데이터를 불러오고 있습니다..."):
    model, tokenizer, sbert = load_models()
    books = load_books()

st.success("✅ 초기화 완료! 이제 문장을 입력해보세요.")
user_input = st.text_input("추천을 위한 문장을 입력하세요:")

if user_input:
    with st.spinner("추천 중..."):
        recommend_books(user_input, model, tokenizer, sbert, books)

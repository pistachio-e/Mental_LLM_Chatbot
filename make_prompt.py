import os
import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# === مسیر فایل‌ها ===
file_path = r"F:\code_payanname\dataset\no_llm_dataset.xlsx"
index_path = "faiss_index_combined.index"
meta_path = "metadata.pkl"

# === مدل embedding ===
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# === ساخت یا بازسازی ایندکس ===
def build_index(alpha=0.3, beta=0.7, force_rebuild=False):
    global index, df
    if force_rebuild or not (os.path.exists(index_path) and os.path.exists(meta_path)):
        print(f"🔄 ساخت ایندکس جدید با alpha={alpha}, beta={beta}...")
        df = pd.read_excel(file_path)
        df = df.drop_duplicates(subset=["Context", "Response"]).reset_index(drop=True)
        q_embeddings = embed_model.encode(df["Context"].tolist(), convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        r_embeddings = embed_model.encode(df["Response"].tolist(), convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        combined_embeddings = alpha * q_embeddings + beta * r_embeddings
        dim = combined_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(combined_embeddings)
        faiss.write_index(index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(df, f)
        print("✅ ایندکس ساخته شد و دیتافریم یکتا است.")
    else:
        print("📂 ایندکس موجود است. بارگذاری از فایل...")
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            df = pickle.load(f)
        print("✅ ایندکس و دیتافریم لود شدند.")

# === تابع جستجوی Hybrid RAG با فیلتر و BM25 بعد FAISS ===
def search_hybrid_rag_final(query, top_k=1, candidate_pool=5, alpha_faiss=0.6, beta_q=0.2, gamma_r=0.2):
    # embedding query
    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    
    # FAISS search برای candidate pool بزرگ‌تر
    D, I = index.search(q_emb, k=min(candidate_pool*2, len(df)))

    # جمع آوری candidates و حذف تکراری
    seen_texts = set()
    candidates = []
    for i, faiss_score in zip(I[0], D[0]):
        context = df.iloc[i]["Context"].strip()
        response = df.iloc[i]["Response"].strip()
        key = (context, response)
        if key in seen_texts:
            continue
        seen_texts.add(key)
        candidates.append({
            "index": i,
            "Context": context,
            "Response": response,
            "FAISS_Score": faiss_score
        })
        if len(candidates) >= candidate_pool:
            break

    # BM25 روی candidates
    tokenized_query = query.split()
    tokenized_context = [c["Context"].split() for c in candidates]
    tokenized_response = [c["Response"].split() for c in candidates]
    bm25_context = BM25Okapi(tokenized_context)
    bm25_response = BM25Okapi(tokenized_response)
    bm25_scores_context = bm25_context.get_scores(tokenized_query)
    bm25_scores_response = bm25_response.get_scores(tokenized_query)

    # محاسبه final score
    for idx, c in enumerate(candidates):
        c["BM25_Context_Score"] = bm25_scores_context[idx]
        c["BM25_Response_Score"] = bm25_scores_response[idx]
        c["Score"] = alpha_faiss * c["FAISS_Score"] + beta_q * c["BM25_Context_Score"] + gamma_r * c["BM25_Response_Score"]

    # مرتب‌سازی بر اساس final score و top_k
    results = sorted(candidates, key=lambda x: x["Score"], reverse=True)[:top_k]
    return results

# === اجرای اصلی ===
if __name__ == "__main__":
    # ساخت یا لود ایندکس
    build_index(alpha=0.3, beta=0.7, force_rebuild=False)

    # پرسش کاربر
    user_query = "hello, i have trauma can you help me ?"

    # جستجوی Hybrid RAG و گرفتن بهترین پاسخ
    results = search_hybrid_rag_final(user_query, top_k=1, candidate_pool=5, alpha_faiss=0.5, beta_q=0.2, gamma_r=0.3)
    best_response = results[0]["Response"]

    # ساخت پرامپت برای LLM
    prompt = f"""### Instruction:
You are a helpful therapist.

Here is a relevant past response:
"{best_response}"

Now, a new user says:
"{user_query}"

Respond with empathy and practical advice.

### Response:
"""

    print("\n📝 پرامپت ساخته شده:")
    print(prompt)

    # نمایش جزئیات بهترین نتیجه
    print("\n🔎 بهترین نتیجه Hybrid RAG:")
    r = results[0]
    print(f"Q: {r['Context']}")
    print(f"A: {r['Response']}")
    print(f"FAISS Score: {r['FAISS_Score']:.4f}")
    print(f"BM25 Context Score: {r['BM25_Context_Score']:.4f}")
    print(f"BM25 Response Score: {r['BM25_Response_Score']:.4f}")
    print(f"Final Score: {r['Score']:.4f}")
    print("-" * 50)

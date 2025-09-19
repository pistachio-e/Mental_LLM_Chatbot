import os
import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import requests
import time

# === مسیر فایل‌ها ===
rag_file_path = r"F:\code_payanname\dataset\no_llm_dataset_new.xlsx"
index_path = "faiss_index_combined.index"
meta_path = "metadata.pkl"
results_excel_path = r"f:\code_payanname_2\llm_vs_rag_results.xlsx"

# === مدل embedding ===
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# === ساخت یا بازسازی ایندکس ===
def build_index(alpha=0.3, beta=0.7, force_rebuild=False):
    global index, df
    if force_rebuild or not (os.path.exists(index_path) and os.path.exists(meta_path)):
        print(f"🔄 ساخت ایندکس جدید با alpha={alpha}, beta={beta}...")
        df = pd.read_excel(rag_file_path)
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

# === تابع جستجوی Hybrid RAG ===
def search_hybrid_rag_final(query, top_k=1, candidate_pool=5, alpha_faiss=0.6, beta_q=0.2, gamma_r=0.2):
    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, k=min(candidate_pool*2, len(df)))

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

    tokenized_query = query.split()
    tokenized_context = [c["Context"].split() for c in candidates]
    tokenized_response = [c["Response"].split() for c in candidates]
    bm25_context = BM25Okapi(tokenized_context)
    bm25_response = BM25Okapi(tokenized_response)
    bm25_scores_context = bm25_context.get_scores(tokenized_query)
    bm25_scores_response = bm25_response.get_scores(tokenized_query)

    for idx, c in enumerate(candidates):
        c["BM25_Context_Score"] = bm25_scores_context[idx]
        c["BM25_Response_Score"] = bm25_scores_response[idx]
        c["Score"] = alpha_faiss * c["FAISS_Score"] + beta_q * c["BM25_Context_Score"] + gamma_r * c["BM25_Response_Score"]

    results = sorted(candidates, key=lambda x: x["Score"], reverse=True)[:top_k]
    return results

# === تابع برای LLM call ===
API_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_NAME = "llama-3.2-3b-instruct"

def call_llm(prompt):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful therapist."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    try:
        r = requests.post(API_URL, json=payload, timeout=60)
        r.raise_for_status()
        obj = r.json()
        content = obj["choices"][0]["message"]["content"] if "choices" in obj and obj["choices"] else obj.get("text","")
        if isinstance(content, dict) and "content" in content:
            return content["content"]
        return content
    except Exception as e:
        return f"❌ LLM call failed: {repr(e)}"

# === تابع محاسبه شباهت معنایی (embedding cosine) ===
def compute_similarity(a, b):
    emb = embed_model.encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
    return cosine_similarity([emb[0]], [emb[1]])[0][0]

# === اجرای اصلی روی ۱۰ ردیف ===
if __name__ == "__main__":
    build_index(alpha=0.3, beta=0.7, force_rebuild=True)
    df_eval = pd.read_excel(rag_file_path)

    results = []

    chosen_indices = np.random.choice(len(df_eval), size=10, replace=False)

    for row_idx in chosen_indices:
        context = df_eval.iloc[row_idx]["Context"]
        ground_truth = df_eval.iloc[row_idx]["Response"]

        print(f"\n📌 Evaluation row index: {row_idx}")
        print(f"👤 Input context: {context}")
        print(f"✅ Ground truth response: {ground_truth}\n")

        # ===== حالت 1: Direct LLM =====
        direct_prompt = f'User: "{context}"'
        direct_answer = call_llm(direct_prompt)
        sim_direct = compute_similarity(direct_answer, ground_truth)

        # ===== حالت 2: RAG + LLM =====
        rag_candidates = search_hybrid_rag_final(context, top_k=1, candidate_pool=5, alpha_faiss=0.5, beta_q=0.2, gamma_r=0.3)
        best_response = rag_candidates[0]["Response"]
        rag_prompt = f'User: "{context}"\nReference (previous similar answer): "{best_response}"\nAnswer the user by following the style and empathy of the reference.'
        rag_answer = call_llm(rag_prompt)
        sim_rag = compute_similarity(rag_answer, ground_truth)

        print(f"🔹 Direct similarity: {sim_direct:.4f}")
        print(f"🔹 RAG similarity: {sim_rag:.4f}\n")
        
        # --- اضافه کردن delay ---
        print("⏳ Waiting 30 seconds before next query...")
        time.sleep(30)

        results.append({
            "Row": row_idx,
            "Context": context,
            "GroundTruth": ground_truth,
            "Direct_Prompt": direct_prompt,
            "Direct_Answer": direct_answer,
            "Direct_Similarity": sim_direct,
            "RAG_Prompt": rag_prompt,
            "RAG_Answer": rag_answer,
            "RAG_Similarity": sim_rag
        })

    # ساخت جدول مقایسه
    df_results = pd.DataFrame(results)
    print("\n📊 Final Comparison Table:")
    print(df_results[["Row", "Direct_Similarity", "RAG_Similarity"]])

    # ذخیره به اکسل برای بررسی توسط LLM-as-a-Judge
    df_results.to_excel(results_excel_path, index=False, engine="openpyxl")
    print(f"\n📂 نتایج در فایل ذخیره شد: {results_excel_path}")

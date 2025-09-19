import tkinter as tk
from tkinter import scrolledtext
import sounddevice as sd
import numpy as np
import tempfile
import scipy.io.wavfile as wav
import threading, queue, requests, re, os, pickle
from io import BytesIO
import torch
from scipy.signal import resample_poly

# ===== Whisper =====
import whisper

# ===== TTS =====
from transformers import pipeline

# ===== Setup =====
API_URL = "http://127.0.0.1:5000/v1/chat/completions"
MODEL_NAME = "llama-3.2-3b-instruct"

SAMPLERATE = 16000
CHANNELS = 1
DURATION = 5
OUTPUT_SR = 16000

EMBEDDINGS_PATH = "qna_embeddings.pkl"  # pickle شامل df + q_embeddings + r_embeddings

# ======================== LLM local ========================
from langchain.llms.base import LLM
from pydantic import BaseModel
from typing import Optional, List

class LocalHTTPLLM(LLM, BaseModel):
    api_url: str
    model_name: str
    temperature: float = 0.7

    @property
    def _llm_type(self) -> str:
        return "local-http"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful therapist."},
                {"role": "user", "content": prompt}
            ],
            "temperature": float(self.temperature),
            "stream": False
        }
        try:
            r = requests.post(self.api_url, json=payload, timeout=60)
            r.raise_for_status()
            obj = r.json()
            content = obj["choices"][0]["message"]["content"] if "choices" in obj and obj["choices"] else obj.get("text","")
            if isinstance(content, dict) and "content" in content:
                return content["content"]
            return content
        except Exception as e:
            return f"❌ LLM call failed: {repr(e)}"

local_llm = LocalHTTPLLM(api_url=API_URL, model_name=MODEL_NAME, temperature=0.7)

# ======================== GUI ========================
root = tk.Tk()
root.title("🎙 Whisper → RAG → LLM → TTS")

chat_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=25, font=("Arial", 11))
chat_box.pack(padx=10, pady=10)

def gui_append(text):
    root.after(0, lambda: (chat_box.insert(tk.END, text), chat_box.yview(tk.END)))

# ======================== Load models ========================
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base", device=torch_device)

hf_device = 0 if torch.cuda.is_available() else -1
tts_model = pipeline("text-to-speech", model="facebook/mms-tts-eng", device=hf_device)

# ======================== Load embeddings for RAG ========================
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

if not os.path.exists(EMBEDDINGS_PATH):
    raise FileNotFoundError("❌ فایل qna_embeddings.pkl پیدا نشد. اول اسکریپت ساخت embedding‌ها را اجرا کنید.")

with open(EMBEDDINGS_PATH, "rb") as f:
    data = pickle.load(f)

df = data["df"]
q_embeddings = data["q_embeddings"]
r_embeddings = data["r_embeddings"]

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
alpha, beta = 0.6, 0.4





### main ###

def build_prompt_with_context(user_query):
    query_emb = embed_model.encode([user_query]).astype("float32")
    sim_q = cosine_similarity(query_emb, q_embeddings)[0]
    sim_r = cosine_similarity(query_emb, r_embeddings)[0]
    scores = alpha * sim_q + beta * sim_r
    best_idx = np.argmax(scores)
    best_response = df["Response"].iloc[best_idx].strip()

    prompt = f"""### Instruction:
You are a helpful therapist.

Here is a relevant past response:
"{best_response}"

Now, a new user says:
"{user_query}"

Respond with empathy and practical advice.

## Response:
"""
    return prompt 

# ======================== chunk for TTS ========================
_SENTENCE_END = re.compile(r"([.!?,;:\n])")

def pop_complete_chunk(buffer, min_len=30, max_len=140):
    for match in _SENTENCE_END.finditer(buffer):
        end_idx = match.end()
        if end_idx >= min_len:
            if end_idx > max_len:
                space_idx = buffer.rfind(" ", min_len, max_len)
                if space_idx == -1:
                    space_idx = max_len
                return buffer[:space_idx].strip(), buffer[space_idx:].strip()
            return buffer[:end_idx].strip(), buffer[end_idx:].strip()
    if len(buffer) >= max_len:
        space_idx = buffer.rfind(" ", int(max_len*0.6), max_len)
        if space_idx == -1:
            space_idx = max_len
        return buffer[:space_idx].strip(), buffer[space_idx:].strip()
    return None, buffer

# ======================== TTS ========================
class TTSSpeaker:
    def __init__(self, tts_pipeline, out_sr=OUTPUT_SR):
        self.tts = tts_pipeline
        self.out_sr = out_sr
        self.text_q = queue.Queue()
        self.audio_q = queue.Queue()
        self._stop = threading.Event()
        self.tts_worker = threading.Thread(target=self._tts_loop, daemon=True)
        self.tts_worker.start()
        self.player_worker = threading.Thread(target=self._player_loop, daemon=True)
        self.player_worker.start()

    def _extract_audio_and_sr(self, output):
        if isinstance(output, dict):
            for k in ("audio", "audio_values", "wav", "waveform"):
                if k in output:
                    data = output[k]
                    sr = output.get("sampling_rate", OUTPUT_SR)
                    return np.asarray(data, dtype=np.float32), int(sr)
        if isinstance(output, np.ndarray):
            return output, OUTPUT_SR
        return None, None

    def _resample_if_needed(self, data, sr):
        if data is None:
            return None, None
        data = np.asarray(data, dtype=np.float32)
        if data.ndim > 1:
            data = data.squeeze()
        if int(sr) != int(self.out_sr):
            from math import gcd
            g = gcd(int(sr), int(self.out_sr))
            up = int(self.out_sr)//g
            down = int(sr)//g
            data = resample_poly(data, up, down).astype(np.float32)
            sr = self.out_sr
        return data, sr

    def _tts_loop(self):
        while not self._stop.is_set():
            try:
                chunk = self.text_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if not chunk or not chunk.strip():
                self.text_q.task_done()
                continue
            output = self.tts(chunk)
            audio_arr, sr = self._extract_audio_and_sr(output)
            audio_arr, sr = self._resample_if_needed(audio_arr, sr)
            if audio_arr is not None:
                self.audio_q.put((audio_arr.copy(), sr))
            self.text_q.task_done()

    def _player_loop(self):
        import sounddevice as sd
        while not self._stop.is_set():
            try:
                audio_arr, sr = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            sd.play(audio_arr, sr, blocking=True)
            self.audio_q.task_done()

    def speak(self, text_chunk):
        self.text_q.put(text_chunk)

    def stop(self):
        self._stop.set()
        try:
            self.text_q.put_nowait("")
            self.audio_q.put_nowait((np.zeros(1, dtype=np.float32), self.out_sr))
        except Exception:
            pass

tts_speaker = TTSSpeaker(tts_model, out_sr=OUTPUT_SR)

# ======================== Stream + Play ========================


#main
def stream_and_play(user_text):
    gui_append("\n🤖 Model: ")
    prompt = build_prompt_with_context(user_text)
    buffer = local_llm(prompt)
    gui_append(buffer)
    tmp_buf = buffer
    while tmp_buf:
        chunk, tmp_buf = pop_complete_chunk(tmp_buf)
        if chunk:
            tts_speaker.speak(chunk)
            
            
            
            
# ======================== Record + Transcribe ========================
def record_and_transcribe():
    gui_append(f"\n🎤 Recording {DURATION} seconds...\n")
    recording = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=CHANNELS)
    sd.wait()
    audio_data = np.clip(recording, -1, 1)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        wav.write(temp_file.name, SAMPLERATE, (audio_data * 32767).astype(np.int16))
        temp_path = temp_file.name
    try:
        result = whisper_model.transcribe(temp_path)
        user_text = result["text"].strip()
        gui_append(f"\n👤 You: {user_text}\n")
    except Exception as e:
        gui_append(f"\n⚠ Whisper error: {e}\n")
        return
    threading.Thread(target=stream_and_play, args=(user_text,), daemon=True).start()

def start_recording():
    threading.Thread(target=record_and_transcribe, daemon=True).start()

# ======================== Button ========================
record_button = tk.Button(root, text="🎙 Record", command=start_recording, font=("Arial", 12))
record_button.pack(pady=5)

root.mainloop()

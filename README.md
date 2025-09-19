# Mental_LLM_Chatbot
# 🎙 Whisper → RAG → LLM → TTS

This project is a simple **GUI application (Tkinter)** that records audio and performs the following pipeline:

1. **Record audio** (using `sounddevice`)
2. **Speech-to-text** via [Whisper](https://github.com/openai/whisper)
3. **Retrieve relevant context** with RAG (Embeddings + Similarity Search)
4. **Generate empathetic responses** from a local LLM served with [LM Studio](https://lmstudio.ai/)
5. **Text-to-Speech (TTS)** using HuggingFace models
6. **Play audio output** inside the application

---

#Datasets
1. https://huggingface.co/datasets/Amod/mental_health_counseling_conversations
2. https://huggingface.co/datasets/ShenLab/MentalChat16K

## 📦 Dependencies

Install required packages:

```bash
pip install tkinter sounddevice numpy scipy torch requests transformers sentence-transformers scikit-learn whisper


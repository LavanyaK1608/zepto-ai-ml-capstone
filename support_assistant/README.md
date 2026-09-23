# Module 3 – Support Assistant (25 marks)

## Purpose
A complete RAG-style GenAI support service for Zepto policies:
document corpus → local embeddings (sentence-transformers) → ChromaDB → LangGraph intent router → FastAPI.

**Graded baseline runs fully offline with `MOCK_LLM=1` (default).**

## Setup
```bash
cd support_assistant
pip install -r requirements.txt
```

## How to run (local)
```bash
# mock mode (default – what gets graded)
uvicorn main:app --host 0.0.0.0 --port 7860

# or
python main.py
```

## Example calls (MOCK_LLM=1)

**Policy question (triggers retrieval):**
```bash
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the delivery fee?"}'
```
Expected shape:
```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials ...",
  "sources": ["doc_01", ...],
  "confidence": 1.0
}
```

**General question (no retrieval):**
```bash
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather today?"}'
```
Expected:
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

## Docker (graded baseline)
```bash
docker build -t zepto-support .
docker run -p 7860:7860 zepto-support
```

## Architecture (RAG pipeline)

1. **Ingestion** – `docs/doc_01.txt` … `doc_08.txt` are loaded by `load_documents()`.
2. **Embedding** – `SentenceTransformer("all-MiniLM-L6-v2")` produces vectors; stored in ChromaDB collection `zepto_policies`.
3. **Retrieval** – `retrieve()` embeds the query and returns top-3 chunks by cosine similarity (always real, no LLM needed).
4. **Generation** – LangGraph nodes:
   - `classify_intent` – keyword heuristic (mock) → routes to
   - `retrieve_and_answer` (policy) or `direct_answer` (general)
   - Generation step inside each node branches on `MOCK_LLM`.

Only the final answer-generation step changes when `MOCK_LLM=0` (optional real LLM). Retrieval and routing stay the same.

## Structured prompt (used only when MOCK_LLM=0)
See `PROMPT_TEMPLATE` in `main.py` – contains Role / Context / Task / Format / Length + negative constraint + few-shot example.

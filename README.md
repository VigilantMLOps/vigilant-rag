# Atlas RAG

A production-grade RAG (Retrieval-Augmented Generation) system that lets you query your knowledge base using local AI models — fully self-hosted, no data leaves your machine.

Integrated with **Atlas MLOps** for full LLM observability: every query traces tokens, latency, and retrieval scores on the monitoring dashboard.

---

## Features

- **Hybrid retrieval** — dense (Ollama embeddings) + sparse (BM25) with RRF fusion
- **Cross-encoder reranking** — reranks candidates before generation for higher accuracy
- **Local models** — `nomic-embed-text` for embeddings, `llama3.2:3b` for generation
- **Live vault sync** — watches your notes directory and re-indexes on file changes
- **LLM observability** — traces pushed to Atlas MLOps (tokens, latency, retrieval scores)
- **Desktop app** — native macOS chat UI

---

## Stack

Python · FastAPI · Qdrant · Ollama · sentence-transformers · fastembed · Docker

---

## Prerequisites

- Docker Desktop running
- [Atlas Pack](https://github.com/VigilantMLOps/vigilant-pack) installed (`pip install vigilantpack`)
- A notes/documents directory (vault)

---

## Setup

**1. Clone and create `.env`**

```bash
git clone https://github.com/VigilantMLOps/vigilant-rag
cd vigilant-rag
cp .env.example .env
```

Edit `.env`:

```env
VAULT_PATH=/path/to/your/notes
VIGILANT_API_URL=https://vigilant-api.duckdns.org   # or your local Atlas MLOps URL
EMIT_TRACES=true
```

**2. Start the stack**

```bash
vigilantpack run
```

This starts Qdrant, pulls and warms up Ollama models, then starts the API. First run downloads ~2 GB of models.

**3. Query**

```bash
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are my open tasks?", "mode": "factual", "top_k": 5}'
```

Or open the Swagger docs: `http://localhost:8080/docs`

---

## Desktop App (macOS)

**First time setup:**

```bash
pip install customtkinter Pillow
python make_icon.py
./build_app.sh
```

`Atlas RAG.app` will appear on your Desktop. Double-click it — the app starts the full stack automatically, then opens the chat UI when ready.

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/query` | Query the knowledge base (blocking) |
| `POST` | `/api/v1/query/stream` | Query with streaming SSE response |
| `POST` | `/api/v1/ingest` | Manually trigger re-indexing |
| `GET` | `/health` | Health check |

### Query request

```json
{
  "query": "Your question here",
  "mode": "factual",
  "top_k": 5,
  "filters": {
    "tags": [],
    "has_tasks": null
  }
}
```

### Query response

```json
{
  "answer": "...",
  "sources": [
    { "title": "Note title", "file_path": "...", "excerpt": "...", "score": 0.91 }
  ],
  "trace_id": "...",
  "retrieval_latency_ms": 120,
  "generation_latency_ms": 3400,
  "total_latency_ms": 3520
}
```

---

## Stop

```bash
vigilantpack stop
```

---

## Contact

**Bara Al-Sedih** — [github.com/baraalsedih](https://github.com/baraalsedih) · baraalsedih@gmail.com

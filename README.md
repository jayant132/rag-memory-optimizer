# rag-memory-optimizer

> A memory-efficient, token-budgeted RAG backend: Qdrant HNSW + 8-bit scalar quantization, cross-encoder reranking, and hard token-budget context packing.

**Stack:** Python · FastAPI · Qdrant · Ollama · sentence-transformers

---

## Overview

Most RAG demos work on five documents. At real scale, three things hurt first:

| Problem | What goes wrong | How this project handles it |
|---|---|---|
| Vector RAM cost | `float32` embeddings are expensive to keep in memory | HNSW index with int8 scalar quantization (SQ8), ~4x smaller vectors in RAM |
| Noisy retrieval | Embedding similarity alone returns loosely related chunks | Cross-encoder reranking scores each query-chunk pair |
| Token spend | Dumping every candidate into the prompt inflates cost | Token-budget packer fits the best chunks under a hard limit |

Every response also returns telemetry (candidates evaluated, chunks packed, tokens used vs. budget).

---

## Architecture

```text
INGESTION (offline)

  Markdown docs --> Recursive chunker --> Async embedder --> Qdrant
  data/sample_docs  structure-aware       Ollama             HNSW + SQ8
                                          nomic-embed-text   int8 in RAM
                                                             float32 on disk


QUERY (online)  POST /query

  User query
      |
      v
  1. Embed query ............ Ollama, nomic-embed-text (768-d)
      |
      v
  2. Dense ANN search ....... Qdrant HNSW over int8 vectors  -> top_k_candidates
      |
      v
  3. Cross-encoder rerank ... BAAI/bge-reranker-base         -> final_top_k
      |
      v
  4. Token-budget packing ... fill context up to max_token_budget
      |
      v
  5. JSON response .......... packed_context + sources + telemetry
```

| Step | File | Responsibility |
|---|---|---|
| 1 | `src/ingestion/embedder.py` | Async query embedding via Ollama |
| 2 | `src/retrieval/vector_store.py` | Qdrant search over the quantized HNSW index |
| 3 | `src/retrieval/rerank.py` | Cross-encoder re-scoring, keeps top results |
| 4 | `src/retrieval/rerank.py` | Packs chunks in rank order until the budget is hit |
| 5 | `src/evaluation/evaluator.py` | Computes telemetry for the response |

---

## Design Decisions

### Memory: HNSW + SQ8

HNSW gives approximate nearest-neighbor search instead of an O(N) exact scan. SQ8 stores each dimension as `int8` instead of `float32`:

| | Per vector (768-d) | 1M vectors |
|---|---|---|
| `float32` | 3,072 B | ~3.1 GB |
| `int8` (SQ8) | 768 B | ~0.77 GB |

Quantized vectors stay in RAM (`always_ram=True`) for fast graph traversal, and the original `float32` vectors are kept on disk (`on_disk=True`). Trade-off: quantization can slightly reduce recall, which is why a benchmark is on the roadmap.

### Precision: two-stage retrieval

Fast vector search casts a wide net, then a slower cross-encoder scores each query-chunk pair jointly. The extra compute runs only on a small candidate set, so latency stays bounded.

### Cost: hard token budget

Chunks are packed in rank order and packing stops at `max_token_budget`, so prompt size stays predictable. Token count is estimated at ~4 characters per token (a heuristic, not tokenizer-exact).

### Concurrency

Async FastAPI with `httpx` and `asyncio.gather` keeps embedding calls non-blocking.

---

## Quick Start

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com) running locally, and a Qdrant Cloud cluster (or a local Qdrant instance).

```bash
ollama pull nomic-embed-text
```

### 1. Install

```bash
git clone https://github.com/jayant132/rag-memory-optimizer.git
cd rag-memory-optimizer

python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn qdrant-client httpx langchain-text-splitters python-dotenv sentence-transformers
```

### 2. Configure

```bash
cp .env.example .env
```

| Variable | Description | Example |
|---|---|---|
| `QDRANT_URL` | Qdrant cluster endpoint | `https://your-cluster.aws.cloud.qdrant.io` |
| `QDRANT_API_KEY` | Qdrant API key | `your_secure_api_key` |
| `EMBEDDING_MODEL_NAME` | Ollama embedding model | `nomic-embed-text` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |

### 3. Ingest documents

```bash
export PYTHONPATH=$PYTHONPATH:.
python3 -m src.ingestion.pipeline
```

### 4. Start the API

```bash
export PYTHONPATH=$PYTHONPATH:.
./venv/bin/python -m uvicorn src.api.main:app --port 8000
```

Interactive docs: http://localhost:8000/docs

---

## API Reference

### `GET /system/status`

Liveness check. Returns the service name, vector store, and reranker model.

### `POST /query`

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | The user's question |
| `top_k_candidates` | int | `10` | Candidates fetched from Qdrant |
| `final_top_k` | int | `2` | Chunks kept after reranking |
| `max_token_budget` | int | `1000` | Upper bound on packed context size |

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the remote work rules for data storage?",
    "top_k_candidates": 4,
    "final_top_k": 2,
    "max_token_budget": 500
  }'
```

Example response (from a real run, context shortened):

```json
{
  "query": "What are the remote work rules for data storage?",
  "telemetry": {
    "candidates_evaluated": 4,
    "final_chunks_packed": 2,
    "estimated_input_tokens": 152,
    "token_budget_limit": 500,
    "retrieval_precision_score": 0.01045
  },
  "packed_context": "## 1. Data Governance & Storage\n1.1 All proprietary source code ... must remain strictly within approved company cloud environments.\n1.2 Local caching of unencrypted enterprise data on personal devices (BYOD) is strictly prohibited ...",
  "retrieved_sources": [
    "data/sample_docs/enterprise_security_policy.md",
    "data/sample_docs/enterprise_security_policy.md"
  ]
}
```

| Telemetry field | Meaning |
|---|---|
| `candidates_evaluated` | Chunks returned by the vector search |
| `final_chunks_packed` | Chunks kept after reranking |
| `estimated_input_tokens` | Approximate tokens in `packed_context` (chars / 4) |
| `token_budget_limit` | The requested `max_token_budget` |
| `retrieval_precision_score` | Mean raw cross-encoder score of kept chunks. A heuristic relevance signal, not labeled precision; its scale depends on the reranker output range |

---

## Project Structure

```text
rag-memory-optimizer/
├── config/
│   └── settings.yaml        # System parameters
├── data/
│   └── sample_docs/         # Source documents (markdown)
├── src/
│   ├── api/
│   │   └── main.py          # FastAPI app and /query pipeline
│   ├── core/
│   │   └── config.py        # Environment configuration
│   ├── evaluation/
│   │   └── evaluator.py     # Retrieval and token telemetry
│   ├── ingestion/
│   │   ├── chunker.py       # Recursive markdown splitter
│   │   ├── embedder.py      # Async Ollama embedding client
│   │   └── pipeline.py      # Ingestion orchestrator
│   └── retrieval/
│       ├── rerank.py        # Cross-encoder reranker + token-budget packer
│       └── vector_store.py  # Qdrant client and SQ8 collection setup
├── tests/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## Limitations & Roadmap

This is a proof of concept on a small corpus. It shows the pipeline works end to end, not production-scale performance.

**Current limitations**

- Tested on a single small policy document; no large-corpus benchmark yet.
- The ~4x memory saving comes from SQ8 data-type math, not a measured benchmark; recall impact is not yet quantified.
- `retrieval_precision_score` is a heuristic, not a labeled metric.
- Token counts are estimated, not tokenizer-exact.
- Requires a local Ollama instance for embeddings.

## Author

**Jayant**: [GitHub](https://github.com/jayant132) · [LinkedIn](https://www.linkedin.com/in/your-profile)

# rag-memory-optimizer

A memory-efficient RAG backend that keeps three costs under control: **vector memory**, **retrieval noise** and **prompt tokens**.

Qdrant HNSW with selectable quantization, cross-encoder reranking, and hard token-budget context packing, served by an async FastAPI app. Includes a reproducible quantization benchmark.

**Stack:** Python · FastAPI · Qdrant · Ollama · sentence-transformers

## Why this exists

Most RAG demos work on five documents. At real scale three things hurt first:

| Problem | What goes wrong | What this project does |
|---|---|---|
| Vector memory | float32 embeddings are expensive to keep in RAM | HNSW index with quantization (SQ8 by default; TurboQuant and binary benchmarked) |
| Noisy retrieval | Embedding similarity alone returns loosely related chunks | Cross-encoder reranking scores each query-chunk pair |
| Token spend | Dumping every candidate into the prompt inflates cost | A packer fits the best chunks under a hard token limit |

Every response also returns telemetry: candidates evaluated, chunks packed, tokens used vs. budget.

## How it works

```
INGESTION (offline)
  Markdown docs -> recursive chunker -> async embedder (Ollama) -> Qdrant (HNSW + quantization)

QUERY (online)  POST /query
  1. Embed the query           Ollama, nomic-embed-text (768-d)
  2. Dense ANN search          Qdrant HNSW over quantized vectors  -> top_k_candidates
  3. Cross-encoder rerank      BAAI/bge-reranker-base              -> final_top_k
  4. Token-budget packing      fill the context up to max_token_budget
  5. JSON response             packed_context + sources + telemetry
```

Fast vector search casts a wide net, then the slower cross-encoder scores only that small candidate set, so latency stays bounded.

## Quick start

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com), Docker (for a local Qdrant).

```bash
# 1. Services
ollama pull nomic-embed-text
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# 2. Install
git clone https://github.com/jayant132/rag-memory-optimizer.git
cd rag-memory-optimizer
python3 -m venv venv && source venv/bin/activate
pip install fastapi uvicorn qdrant-client httpx langchain-text-splitters \
            python-dotenv sentence-transformers numpy

# 3. Configure (the defaults work for a local setup)
cp .env.example .env

# 4. Ingest the sample documents
export PYTHONPATH=.
python -m src.ingestion.pipeline

# 5. Start the API
uvicorn src.api.main:app --port 8000
```

Interactive docs: <http://localhost:8000/docs>

### Configuration

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint (use your cluster URL for Qdrant Cloud) |
| `QDRANT_API_KEY` | none | Only needed for Qdrant Cloud |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `EMBEDDING_MODEL_NAME` | `nomic-embed-text` | Embedding model |
| `EMBEDDING_DIM` | `768` | Vector size |
| `COLLECTION_NAME` | `rag_memory_index` | Qdrant collection |

## API

### `GET /system/status`
Liveness check. Returns the service name, vector store and reranker model.

### `POST /query`

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | The user's question |
| `top_k_candidates` | int | 10 | Candidates fetched from Qdrant |
| `final_top_k` | int | 2 | Chunks kept after reranking |
| `max_token_budget` | int | 1000 | Upper bound on packed context size |

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

Example response (context shortened):

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
  "packed_context": "## 1. Data Governance & Storage\n1.1 All proprietary source code ...",
  "retrieved_sources": ["data/sample_docs/enterprise_security_policy.md"]
}
```

| Telemetry field | Meaning |
|---|---|
| `candidates_evaluated` | Chunks returned by the vector search |
| `final_chunks_packed` | Chunks kept after reranking |
| `estimated_input_tokens` | Approximate tokens in `packed_context` (characters / 4) |
| `token_budget_limit` | The requested `max_token_budget` |
| `retrieval_precision_score` | Mean raw cross-encoder score of the kept chunks. A relevance signal, not labeled precision; its scale depends on the reranker output |

## Benchmark: what does quantization cost?

The benchmark measures what each quantization setting does to **retrieval quality**, **latency** and **memory** on a public dataset. It is reproducible end to end.

### Setup

- **Dataset:** [BEIR SciFact](https://github.com/beir-cellar/beir): 5,183 documents, 300 test queries with human relevance labels.
- **Embeddings:** `nomic-embed-text` via Ollama (768-d), using the model's `search_document:` and `search_query:` task prefixes. A 256-d variant is built by Matryoshka truncation (layer norm, truncate, L2 normalize).
- **Search settings:** HNSW `ef=128`, rescoring on, 2x oversampling. A second run with 1x oversampling shows how much the rescoring stage contributes.
- **Configurations:** float32 (kept in RAM, the baseline), SQ8, TurboQuant at 4, 2 and 1 bit, and 2-bit binary quantization, at 768 and 256 dimensions.

### Metrics

| Metric | Definition |
|---|---|
| recall@10 | Overlap of the top 10 with **exact float32 search** over the same vectors. It isolates what truncation and quantization cost, independent of the dataset's labels |
| nDCG@10 | Ranking quality against the human relevance labels |
| p50 / p95 latency | Single-query, sequential, after 20 warm-up queries |
| Bytes per vector | **Computed from the storage format**, not measured |
| Process RAM | Measured with `docker stats` on a fresh Qdrant container per configuration (see results) |

### Results

> **Status:** the full-corpus run is in progress. Cells marked `TBD` are filled from `benchmark/results.json` once it completes. No preliminary numbers are reported here, because a run on a partial corpus is not representative.

| Config | Bytes / vector | recall@10 | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| 768-d float32 | 3,072 | TBD | TBD | TBD | TBD |
| 768-d SQ8 | 768 | TBD | TBD | TBD | TBD |
| 768-d TurboQuant 4-bit | 384 | TBD | TBD | TBD | TBD |
| 768-d TurboQuant 2-bit | 192 | TBD | TBD | TBD | TBD |
| 768-d TurboQuant 1-bit | 96 | TBD | TBD | TBD | TBD |
| 768-d binary (2-bit) | 192 | TBD | TBD | TBD | TBD |
| 256-d float32 | 1,024 | TBD | TBD | TBD | TBD |
| 256-d SQ8 | 256 | TBD | TBD | TBD | TBD |
| 256-d TurboQuant 4-bit | 128 | TBD | TBD | TBD | TBD |
| 256-d TurboQuant 2-bit | 64 | TBD | TBD | TBD | TBD |
| 256-d TurboQuant 1-bit | 32 | TBD | TBD | TBD | TBD |

**Measured process RAM (float32 vs. best quantized configuration):** TBD

### How to read these results

- **Rescoring.** Quantized search proposes candidates (2x the requested number) and the original float32 vectors rescore them. The table therefore shows quantization as candidate generation followed by exact ranking. The compressed index is smaller, but the originals must still be stored, on disk if you want to save RAM.
- **Bytes per vector is the compressed vector only.** The HNSW graph and payloads cost extra, which is why process RAM is measured separately.
- **Latency.** At this corpus size every configuration answers in a few milliseconds, and differences between configurations are within run-to-run noise. Use the table for quality and memory, not for speed claims.
- **256-d rows** are only meaningful if exact-search nDCG at 256-d is close to the 768-d value. The benchmark prints both so you can check that the model tolerates truncation.

### Reproduce

```bash
export PYTHONPATH=.

# 1. Embed the corpus once (CPU Ollama: roughly 100 minutes for 5,183 documents;
#    resumes automatically from benchmark/cache if interrupted)
python benchmark/embed_corpus.py --n 5183

# 2. Run the benchmark with nothing else running (latency is CPU-sensitive)
python benchmark/run_benchmark.py                       # 2x oversampling
python benchmark/run_benchmark.py --oversampling 1.0    # quantization alone
```

Results are written to `benchmark/results.json` and printed as a markdown table.

## Design decisions

- **Quantized HNSW.** HNSW gives approximate nearest-neighbor search instead of an O(N) scan. Quantized vectors stay in RAM for fast graph traversal and the original float32 vectors live on disk. SQ8 stores each dimension as one byte instead of four, so a 768-d vector goes from 3,072 B to 768 B (computed; see the benchmark for the measured effect).
- **Two-stage retrieval.** Wide, cheap vector search first; the cross-encoder runs only on that small candidate set.
- **Hard token budget.** Chunks are packed in rank order and packing stops at `max_token_budget`, so prompt size is predictable. Token counts are estimated at about 4 characters per token.
- **Embedding client.** Async and batched with bounded concurrency, and it applies the nomic task prefixes. Omitting them lowers retrieval quality.

## Project structure

```
rag-memory-optimizer/
├── benchmark/
│   ├── embed_corpus.py      # embeds SciFact once and caches the vectors
│   └── run_benchmark.py     # quantization sweep: recall, nDCG, latency
├── config/settings.yaml     # system parameters
├── data/sample_docs/        # source documents (markdown)
├── src/
│   ├── api/main.py          # FastAPI app and /query pipeline
│   ├── core/config.py       # environment configuration
│   ├── evaluation/evaluator.py   # retrieval and token telemetry
│   ├── ingestion/           # chunker, embedder, ingestion pipeline
│   └── retrieval/
│       ├── rerank.py        # cross-encoder reranker + token-budget packer
│       └── vector_store.py  # Qdrant client and quantized collection setup
├── tests/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Limitations and roadmap

This is a proof of concept. It shows the pipeline works end to end and quantifies the quantization trade-off; it is not a production-scale deployment.

- The API was exercised on a single small policy document. Retrieval quality at scale is measured only by the benchmark, on SciFact.
- Memory figures for bytes per vector come from the storage format. Process RAM is measured separately (see results).
- `retrieval_precision_score` is a heuristic, not a labeled metric.
- Token counts are estimated, not tokenizer-exact.
- Requires a running Ollama instance for embeddings.

Planned:

- [ ] Publish the full-corpus benchmark and measured RAM table
- [ ] Tokenizer-exact token counting
- [ ] End-to-end retrieval evaluation of the full pipeline (reranker included) on labeled queries

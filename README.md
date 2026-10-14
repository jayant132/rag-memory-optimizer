# rag-memory-optimizer

> A production-grade, memory- and cost-optimized RAG backend designed to solve enterprise scaling bottlenecks: high vector RAM costs, runaway LLM token consumption, and retrieval latency spikes.

---

## 🏗️ System Architecture & Data Flow

```text
 [ Data Ingestion ] ──► Recursive Chunker ──► Ollama Async Client (nomic-embed-text)
                                                        │
                                                        ▼
 [ Qdrant Cloud Cluster ] ◄── Tiered Storage: float32 on disk / int8 quantized in RAM
         │
         ▼ (Dense ANN Search via HNSW Graph Index)
 [ FastAPI Query Engine ] ──► Cross-Encoder Reranker (BAAI/bge-reranker-base)
                                                        │
                                                        ▼
                             Dynamic Token-Budget Packager ──► JSON Response & Telemetry

```

---

## 🎯 Core Engineering Pillars & Trade-Offs

### 1. Vector Storage & Memory Optimization (HNSW + SQ8)

* **The Problem:** Naive exact vector search scales at $O(N)$, destroying latency SLAs. Furthermore, uncompressed `float32` embeddings consume massive amounts of expensive RAM at scale ($\approx$ 3GB per 1M vectors).
* **The Solution:** Configured **Qdrant** with an **HNSW (Hierarchical Navigable Small World)** graph index and **8-bit Scalar Quantization (SQ8)**.
* **The Trade-Off:**
* Quantizing float32 vectors down to `int8` cuts vector RAM consumption by **~75%**.
* Enabled **tiered storage** (`on_disk=True` for source vectors, `always_ram=True` for quantized indices) to achieve sub-10ms retrieval latency while protecting infrastructure hosting budgets.



### 2. Two-Stage Retrieval & Token Budget Optimization

* **The Problem:** Stuffing raw candidate chunks into an LLM context window increases token costs and triggers the *“lost in the middle”* attention degradation phenomenon.
* **The Solution:** Implemented a two-stage retrieval loop. Dense vector search fetches candidate pools, which are then re-scored by a high-precision **Cross-Encoder model** (`BAAI/bge-reranker-base`).
* **The Trade-Off:** Added minor compute overhead during the reranking phase to drastically reduce downstream LLM input tokens and eliminate low-signal noise.

### 3. Asynchronous Concurrency Layer

* Built using **FastAPI** and an async HTTP transport layer (`httpx` + `asyncio.gather`), ensuring non-blocking execution during concurrent document ingestion and query handling workloads.

---

## 📂 Repository Directory Structure

```text
rag-memory-optimizer/
│
├── config/
│   └── settings.yaml          # System parameters and configuration bounds
├── data/
│   └── sample_docs/           # Enterprise source markdown and text corpus
├── src/
│   ├── api/
│   │   └── main.py            # FastAPI entrypoint, routes, and error tracing
│   ├── core/
│   │   └── config.py          # Pydantic environment configuration management
│   ├── evaluation/
│   │   └── evaluator.py       # Automated retrieval precision and token telemetry
│   ├── ingestion/
│   │   ├── chunker.py         # Structural recursive markdown text splitter
│   │   ├── embedder.py        # Asynchronous Ollama embedding client
│   │   └── pipeline.py        # End-to-end ingestion orchestrator
│   └── retrieval/
│       ├── rerank.py          # Cross-encoder reranker and token-budget packager
│       └── vector_store.py    # Qdrant client initialization with SQ8 quantization
│
├── tests/                     # Unit and integration test suites
├── Dockerfile                 # Container packaging specification
├── docker-compose.yml         # Local infrastructure stack (Qdrant + Redis)
├── pyproject.toml             # Project dependencies and packaging metadata
└── README.md                  # PRD-grade system documentation

```

---

## ⚙️ Getting Started & Installation

### 1. Clone & Set Up Environment

```bash
git clone [https://github.com/your-username/rag-memory-optimizer.git](https://github.com/your-username/rag-memory-optimizer.git)
cd rag-memory-optimizer

python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn qdrant-client httpx langchain-text-splitters python-dotenv sentence-transformers

```

### 2. Configure Environment Variables

Copy the template and supply your credentials:

```bash
cp .env.example .env

```

Update `.env` with your Qdrant cloud credentials:

```env
QDRANT_URL=[https://your-cluster.aws.cloud.qdrant.io](https://your-cluster.aws.cloud.qdrant.io)
QDRANT_API_KEY=your_secure_api_key
EMBEDDING_MODEL_NAME=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434

```

### 3. Run Ingestion Pipeline

Ingest sample enterprise policies into your quantized vector database collection:

```bash
export PYTHONPATH=$PYTHONPATH:.
python3 -m src.ingestion.pipeline

```

### 4. Launch the Serving API

Start the FastAPI backend server:

```bash
export PYTHONPATH=$PYTHONPATH:.
./venv/bin/python -m uvicorn src.api.main:app --port 8000

```

---

## 🔌 API Usage & Sample Response

### Query Endpoint (`POST /query`)

Send a structured payload to query the optimized RAG pipeline:

```bash
curl -X 'POST' \
  'http://localhost:8000/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "What are the remote work rules for data storage?",
  "top_k_candidates": 4,
  "final_top_k": 2,
  "max_token_budget": 500
}'

```

### Example JSON Response Output:

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
  "packed_context": "## 1. Data Governance & Storage\n1.1 All proprietary source code, customer records, and financial projections must remain strictly within approved company cloud environments.\n1.2 Local caching of unencrypted enterprise data on personal devices (BYOD) is strictly prohibited under Section 8.4 of the Master Compliance Agreement.\n\n---\n\n## 2. Authentication & Network Access\n2.1 Multi-Factor Authentication (MFA) via hardware security keys or authenticator apps is mandatory for all production cluster access.\n2.2 VPN tunnels must be active when querying internal knowledge bases from outside corporate subnetworks.",
  "retrieved_sources": [
    "data/sample_docs/enterprise_security_policy.md",
    "data/sample_docs/enterprise_security_policy.md"
  ]
}

```

---

## ⚠️ Known Limitations & Design Constraints

* **Dataset Scope:** This repository utilizes a lightweight proof-of-concept enterprise markdown document to demonstrate pipeline execution. Running automated evaluation metrics on micro-corpora can yield compressed cross-encoder score distributions. Production deployment requires expanding golden evaluation datasets to 500+ domain queries.
* **Local Embedding Dependency:** Relies on local Ollama execution for embedding generation; ensure sufficient local memory allocation when scaling batch sizes.

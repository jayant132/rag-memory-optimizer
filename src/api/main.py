import os
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.retrieval.vector_store import get_vector_store_client
from src.ingestion.embedder import OllamaEmbeddingClient
from src.retrieval.rerank import RetrievalOptimizer
from src.evaluation.evaluator import RAGEvaluator

load_dotenv()

app = FastAPI(
    title="rag-memory-optimizer",
    description="Production-grade RAG backend optimized for memory (SQ8) and token budget allocation.",
    version="1.0.0"
)

# Initialize global clients/optimizers on startup to avoid connection overhead per request
print("[INFO] Initializing system models and database clients...")
qdrant_client = get_vector_store_client()
embedder = OllamaEmbeddingClient(model_name=os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text"))
reranker = RetrievalOptimizer(model_name="BAAI/bge-reranker-base")
collection_name = "rag_memory_index"

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's query text")
    top_k_candidates: int = Field(default=10, description="Initial candidates to fetch from vector DB")
    final_top_k: int = Field(default=2, description="Top chunks to keep after cross-encoder reranking")
    max_token_budget: int = Field(default=1000, description="Maximum token budget for packed context")

@app.get("/system/status")
async def system_status():
    """Unique system liveness and state verification endpoint."""
    return {
        "status": "operational",
        "service": "rag-memory-optimizer",
        "vector_store": "qdrant_cloud_sq8",
        "reranker_model": "BAAI/bge-reranker-base"
    }

@app.post("/query")
async def query_knowledge_base(payload: QueryRequest):
    """
    Executes a multi-stage optimized RAG pipeline:
    1. Vector embedding generation via Ollama.
    2. Dense vector search against Qdrant (SQ8 quantized index).
    3. Cross-encoder reranking for precision filtering.
    4. Dynamic token-budget context packing & evaluation telemetry.
    """
    try:
        # Step 1: Embed incoming user query
        query_vector = await embedder.generate_embedding(payload.query)
        if not query_vector:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding.")

        # Step 2: Retrieve candidate chunks from Qdrant vector database using query_points API
        search_result = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=payload.top_k_candidates
        )
        search_results = search_result.points

        # Map Qdrant scored points into standard dictionary structures
        candidates = []
        for scored_point in search_results:
            candidates.append({
                "id": scored_point.id,
                "score": scored_point.score,
                "text": scored_point.payload.get("text"),
                "source": scored_point.payload.get("source"),
                "domain": scored_point.payload.get("domain")
            })

        if not candidates:
            return {
                "query": payload.query,
                "telemetry": {"candidates_evaluated": 0, "retrieval_precision_score": 0.0},
                "packed_context": "",
                "message": "No relevant documents found in knowledge base."
            }

        # Step 3: Rerank candidates using Cross-Encoder
        reranked_docs = reranker.rerank_results(
            query=payload.query,
            retrieved_docs=candidates,
            top_k=payload.final_top_k
        )

        # Step 4: Pack context strictly within token budget bounds
        packed_context = reranker.pack_context_within_budget(
            reranked_docs=reranked_docs,
            max_tokens=payload.max_token_budget
        )

        # Step 5: Calculate automated evaluation metrics & token usage
        retrieval_precision = RAGEvaluator.evaluate_retrieval_precision(payload.query, reranked_docs)
        estimated_input_tokens = len(packed_context) // 4

        return {
            "query": payload.query,
            "telemetry": {
                "candidates_evaluated": len(candidates),
                "final_chunks_packed": len(reranked_docs),
                "estimated_input_tokens": estimated_input_tokens,
                "token_budget_limit": payload.max_token_budget,
                "retrieval_precision_score": retrieval_precision
            },
            "packed_context": packed_context,
            "retrieved_sources": [doc["source"] for doc in reranked_docs]
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
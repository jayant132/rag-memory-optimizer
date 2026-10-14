import os
import asyncio
import uuid
from dotenv import load_dotenv
from qdrant_client.http import models

from src.retrieval.vector_store import get_vector_store_client, init_optimized_collection
from src.ingestion.chunker import EnterpriseDocumentChunker
from src.ingestion.embedder import OllamaEmbeddingClient

# Load environment configuration variables
load_dotenv()

async def run_ingestion_pipeline() -> None:
    """
    Orchestrates the end-to-end asynchronous ingestion pipeline:
    1. Reads sample enterprise policy document.
    2. Chunks text using recursive structural boundaries.
    3. Generates vector embeddings concurrently via Ollama.
    4. Upserts points into Qdrant collection with SQ8 quantization.
    """
    collection_name = "rag_memory_index"
    file_path = "data/sample_docs/enterprise_security_policy.md"
    
    if not os.path.exists(file_path):
        print(f"[ERROR] Sample file not found at {file_path}. Please create it first.")
        return

    print("[INFO] Initializing Qdrant client and collection schema...")
    client = get_vector_store_client()
    # Assuming nomic-embed-text size is 768 dimensions
    init_optimized_collection(client, collection_name=collection_name, vector_size=768)

    print(f"[INFO] Reading document from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("[INFO] Splitting text into structured chunks...")
    chunker = EnterpriseDocumentChunker(chunk_size=400, chunk_overlap=50)
    chunks = chunker.split_text(raw_text, metadata={"source": file_path, "domain": "security_policy"})
    
    texts_to_embed = [chunk["text"] for chunk in chunks]
    print(f"[INFO] Generated {len(texts_to_embed)} chunks. Requesting async embeddings from Ollama...")

    embedder = OllamaEmbeddingClient(model_name="nomic-embed-text")
    try:
        embeddings = await embedder.generate_batch_embeddings(texts_to_embed)
    finally:
        await embedder.close()

    print("[INFO] Preparing payload points and upserting into Qdrant storage...")
    points = []
    for idx, chunk in enumerate(chunks):
        # Best Practice: Use deterministic UUID generation based on content path 
        # to prevent duplicate point injections on re-runs
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_path}-{chunk['chunk_id']}"))
        
        points.append(
            models.PointStruct(
                id=point_id,
                vector=embeddings[idx],
                payload={
                    "text": chunk["text"],
                    "source": chunk["metadata"].get("source"),
                    "domain": chunk["metadata"].get("domain"),
                    "chunk_id": chunk["chunk_id"]
                }
            )
        )

    # Perform batch upsert into Qdrant
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    print(f"[SUCCESS] Successfully ingested {len(points)} points into Qdrant collection '{collection_name}'!")

if __name__ == "__main__":
    asyncio.run(run_ingestion_pipeline())
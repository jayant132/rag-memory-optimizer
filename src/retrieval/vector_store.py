import os
from qdrant_client import QdrantClient
from qdrant_client.http import models

def get_vector_store_client() -> QdrantClient:
    """
    Initializes and returns a persistent Qdrant client connection 
    pointing to the local or cloud infrastructure instance.
    """
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
    
    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

def init_optimized_collection(client: QdrantClient, collection_name: str = "rag_memory_index", vector_size: int = 768) -> None:
    """
    Provisions a production-grade Qdrant collection configured with:
    1. HNSW index for fast approximate nearest neighbor (ANN) search.
    2. Scalar Quantization (SQ8) to cut RAM usage by ~4x.
    3. Tiered storage (float32 on disk, int8 quantized vectors in RAM).
    """
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,  # E.g., 768 for nomic-embed-text
                distance=models.Distance.COSINE,
                on_disk=True       # Keeps source float32 vectors on disk storage to save RAM
            ),
            # Memory optimization: SQ8 compression tier
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,   # Excludes extreme 1% outliers to maintain stable scaling bounds
                    always_ram=True  # Keeps lightweight int8 index in RAM for sub-10ms search
                )
            ),
            # Graph index parameters optimized for recall vs memory trade-off
            hnsw_config=models.HnswConfigDiff(
                m=16,             # Max edges per node in the index graph
                ef_construct=100  # Index build exploration depth
            )
        )
        print(f"[INFO] Collection '{collection_name}' successfully created with SQ8 quantization.")
    else:
        print(f"[INFO] Collection '{collection_name}' already exists. Skipping initialization.")
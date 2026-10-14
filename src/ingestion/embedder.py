import os
import asyncio
import httpx
from typing import List

class OllamaEmbeddingClient:
    """
    Asynchronous client for interacting with local Ollama embedding models.
    Designed for non-blocking batch processing in enterprise ingestion pipelines.
    """
    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        # Best Practice: Reusing an async HTTP client session reduces TCP handshake overhead
        self.client = httpx.AsyncClient(timeout=60.0)

    async def generate_embedding(self, text: str) -> List[float]:
        """Generates a dense vector embedding for a single text chunk."""
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        return data.get("embedding", [])

    async def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Processes batch embeddings concurrently using asyncio.gather 
        to maximize ingestion throughput.
        """
        tasks = [self.generate_embedding(text) for text in texts]
        embeddings = await asyncio.gather(*tasks)
        return embeddings

    async def close(self) -> None:
        """Gracefully closes the underlying HTTP transport layer."""
        await self.client.aclose()
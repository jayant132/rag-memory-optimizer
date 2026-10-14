from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class RetrievalOptimizer:
    """
    Handles second-stage cross-encoder reranking and strict token-budget 
    allocation to optimize LLM generation costs and accuracy.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        print(f"[INFO] Loading cross-encoder reranking model: {model_name}...")
        # Cross-encoders jointly process query and document text for high-precision semantic scoring
        self.encoder = CrossEncoder(model_name)

    def rerank_results(self, query: str, retrieved_docs: List[Dict[str, Any]], top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Re-scores a candidate pool of retrieved chunks using a cross-encoder 
        and filters down to the top_k highest-signal chunks.
        """
        if not retrieved_docs:
            return []

        # Prepare pairs of [query, document_text] for cross-encoder scoring
        pairs = [[query, doc["text"]] for doc in retrieved_docs]
        scores = self.encoder.predict(pairs)

        # Attach scores back to documents
        for i, doc in enumerate(retrieved_docs):
            doc["rerank_score"] = float(scores[i])

        # Sort descending by cross-encoder score
        reranked = sorted(retrieved_docs, key=lambda x: x["rerank_score"], reverse=True)
        
        # Return only the top_k to enforce strict token budget limits
        return reranked[:top_k]

    def pack_context_within_budget(self, reranked_docs: List[Dict[str, Any]], max_tokens: int = 1500) -> str:
        """
        Packs filtered chunks into a unified context string while respecting 
        a maximum token budget constraint.
        """
        context_parts = []
        current_estimated_tokens = 0

        for doc in reranked_docs:
            text = doc["text"]
            # Rough estimation: 1 token ≈ 4 characters in English text
            est_tokens = len(text) // 4
            
            if current_estimated_tokens + est_tokens > max_tokens:
                break # Stop packing if budget is exceeded
                
            context_parts.append(text)
            current_estimated_tokens += est_tokens

        return "\n\n---\n\n".join(context_parts)

if __name__ == "__main__":
    # Quick standalone test
    optimizer = RetrievalOptimizer()
    mock_docs = [
        {"text": "Section 1.1 governs data storage compliance.", "id": "1"},
        {"text": "Unrelated paragraph about cafeteria menus.", "id": "2"}
    ]
    results = optimizer.rerank_results("What are the data storage rules?", mock_docs, top_k=1)
    print("Reranked result:", results)
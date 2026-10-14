from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

class EnterpriseDocumentChunker:
    """
    Splits raw enterprise documentation into context-aware chunks 
    using recursive structural boundaries.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_text(self, text: str, metadata: dict = None) -> List[dict]:
        """
        Splits raw text string into structured chunk dictionaries 
        containing the text payload and inherited file metadata.
        """
        docs = self.splitter.create_documents(
            texts=[text], 
            metadatas=[metadata or {}]
        )
        
        # Format into clean dictionaries for downstream vector upserting
        structured_chunks = []
        for i, doc in enumerate(docs):
            structured_chunks.append({
                "chunk_id": i,
                "text": doc.page_content,
                "metadata": doc.metadata
            })
            
        return structured_chunks

if __name__ == "__main__":
    # Quick module test execution
    sample_text = "# Title\n\nSection 1 details go here.\n\nSection 2 follows with more data."
    chunker = EnterpriseDocumentChunker(chunk_size=30, chunk_overlap=5)
    chunks = chunker.split_text(sample_text, {"source": "test.md"})
    print(f"Generated {len(chunks)} chunks successfully.")
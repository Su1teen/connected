from typing import List, Dict, Any, Tuple
from langchain.schema import Document
from sentence_transformers import CrossEncoder

class Reranker:
    """Rerank retrieved documents using a cross-encoder model."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Document]:
        """
        Rerank documents based on relevance to the query.
        
        Args:
            query: The user query
            documents: List of retrieved documents
            top_k: Number of documents to return after reranking
        
        Returns:
            List of reranked documents
        """
        if not documents:
            return []
        
        # Create document-query pairs for scoring
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get scores from the cross-encoder
        scores = self.model.predict(pairs)
        
        # Create (document, score) pairs
        doc_score_pairs = list(zip(documents, scores))
        
        # Sort by score in descending order
        reranked_docs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        
        # Take top_k if specified
        if top_k is not None and top_k < len(reranked_docs):
            reranked_docs = reranked_docs[:top_k]
        
        # Return just the documents
        return [doc for doc, _ in reranked_docs]
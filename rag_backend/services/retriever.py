from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.vectorstores import VectorStore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

class HybridRetriever:
    """
    Hybrid retriever that combines dense and sparse retrieval methods.
    This provides better recall by leveraging both semantic similarity and keyword matching.
    """
    
    def __init__(self, 
                 vectorstore: VectorStore, 
                 documents: List[Document],
                 dense_weight: float = 0.7,
                 sparse_weight: float = 0.3,
                 k: int = 4):
        """
        Initialize the hybrid retriever.
        
        Args:
            vectorstore: Vector store for dense retrieval
            documents: List of documents for sparse retrieval
            dense_weight: Weight for dense retriever (0-1)
            sparse_weight: Weight for sparse retriever (0-1)
            k: Default number of documents to retrieve
        """
        self.vectorstore = vectorstore
        self.documents = documents
        self.k = k
        
        # Initialize dense retriever
        self.dense_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        # Initialize sparse retriever (BM25)
        self.sparse_retriever = BM25Retriever.from_documents(documents)
        self.sparse_retriever.k = k
        
        # Initialize ensemble retriever
        self.update_weights(dense_weight, sparse_weight)
    
    def update_weights(self, dense_weight: float, sparse_weight: float):
        """
        Update the weights of the retrievers.
        
        Args:
            dense_weight: Weight for dense retriever (0-1)
            sparse_weight: Weight for sparse retriever (0-1)
        """
        if not (0 <= dense_weight <= 1) or not (0 <= sparse_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")
        
        if abs(dense_weight + sparse_weight - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1")
        
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        self.retriever = EnsembleRetriever(
            retrievers=[self.dense_retriever, self.sparse_retriever],
            weights=[dense_weight, sparse_weight]
        )
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Retrieve documents relevant to the query.
        
        Args:
            query: The query string
            k: Number of documents to retrieve (overrides default)
            
        Returns:
            List of retrieved documents
        """
        if k is not None:
            # Temporarily adjust k for both retrievers
            original_k = self.k
            self.dense_retriever.search_kwargs["k"] = k
            self.sparse_retriever.k = k
            
            # Get results
            results = self.retriever.get_relevant_documents(query)
            
            # Restore original k
            self.dense_retriever.search_kwargs["k"] = original_k
            self.sparse_retriever.k = original_k
            
            return results
        
        return self.retriever.get_relevant_documents(query)
    
    def get_retriever_info(self) -> Dict[str, Any]:
        """
        Get information about the retriever.
        
        Returns:
            Dictionary with retriever information
        """
        return {
            "type": "hybrid",
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
            "default_k": self.k,
            "dense_retriever": self.dense_retriever.__class__.__name__,
            "sparse_retriever": self.sparse_retriever.__class__.__name__,
            "document_count": len(self.documents)
        }
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_community.callbacks.manager import get_openai_callback
import time
import os

from rag_backend.utils.prompt_template import PromptTemplates
from rag_backend.utils.reranking import Reranker

class RAGChainService:
    """
    Service for running RAG (Retrieval-Augmented Generation) chains.
    This service handles the retrieval of relevant documents and generation of responses.
    """
    
    def __init__(self, retriever, reranker: Optional[Reranker] = None):
        """
        Initialize the RAG chain service.
        
        Args:
            retriever: Document retriever
            reranker: Optional reranker for improving retrieval quality
        """
        self.retriever = retriever
        self.reranker = reranker
        
        # Initialize LLM
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.2,
            openai_api_key=api_key
        )
        
        # Initialize prompt templates
        self.rag_prompt = PromptTemplates.get_rag_prompt()
        
        # Initialize LLM chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.rag_prompt
        )
    
    def query(self, 
              query: str, 
              k: int = 4, 
              rerank: bool = True, 
              rerank_top_k: Optional[int] = None,
              memory_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a query using the RAG system, with optional conversation memory.
        
        Args:
            query: The query string
            k: Number of documents to retrieve
            rerank: Whether to rerank the retrieved documents
            rerank_top_k: Number of top documents to keep after reranking
            memory_context: Optional context from previous conversation turns
            
        Returns:
            Dictionary with the generated answer and retrieved documents
        """
        start_time = time.time()
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query, k=k)
        
        # Rerank documents if requested
        if rerank and self.reranker and retrieved_docs:
            retrieved_docs = self.reranker.rerank(
                query=query,
                documents=retrieved_docs,
                top_k=rerank_top_k or k
            )
        
        # Format context from retrieved documents
        context = self._format_context(retrieved_docs)
        
        # If memory_context is provided, use chat prompt
        if memory_context:
            prompt = PromptTemplates.get_chat_prompt()
            chain = LLMChain(
                llm=self.llm,
                prompt=prompt
            )
            chain_input = {
                "input": query,
                "context": context,
                "chat_history": memory_context
            }
        else:
            chain = self.chain
            chain_input = {"input": query, "context": context}
        
        token_usage = {}
        with get_openai_callback() as cb:
            answer = chain.run(**chain_input)
            token_usage = {
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_tokens": cb.total_tokens,
                "cost": cb.total_cost
            }
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare response
        return {
            "answer": answer,
            "documents": [self._format_document(doc) for doc in retrieved_docs],
            "metadata": {
                "processing_time": processing_time,
                "token_usage": token_usage,
                "retrieval_method": "hybrid" if hasattr(self.retriever, "get_retriever_info") else "standard",
                "reranked": rerank and self.reranker is not None,
                "document_count": len(retrieved_docs)
            }
        }
    
    def _format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a context string.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", "Unknown")
            if isinstance(source, str) and os.path.exists(source):
                source = os.path.basename(source)
            
            context_parts.append(f"Document {i+1} [Source: {source}]:\n{doc.page_content}\n")
        
        return "\n".join(context_parts)
    
    def _format_document(self, doc: Document) -> Dict[str, Any]:
        """
        Format a document for the API response.
        
        Args:
            doc: Document to format
            
        Returns:
            Dictionary with document information
        """
        source = doc.metadata.get("source", "Unknown")
        if isinstance(source, str) and os.path.exists(source):
            source = os.path.basename(source)
            
        return {
            "content": doc.page_content,
            "metadata": {
                "source": source,
                "chunk_id": doc.metadata.get("chunk_id", "Unknown"),
                "page": doc.metadata.get("page", None),
                "score": doc.metadata.get("score", None)
            }
        }
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    """Request model for querying the RAG system."""
    query: str = Field(..., description="The query string")
    k: int = Field(4, description="Number of documents to retrieve")
    rerank: bool = Field(True, description="Whether to rerank the retrieved documents")
    rerank_top_k: Optional[int] = Field(None, description="Number of top documents to keep after reranking")

class DocumentResponse(BaseModel):
    """Response model for a retrieved document."""
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")

class QueryResponse(BaseModel):
    """Response model for a query to the RAG system."""
    answer: str = Field(..., description="Generated answer")
    documents: List[DocumentResponse] = Field(..., description="Retrieved documents")
    metadata: Dict[str, Any] = Field(..., description="Response metadata")

class IngestRequest(BaseModel):
    """Request model for ingesting documents."""
    directory: Optional[str] = Field(None, description="Directory containing documents to ingest")

class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    status: str = Field(..., description="Status of the ingestion operation")
    document_count: int = Field(..., description="Number of documents ingested")
    message: str = Field(..., description="Status message")
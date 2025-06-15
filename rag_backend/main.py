import os
import shutil
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from rag_backend.models.schema import QueryRequest, QueryResponse, IngestRequest, IngestResponse
from rag_backend.utils.document_loader import DocumentProcessor
from rag_backend.utils.chunking import DocumentChunker
from rag_backend.utils.embeddings import get_hf_embeddings
from rag_backend.utils.reranking import Reranker
from rag_backend.services.vectorstore import VectorStoreService
from rag_backend.services.retriever import HybridRetriever
from rag_backend.services.rag_chain import RAGChainService
from rag_backend.services.conversation_manager import ConversationManager
from rag_backend.models.conversation_schema import (
    ConversationCreateRequest, ConversationCreateResponse,
    ConversationHistoryResponse, ConversationListResponse, ConversationDeleteResponse
)

# Create FastAPI app
app = FastAPI(title="RAG API", description="API for Retrieval-Augmented Generation")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables
DATA_DIR = "data_all"
TMP_DIR = "tmp"
CHROMA_DIR = "chroma_db"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# Initialize components
embeddings = get_hf_embeddings()
vectorstore_service = VectorStoreService(embeddings, CHROMA_DIR)
reranker = Reranker()
conversation_manager = ConversationManager()

# Global variables to store documents and retriever
all_documents = []
retriever = None
rag_chain = None

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup."""
    global all_documents, retriever, rag_chain
    
    # Load all documents from the data directory
    all_documents = DocumentProcessor.load_documents_from_directory(DATA_DIR)
    
    if all_documents:
        # Chunk the documents
        chunked_documents = DocumentChunker.split_documents(all_documents)
        
        # Add documents to vector store
        vectorstore_service.add_documents(chunked_documents)
        
        # Initialize retriever
        retriever = HybridRetriever(vectorstore_service.vectorstore, chunked_documents)
        
        # Initialize RAG chain
        rag_chain = RAGChainService(retriever, reranker)
        print(f"System initialized with {len(all_documents)} documents")
    else:
        print("No documents found in the data directory")

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"status": "ok", "message": "RAG API is running"}

@app.post("/conversation/create", response_model=ConversationCreateResponse)
async def create_conversation(request: ConversationCreateRequest):
    conversation_id = conversation_manager.create_conversation(request.metadata)
    return ConversationCreateResponse(conversation_id=conversation_id)

@app.get("/conversation/{conversation_id}/history", response_model=ConversationHistoryResponse)
async def get_conversation_history(conversation_id: str):
    history = conversation_manager.get_history(conversation_id)
    return ConversationHistoryResponse(history=history)

@app.get("/conversation/list", response_model=ConversationListResponse)
async def list_conversations():
    conversations = conversation_manager.list_conversations()
    return ConversationListResponse(conversations=conversations)

@app.delete("/conversation/{conversation_id}", response_model=ConversationDeleteResponse)
async def delete_conversation(conversation_id: str):
    success = conversation_manager.delete_conversation(conversation_id)
    return ConversationDeleteResponse(success=success)

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a query using the RAG system, with optional conversation memory.
    
    Args:
        request: The query request containing the query and parameters
    
    Returns:
        The generated answer and retrieved documents
    """
    global rag_chain
    
    if not rag_chain:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        conversation_id = getattr(request, 'conversation_id', None)
        memory_context = None
        if conversation_id:
            memory_context = conversation_manager.get_conversation_context(conversation_id)
        result = rag_chain.query(
            query=request.query,
            k=request.k,
            rerank=request.rerank,
            rerank_top_k=request.rerank_top_k,
            memory_context=memory_context
        )
        
        # Save to memory if conversation_id is provided
        if conversation_id:
            conversation_manager.add_message(conversation_id, request.query, result["answer"])
        
        return QueryResponse(
            answer=result["answer"],
            documents=result["documents"],
            metadata=result["metadata"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/upload", response_model=IngestResponse)
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a file to be processed and added to the knowledge base.
    
    Args:
        background_tasks: FastAPI background tasks
        file: The uploaded file
    
    Returns:
        Status of the upload operation
    """
    try:
        # Save the file to the temporary directory
        file_path = os.path.join(TMP_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the file in the background
        background_tasks.add_task(process_uploaded_file, file_path)
        
        return IngestResponse(
            status="success",
            document_count=1,
            message=f"File {file.filename} uploaded and being processed"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """
    Ingest documents from a specified directory.
    
    Args:
        request: The ingest request containing the directory path
    
    Returns:
        Status of the ingestion operation
    """
    global all_documents, retriever, rag_chain
    
    try:
        directory = request.directory or DATA_DIR
        
        # Load documents from the directory
        new_documents = DocumentProcessor.load_documents_from_directory(directory)
        
        if not new_documents:
            return IngestResponse(
                status="warning",
                document_count=0,
                message=f"No documents found in {directory}"
            )
        
        # Chunk the documents
        chunked_documents = DocumentChunker.split_documents(new_documents)
        
        # Add documents to vector store
        vectorstore_service.add_documents(chunked_documents)
        
        # Update global documents
        all_documents.extend(new_documents)
        
        # Reinitialize retriever and RAG chain
        retriever = HybridRetriever(vectorstore_service.vectorstore, all_documents)
        rag_chain = RAGChainService(retriever, reranker)
        
        return IngestResponse(
            status="success",
            document_count=len(new_documents),
            message=f"Successfully ingested {len(new_documents)} documents"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting documents: {str(e)}")

@app.get("/status")
async def get_status():
    """Get the status of the RAG system."""
    return {
        "documents_count": len(all_documents) if all_documents else 0,
        "system_initialized": rag_chain is not None,
        "data_directory": DATA_DIR,
        "temp_directory": TMP_DIR
    }

@app.post("/update_retriever")
async def update_retriever(dense_weight: float = 0.7, sparse_weight: float = 0.3):
    """
    Update the weights of the hybrid retriever.
    
    Args:
        dense_weight: Weight for dense retriever (0-1)
        sparse_weight: Weight for sparse retriever (0-1)
    
    Returns:
        Status of the update operation
    """
    global retriever
    
    if not retriever:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
    
    try:
        retriever.update_weights(dense_weight, sparse_weight)
        return {"status": "success", "message": f"Retriever weights updated: dense={dense_weight}, sparse={sparse_weight}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating retriever: {str(e)}")

async def process_uploaded_file(file_path: str):
    """
    Process an uploaded file and add it to the knowledge base.
    
    Args:
        file_path: Path to the uploaded file
    """
    global all_documents, retriever, rag_chain
    
    try:
        # Load the document
        new_documents = DocumentProcessor.load_document(file_path)
        
        if not new_documents:
            print(f"No content extracted from {file_path}")
            return
        
        # Move the file to the data directory
        filename = os.path.basename(file_path)
        destination = os.path.join(DATA_DIR, filename)
        shutil.move(file_path, destination)
        
        # Update metadata to reflect new location
        for doc in new_documents:
            doc.metadata["source"] = destination
        
        # Chunk the documents
        chunked_documents = DocumentChunker.split_documents(new_documents)
        
        # Add documents to vector store
        vectorstore_service.add_documents(chunked_documents)
        
        # Update global documents
        all_documents.extend(new_documents)
        
        # Reinitialize retriever and RAG chain if needed
        if retriever and rag_chain:
            retriever = HybridRetriever(vectorstore_service.vectorstore, all_documents)
            rag_chain = RAGChainService(retriever, reranker)
            
        print(f"Successfully processed {filename}")
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
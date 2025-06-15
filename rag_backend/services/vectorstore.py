from typing import List, Optional
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings

class VectorStoreService:
    """Service for managing the vector store."""
    
    def __init__(self, embedding_function: Embeddings, persist_directory: str = "./chroma_db"):
        """
        Initialize the vector store service.
        
        Args:
            embedding_function: The embedding function to use
            persist_directory: Directory to persist the vector store
        """
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        self.vectorstore = self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize the vector store."""
        try:
            # Try to load an existing vector store
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function
            )
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            print("Creating a new vector store...")
            # Create a new vector store if loading fails
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function
            )
    
    def add_documents(self, documents: List[Document]):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        if not documents:
            return
        
        try:
            self.vectorstore.add_documents(documents)
            self.vectorstore.persist()
            print(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            print(f"Error adding documents to vector store: {str(e)}")
    
    def search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search the vector store for documents similar to the query.
        
        Args:
            query: The query string
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        return self.vectorstore.similarity_search(query, k=k)
    
    def delete_collection(self):
        """Delete the entire collection from the vector store."""
        try:
            self.vectorstore.delete_collection()
            self.vectorstore = self._initialize_vectorstore()
            print("Vector store collection deleted and reinitialized")
        except Exception as e:
            print(f"Error deleting vector store collection: {str(e)}")
    
    def get_collection_stats(self):
        """
        Get statistics about the vector store collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            return {
                "document_count": count,
                "embedding_dimension": self.vectorstore._embedding_function.dimension if hasattr(self.vectorstore._embedding_function, "dimension") else "unknown",
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            print(f"Error getting collection stats: {str(e)}")
            return {
                "document_count": 0,
                "error": str(e)
            }
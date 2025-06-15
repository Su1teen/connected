from typing import List, Dict, Any
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

class DocumentChunker:
    """Utility for chunking documents into smaller pieces."""
    
    @staticmethod
    def split_documents(documents: List[Document], 
                        chunk_size: int = 512, 
                        chunk_overlap: int = 200) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of chunked documents
        """
        if not documents:
            return []
        
        # First split by character to get rough chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        # Then split by tokens to ensure we don't exceed model context
        token_splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=20,  # Token overlap
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Process documents
        chunked_docs = []
        for doc in documents:
            # First split by character
            char_split_docs = text_splitter.split_documents([doc])
            
            # Then split by tokens if needed
            for char_doc in char_split_docs:
                # Only apply token splitting if the chunk is large
                if len(char_doc.page_content) > chunk_size:
                    token_docs = token_splitter.split_documents([char_doc])
                    chunked_docs.extend(token_docs)
                else:
                    chunked_docs.append(char_doc)
        
        # Ensure all chunks have proper metadata
        for i, doc in enumerate(chunked_docs):
            if "chunk_id" not in doc.metadata:
                doc.metadata["chunk_id"] = i
        
        return chunked_docs
    
    @staticmethod
    def get_chunk_stats(documents: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about document chunks.
        
        Args:
            documents: List of chunked documents
            
        Returns:
            Dictionary with chunk statistics
        """
        if not documents:
            return {"count": 0, "avg_size": 0, "min_size": 0, "max_size": 0}
        
        sizes = [len(doc.page_content) for doc in documents]
        return {
            "count": len(documents),
            "avg_size": sum(sizes) / len(sizes),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "total_tokens_estimate": sum(sizes) // 4  # Rough estimate of tokens
        }
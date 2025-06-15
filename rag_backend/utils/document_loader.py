import os
from typing import List, Optional
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredImageLoader,
    UnstructuredMarkdownLoader
)

class DocumentProcessor:
    """Utility for loading and processing documents from various file formats."""
    
    @staticmethod
    def load_document(file_path: str) -> List[Document]:
        """
        Load a document from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of documents extracted from the file
        """
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return []
        
        try:
            # Get file extension
            _, ext = os.path.splitext(file_path.lower())
            
            # Select appropriate loader based on file extension
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext == '.txt':
                loader = TextLoader(file_path)
            elif ext in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            elif ext == '.csv':
                loader = CSVLoader(file_path)
            elif ext in ['.xlsx', '.xls']:
                loader = UnstructuredExcelLoader(file_path)
            elif ext in ['.jpg', '.jpeg', '.png']:
                loader = UnstructuredImageLoader(file_path)
            elif ext == '.md':
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                print(f"Unsupported file format: {ext}")
                return []
            
            # Load the document
            documents = loader.load()
            
            # Add source metadata if not present
            for doc in documents:
                if "source" not in doc.metadata:
                    doc.metadata["source"] = file_path
            
            return documents
        except Exception as e:
            print(f"Error loading document {file_path}: {str(e)}")
            return []
    
    @staticmethod
    def load_documents_from_directory(directory: str, recursive: bool = True) -> List[Document]:
        """
        Load all documents from a directory.
        
        Args:
            directory: Directory path
            recursive: Whether to search subdirectories
            
        Returns:
            List of documents
        """
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            return []
        
        documents = []
        
        # Define supported file extensions
        supported_extensions = [
            '.pdf', '.txt', '.docx', '.doc', 
            '.csv', '.xlsx', '.xls', 
            '.jpg', '.jpeg', '.png', '.md'
        ]
        
        # Walk through the directory
        for root, _, files in os.walk(directory):
            if not recursive and root != directory:
                continue
                
            for file in files:
                _, ext = os.path.splitext(file.lower())
                if ext in supported_extensions:
                    file_path = os.path.join(root, file)
                    docs = DocumentProcessor.load_document(file_path)
                    if docs:
                        documents.extend(docs)
                        print(f"Loaded {len(docs)} documents from {file_path}")
        
        return documents
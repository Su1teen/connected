from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
import os

def get_hf_embeddings(model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    """
    Get HuggingFace embeddings model.
    
    Args:
        model_name: Name of the HuggingFace model
        
    Returns:
        HuggingFaceEmbeddings instance
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def get_openai_embeddings():
    """
    Get OpenAI embeddings model.
    
    Returns:
        OpenAIEmbeddings instance
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key
    )
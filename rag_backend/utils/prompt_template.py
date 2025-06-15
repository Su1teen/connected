from langchain.prompts import PromptTemplate

class PromptTemplates:
    """Collection of prompt templates for different use cases."""
    
    @staticmethod
    def get_rag_prompt():
        """
        Get the prompt template for basic RAG.
        This prompt is designed for direct question answering from retrieved documents.
        """
        template = """You are an AI assistant that helps users find information from a knowledge base.
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that 
        you don't know. Use three sentences maximum and keep the answer concise.
        
        Question: {input}
        
        Context:
        {context}
        
        Answer:"""
        
        return PromptTemplate.from_template(template)
    
    @staticmethod
    def get_chat_prompt():
        """
        Get the prompt template for conversational RAG.
        This prompt includes chat history for maintaining context across turns.
        """
        template = """You are an AI assistant that helps users with their questions.
        You have access to a knowledge base that contains information about various topics.
        If the information is available in the context, use it to answer the question.
        If the information is not available, you can use your general knowledge to answer the question.
        Always be helpful, concise, and accurate.
        
        Previous conversation:
        {chat_history}
        
        Context from knowledge base:
        {context}
        
        User question: {input}
        
        Answer:"""
        
        return PromptTemplate.from_template(template)
    
    @staticmethod
    def get_summarization_prompt():
        """
        Get the prompt template for document summarization.
        This prompt is designed to create concise summaries of retrieved documents.
        """
        template = """You are an AI assistant that summarizes documents.
        Summarize the following document in a concise manner, capturing the key points.
        Keep the summary to 3-5 sentences.
        
        Document:
        {context}
        
        Summary:"""
        
        return PromptTemplate.from_template(template)
    
    @staticmethod
    def get_extraction_prompt():
        """
        Get the prompt template for information extraction.
        This prompt is designed to extract specific information from documents.
        """
        template = """You are an AI assistant that extracts specific information from documents.
        Extract the following information from the document:
        
        {input}
        
        Document:
        {context}
        
        Extracted information:"""
        
        return PromptTemplate.from_template(template)
    
    @staticmethod
    def get_custom_prompt(template_string):
        """
        Create a custom prompt template from a string.
        
        Args:
            template_string: The template string
            
        Returns:
            A PromptTemplate object
        """
        return PromptTemplate.from_template(template_string)
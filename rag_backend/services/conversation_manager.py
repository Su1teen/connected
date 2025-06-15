from typing import List, Dict, Any, Optional
from langchain.memory import ConversationBufferMemory
import uuid
import time
import json
import os

class ConversationManager:
    """
    Service for managing conversation history and memory.
    This allows for maintaining context across multiple queries.
    """
    
    def __init__(self, memory_dir: str = "conversation_memory"):
        """
        Initialize the conversation manager.
        
        Args:
            memory_dir: Directory to store conversation memory
        """
        self.memory_dir = memory_dir
        self.active_conversations = {}
        self.max_conversations = 100  # Maximum number of active conversations
        
        # Create memory directory if it doesn't exist
        os.makedirs(memory_dir, exist_ok=True)
    
    def create_conversation(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new conversation.
        
        Args:
            metadata: Optional metadata for the conversation
            
        Returns:
            Conversation ID
        """
        # Generate a unique ID
        conversation_id = str(uuid.uuid4())
        
        # Initialize memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create conversation record
        conversation = {
            "id": conversation_id,
            "memory": memory,
            "created_at": time.time(),
            "last_updated": time.time(),
            "metadata": metadata or {},
            "message_count": 0
        }
        
        # Store in active conversations
        self.active_conversations[conversation_id] = conversation
        
        # Clean up old conversations if needed
        self._cleanup_old_conversations()
        
        return conversation_id
    
    def add_message(self, 
                    conversation_id: str, 
                    user_message: str, 
                    ai_message: str) -> bool:
        """
        Add a message pair to a conversation.
        
        Args:
            conversation_id: ID of the conversation
            user_message: User's message
            ai_message: AI's response
            
        Returns:
            True if successful, False otherwise
        """
        # Check if conversation exists
        if conversation_id not in self.active_conversations:
            # Try to load from disk
            if not self._load_conversation(conversation_id):
                return False
        
        # Get conversation
        conversation = self.active_conversations[conversation_id]
        
        # Add messages to memory
        conversation["memory"].save_context(
            {"input": user_message},
            {"output": ai_message}
        )
        
        # Update conversation metadata
        conversation["last_updated"] = time.time()
        conversation["message_count"] += 1
        
        # Save conversation to disk
        self._save_conversation(conversation_id)
        
        return True
    
    def get_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """
        Get the history of a conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            List of message pairs
        """
        # Check if conversation exists
        if conversation_id not in self.active_conversations:
            # Try to load from disk
            if not self._load_conversation(conversation_id):
                return []
        
        # Get conversation
        conversation = self.active_conversations[conversation_id]
        
        # Extract messages from memory
        messages = conversation["memory"].chat_memory.messages
        
        # Format messages
        history = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                history.append({
                    "user": messages[i].content,
                    "ai": messages[i + 1].content
                })
        
        return history
    
    def get_conversation_context(self, conversation_id: str) -> str:
        """
        Get the conversation context as a string.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Conversation context string
        """
        # Check if conversation exists
        if conversation_id not in self.active_conversations:
            # Try to load from disk
            if not self._load_conversation(conversation_id):
                return ""
        
        # Get conversation
        conversation = self.active_conversations[conversation_id]
        
        # Get variables from memory
        return conversation["memory"].buffer
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            True if successful, False otherwise
        """
        # Check if conversation exists
        if conversation_id not in self.active_conversations:
            # Check if it exists on disk
            file_path = os.path.join(self.memory_dir, f"{conversation_id}.json")
            if not os.path.exists(file_path):
                return False
        
        # Remove from active conversations
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]
        
        # Remove from disk
        file_path = os.path.join(self.memory_dir, f"{conversation_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return True
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """
        List all conversations.
        
        Returns:
            List of conversation metadata
        """
        # Load all conversations from disk
        self._load_all_conversations()
        
        # Format conversation metadata
        conversations = []
        for conv_id, conv in self.active_conversations.items():
            conversations.append({
                "id": conv_id,
                "created_at": conv["created_at"],
                "last_updated": conv["last_updated"],
                "message_count": conv["message_count"],
                "metadata": conv["metadata"]
            })
        
        # Sort by last updated (newest first)
        conversations.sort(key=lambda x: x["last_updated"], reverse=True)
        
        return conversations
    
    def _save_conversation(self, conversation_id: str) -> bool:
        """
        Save a conversation to disk.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            True if successful, False otherwise
        """
        if conversation_id not in self.active_conversations:
            return False
        
        conversation = self.active_conversations[conversation_id]
        
        # Prepare data for serialization
        data = {
            "id": conversation["id"],
            "created_at": conversation["created_at"],
            "last_updated": conversation["last_updated"],
            "metadata": conversation["metadata"],
            "message_count": conversation["message_count"],
            "messages": conversation["memory"].chat_memory.messages
        }
        
        # Convert messages to dict
        messages_data = []
        for msg in data["messages"]:
            messages_data.append({
                "type": msg.type,
                "content": msg.content
            })
        
        data["messages"] = messages_data
        
        # Save to file
        file_path = os.path.join(self.memory_dir, f"{conversation_id}.json")
        try:
            with open(file_path, "w") as f:
                json.dump(data, f)
            return True
        except Exception as e:
            print(f"Error saving conversation {conversation_id}: {str(e)}")
            return False
    
    def _load_conversation(self, conversation_id: str) -> bool:
        """
        Load a conversation from disk.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            True if successful, False otherwise
        """
        file_path = os.path.join(self.memory_dir, f"{conversation_id}.json")
        if not os.path.exists(file_path):
            return False
        
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Create memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Reconstruct messages
            for i in range(0, len(data["messages"]), 2):
                if i + 1 < len(data["messages"]):
                    user_msg = data["messages"][i]
                    ai_msg = data["messages"][i + 1]
                    memory.save_context(
                        {"input": user_msg["content"]},
                        {"output": ai_msg["content"]}
                    )
            
            # Create conversation record
            conversation = {
                "id": data["id"],
                "memory": memory,
                "created_at": data["created_at"],
                "last_updated": data["last_updated"],
                "metadata": data["metadata"],
                "message_count": data["message_count"]
            }
            
            # Store in active conversations
            self.active_conversations[conversation_id] = conversation
            
            return True
        except Exception as e:
            print(f"Error loading conversation {conversation_id}: {str(e)}")
            return False
    
    def _load_all_conversations(self):
        """Load all conversations from disk."""
        try:
            for filename in os.listdir(self.memory_dir):
                if filename.endswith(".json"):
                    conversation_id = filename[:-5]  # Remove .json extension
                    if conversation_id not in self.active_conversations:
                        self._load_conversation(conversation_id)
        except Exception as e:
            print(f"Error loading all conversations: {str(e)}")
    
    def _cleanup_old_conversations(self):
        """Remove old conversations if exceeding max_conversations."""
        if len(self.active_conversations) > self.max_conversations:
            # Sort by last_updated
            sorted_convs = sorted(self.active_conversations.items(), key=lambda x: x[1]["last_updated"])
            # Remove oldest
            for conv_id, _ in sorted_convs[:-self.max_conversations]:
                self.delete_conversation(conv_id)

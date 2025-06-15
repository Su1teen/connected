from pydantic import BaseModel
from typing import Optional

class ConversationCreateRequest(BaseModel):
    metadata: Optional[dict] = None

class ConversationCreateResponse(BaseModel):
    conversation_id: str

class ConversationHistoryResponse(BaseModel):
    history: list

class ConversationListResponse(BaseModel):
    conversations: list

class ConversationDeleteResponse(BaseModel):
    success: bool

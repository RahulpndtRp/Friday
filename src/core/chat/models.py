from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
import uuid
import json
from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Get current UTC datetime - Python 3.12 compatible."""
    return datetime.now(timezone.utc)


class MessageType(Enum):
    """Types of chat messages."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    ERROR = "error"
    INFO = "info"


class MessageStatus(Enum):
    """Status of message processing."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    STREAMING = "streaming"


class ResponseType(Enum):
    """Types of assistant responses."""

    TEXT = "text"
    DOCUMENT_SEARCH = "document_search"
    MEMORY_RECALL = "memory_recall"
    FUNCTION_CALL = "function_call"
    ERROR = "error"


# Pydantic models for API
class ChatMessageRequest(BaseModel):
    """Request model for chat messages."""

    message: str = Field(
        ..., min_length=1, max_length=10000, description="User message"
    )
    conversation_id: Optional[str] = Field(
        None, description="Conversation ID for context"
    )
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional context"
    )
    stream: bool = Field(default=True, description="Enable streaming response")
    include_sources: bool = Field(default=True, description="Include source documents")
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum response tokens"
    )


class ChatMessageResponse(BaseModel):
    """Response model for chat messages."""

    message_id: str = Field(..., description="Unique message ID")
    conversation_id: str = Field(..., description="Conversation ID")
    response: str = Field(..., description="Assistant response")
    response_type: str = Field(..., description="Type of response")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list, description="Source documents/memories"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Response metadata"
    )
    processing_time: float = Field(..., description="Processing time in seconds")
    token_count: Optional[int] = Field(None, description="Token count used")
    status: str = Field(..., description="Message status")


class StreamChunk(BaseModel):
    """Model for streaming response chunks."""

    chunk_id: str = Field(..., description="Chunk ID")
    message_id: str = Field(..., description="Message ID")
    content: str = Field(..., description="Chunk content")
    is_final: bool = Field(default=False, description="Is this the final chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")


@dataclass
class ChatMessage:
    """Internal chat message representation."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    user_id: str = ""
    message_type: MessageType = MessageType.USER
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Response information
    response_type: Optional[ResponseType] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    processing_time: float = 0.0
    token_count: Optional[int] = None

    # Status and timestamps
    status: MessageStatus = MessageStatus.PENDING
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    # Context and relationships
    parent_message_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "message_type": self.message_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "response_type": self.response_type.value if self.response_type else None,
            "sources": self.sources,
            "processing_time": self.processing_time,
            "token_count": self.token_count,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "parent_message_id": self.parent_message_id,
            "context": self.context,
        }


@dataclass
class Conversation:
    """Chat conversation representation."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    title: str = ""
    description: str = ""

    # Conversation metadata
    message_count: int = 0
    last_activity: datetime = field(default_factory=utc_now)
    created_at: datetime = field(default_factory=utc_now)

    # Configuration
    settings: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Status
    is_active: bool = True
    is_archived: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "description": self.description,
            "message_count": self.message_count,
            "last_activity": self.last_activity.isoformat(),
            "created_at": self.created_at.isoformat(),
            "settings": self.settings,
            "tags": self.tags,
            "is_active": self.is_active,
            "is_archived": self.is_archived,
        }

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from src.core.utils.datetime_utils import utc_now


class MemoryType(Enum):
    """Different types of memory in the system."""

    CORE = "core"  # Immutable facts about user
    WORKING = "working"  # Current conversation context
    SHORT_TERM = "short_term"  # Recent interactions (hours/days)
    LONG_TERM = "long_term"  # Persistent knowledge (weeks/months)
    EPISODIC = "episodic"  # Specific events and experiences
    SEMANTIC = "semantic"  # General knowledge and facts
    PROCEDURAL = "procedural"  # How to do things, preferences
    EMOTIONAL = "emotional"  # Emotional associations and context


class MemoryImportance(Enum):
    """Memory importance levels for retention and retrieval."""

    CRITICAL = 5  # Never forget (core identity, critical preferences)
    HIGH = 4  # Important facts and preferences
    MEDIUM = 3  # Regular interactions and information
    LOW = 2  # Casual mentions and temporary info
    MINIMAL = 1  # Debug info and temporary context


class MemoryStatus(Enum):
    """Status of memory entries."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    CONSOLIDATED = "consolidated"


@dataclass
class MemoryEntry:
    """Core memory entry structure."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    memory_type: MemoryType = MemoryType.WORKING
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: MemoryImportance = MemoryImportance.MEDIUM
    tags: List[str] = field(default_factory=list)

    # Temporal information
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    accessed_at: datetime = field(default_factory=utc_now)
    expires_at: Optional[datetime] = None

    # Relationships and context
    related_memories: List[str] = field(default_factory=list)
    source_context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    # Status and lifecycle
    status: MemoryStatus = MemoryStatus.ACTIVE
    access_count: int = 0
    consolidation_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory entry to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "importance": self.importance.value,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "related_memories": self.related_memories,
            "source_context": self.source_context,
            "confidence": self.confidence,
            "status": self.status.value,
            "access_count": self.access_count,
            "consolidation_count": self.consolidation_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create memory entry from dictionary."""
        entry = cls()
        entry.id = data.get("id", entry.id)
        entry.user_id = data.get("user_id", "")
        entry.memory_type = MemoryType(data.get("memory_type", "working"))
        entry.content = data.get("content", "")
        entry.metadata = data.get("metadata", {})
        entry.importance = MemoryImportance(data.get("importance", 3))
        entry.tags = data.get("tags", [])

        # Parse datetime fields
        if data.get("created_at"):
            entry.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            entry.updated_at = datetime.fromisoformat(data["updated_at"])
        if data.get("accessed_at"):
            entry.accessed_at = datetime.fromisoformat(data["accessed_at"])
        if data.get("expires_at"):
            entry.expires_at = datetime.fromisoformat(data["expires_at"])

        entry.related_memories = data.get("related_memories", [])
        entry.source_context = data.get("source_context", {})
        entry.confidence = data.get("confidence", 1.0)
        entry.status = MemoryStatus(data.get("status", "active"))
        entry.access_count = data.get("access_count", 0)
        entry.consolidation_count = data.get("consolidation_count", 0)

        return entry

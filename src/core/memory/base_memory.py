from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import asyncio

from src.core.telemetry.logger import StructuredLogger
from src.core.config.settings import Settings
from src.core.memory.models import MemoryEntry, MemoryType


class BaseMemoryStore(ABC):
    """Abstract base class for memory storage implementations."""

    def __init__(self, user_id: str, settings: "Settings"):
        self.user_id = user_id
        self.settings = settings
        self.logger = StructuredLogger(f"memory.{self.__class__.__name__.lower()}")
        self.is_initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the memory store."""
        pass

    @abstractmethod
    async def store_memory(self, memory: MemoryEntry) -> str:
        """Store a memory entry and return its ID."""
        pass

    @abstractmethod
    async def retrieve_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory by ID."""
        pass

    @abstractmethod
    async def search_memories(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> List[MemoryEntry]:
        """Search memories by content and filters."""
        pass

    @abstractmethod
    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory entry."""
        pass

    @abstractmethod
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        pass

    @abstractmethod
    async def get_memories_by_type(
        self, memory_type: MemoryType, limit: Optional[int] = None
    ) -> List[MemoryEntry]:
        """Get memories by type."""
        pass

    @abstractmethod
    async def get_recent_memories(
        self, hours: int = 24, limit: int = 50
    ) -> List[MemoryEntry]:
        """Get recent memories within specified hours."""
        pass

    @abstractmethod
    async def consolidate_memories(self) -> Dict[str, Any]:
        """Consolidate and optimize memory storage."""
        pass

    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        pass

    async def shutdown(self) -> None:
        """Cleanup and shutdown the memory store."""
        self.logger.info(f"Shutting down memory store for user {self.user_id}")
        self.is_initialized = False

from typing import Dict, Any, List
import json
import time


from src.core.config.settings import Settings
from src.core.brain.base_brain import BaseBrain, BrainType, BrainRequest, BrainResponse
from src.core.memory.memory_manager import MemoryManager, MemoryType, MemoryImportance


class MemoryBrain(BaseBrain):
    """Brain component responsible for memory operations."""

    def __init__(self, settings: "Settings"):
        super().__init__(BrainType.MEMORY, settings)
        self.memory_managers: Dict[str, MemoryManager] = {}

    async def initialize(self) -> None:
        """Initialize the memory brain."""
        self.logger.info("Initializing Memory Brain")
        self.is_initialized = True

    async def process(self, request: BrainRequest) -> BrainResponse:
        """Process memory-related requests."""
        start_time = time.time()

        if not self._validate_request(request):
            return BrainResponse(
                request_id=request.request_id,
                brain_type=self.brain_type,
                response={},
                confidence=0.0,
                processing_time=time.time() - start_time,
                metadata={},
                success=False,
                error="Invalid request format",
            )

        try:
            # Get or create memory manager for user
            memory_manager = await self._get_memory_manager(request.user_id)

            # Determine operation type
            operation = request.context.get("operation", "store")

            if operation == "store":
                result = await self._store_memory(memory_manager, request)
            elif operation == "recall":
                result = await self._recall_memory(memory_manager, request)
            elif operation == "search":
                result = await self._search_memories(memory_manager, request)
            elif operation == "context":
                result = await self._get_context(memory_manager, request)
            elif operation == "summary":
                result = await self._get_summary(memory_manager, request)
            else:
                result = {"error": f"Unknown operation: {operation}"}

            processing_time = time.time() - start_time

            return BrainResponse(
                request_id=request.request_id,
                brain_type=self.brain_type,
                response=result,
                confidence=1.0 if "error" not in result else 0.0,
                processing_time=processing_time,
                metadata={"operation": operation},
                success="error" not in result,
            )

        except Exception as e:
            self.logger.error(
                "Memory processing failed", request_id=request.request_id, error=str(e)
            )

            return BrainResponse(
                request_id=request.request_id,
                brain_type=self.brain_type,
                response={},
                confidence=0.0,
                processing_time=time.time() - start_time,
                metadata={},
                success=False,
                error=str(e),
            )

    async def _get_memory_manager(self, user_id: str) -> MemoryManager:
        """Get or create memory manager for user."""
        if user_id not in self.memory_managers:
            manager = MemoryManager(user_id, self.settings)
            await manager.initialize()
            self.memory_managers[user_id] = manager

        return self.memory_managers[user_id]

    async def _store_memory(
        self, manager: MemoryManager, request: BrainRequest
    ) -> Dict[str, Any]:
        """Store a new memory."""
        content = request.message
        memory_type = MemoryType(request.context.get("memory_type", "working"))
        importance = MemoryImportance(request.context.get("importance", 3))
        tags = request.context.get("tags", [])
        metadata = request.context.get("metadata", {})

        memory_id = await manager.create_memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags,
            metadata=metadata,
        )

        return {"memory_id": memory_id, "operation": "store", "success": True}

    async def _recall_memory(
        self, manager: MemoryManager, request: BrainRequest
    ) -> Dict[str, Any]:
        """Recall a specific memory."""
        memory_id = request.context.get("memory_id")
        if not memory_id:
            return {"error": "memory_id required for recall operation"}

        memory = await manager.recall_memory(memory_id)
        if not memory:
            return {"error": "Memory not found"}

        return {"memory": memory.to_dict(), "operation": "recall", "success": True}

    async def _search_memories(
        self, manager: MemoryManager, request: BrainRequest
    ) -> Dict[str, Any]:
        """Search memories."""
        query = request.message
        memory_types = request.context.get("memory_types")
        if memory_types:
            memory_types = [MemoryType(mt) for mt in memory_types]

        limit = request.context.get("limit", 10)

        memories = await manager.search_memories(
            query=query, memory_types=memory_types, limit=limit
        )

        return {
            "memories": [m.to_dict() for m in memories],
            "count": len(memories),
            "query": query,
            "operation": "search",
            "success": True,
        }

    async def _get_context(
        self, manager: MemoryManager, request: BrainRequest
    ) -> Dict[str, Any]:
        """Get context memories."""
        context_size = request.context.get("context_size", 20)

        memories = await manager.get_context_memories(context_size)

        return {
            "context_memories": [m.to_dict() for m in memories],
            "count": len(memories),
            "operation": "context",
            "success": True,
        }

    async def _get_summary(
        self, manager: MemoryManager, request: BrainRequest
    ) -> Dict[str, Any]:
        """Get memory system summary."""
        summary = await manager.get_memory_summary()

        return {"summary": summary, "operation": "summary", "success": True}

    async def health_check(self) -> Dict[str, Any]:
        """Check health of memory brain."""
        health = {
            "status": "healthy",
            "active_users": len(self.memory_managers),
            "initialized": self.is_initialized,
        }

        # Check health of each memory manager
        user_health = {}
        for user_id, manager in self.memory_managers.items():
            try:
                summary = await manager.get_memory_summary()
                user_health[user_id] = summary["system_health"]
            except Exception as e:
                user_health[user_id] = {"status": "error", "error": str(e)}

        health["users"] = user_health
        return health

    async def shutdown(self) -> None:
        """Shutdown memory brain and all managers."""
        await super().shutdown()

        for manager in self.memory_managers.values():
            await manager.shutdown()

        self.memory_managers.clear()

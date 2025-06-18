from typing import Dict, List, Optional, Any, Union
import asyncio
from datetime import datetime, timedelta


from src.core.telemetry.logger import StructuredLogger, log_execution_time
from src.core.config.settings import Settings
from src.core.memory.models import (
    MemoryEntry,
    MemoryType,
    MemoryStatus,
    MemoryImportance,
)
from src.core.memory.base_memory import BaseMemoryStore
from src.core.memory.sqlite_memory import SQLiteMemoryStore


class MemoryManager:
    """High-level memory management interface."""

    def __init__(self, user_id: str, settings: "Settings"):
        self.user_id = user_id
        self.settings = settings
        self.logger = StructuredLogger("memory.manager")

        # Initialize different memory stores
        self.stores: Dict[str, BaseMemoryStore] = {}
        self.primary_store = SQLiteMemoryStore(user_id, settings)
        self.stores["primary"] = self.primary_store

        # Memory type configurations
        self.memory_configs = {
            MemoryType.CORE: {"max_entries": 100, "retention_days": None},
            MemoryType.WORKING: {"max_entries": 50, "retention_days": 1},
            MemoryType.SHORT_TERM: {"max_entries": 500, "retention_days": 7},
            MemoryType.LONG_TERM: {"max_entries": 5000, "retention_days": 365},
            MemoryType.EPISODIC: {"max_entries": 1000, "retention_days": 90},
            MemoryType.SEMANTIC: {"max_entries": 10000, "retention_days": None},
            MemoryType.PROCEDURAL: {"max_entries": 200, "retention_days": None},
            MemoryType.EMOTIONAL: {"max_entries": 1000, "retention_days": 180},
        }

    async def initialize(self) -> None:
        """Initialize all memory stores."""
        self.logger.info(f"Initializing memory manager for user {self.user_id}")

        initialization_tasks = []
        for store_name, store in self.stores.items():
            task = asyncio.create_task(self._safe_init_store(store_name, store))
            initialization_tasks.append(task)

        results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
        successful_inits = sum(1 for result in results if result is True)

        self.logger.info(
            "Memory manager initialization completed",
            successful_stores=successful_inits,
            total_stores=len(self.stores),
        )

    async def _safe_init_store(self, store_name: str, store: BaseMemoryStore) -> bool:
        """Safely initialize a memory store."""
        try:
            await store.initialize()
            self.logger.info(f"Memory store initialized", store_name=store_name)
            return True
        except Exception as e:
            self.logger.error(
                f"Memory store initialization failed",
                store_name=store_name,
                error=str(e),
            )
            return False

    @log_execution_time
    async def create_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.WORKING,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expires_in_hours: Optional[int] = None,
    ) -> str:
        """Create a new memory entry."""

        expires_at = None
        if expires_in_hours:
            expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)

        memory = MemoryEntry(
            user_id=self.user_id,
            memory_type=memory_type,
            content=content,
            importance=importance,
            tags=tags or [],
            metadata=metadata or {},
            expires_at=expires_at,
        )

        memory_id = await self.primary_store.store_memory(memory)

        self.logger.info(
            "Memory created",
            memory_id=memory_id,
            memory_type=memory_type.value,
            importance=importance.value,
            content_length=len(content),
        )

        return memory_id

    async def recall_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Recall a specific memory by ID."""
        return await self.primary_store.retrieve_memory(memory_id)

    async def search_memories(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        importance_filter: Optional[MemoryImportance] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Search memories with various filters."""

        results = await self.primary_store.search_memories(
            query=query, memory_types=memory_types, limit=limit
        )

        # Apply importance filter if specified
        if importance_filter:
            results = [
                m for m in results if m.importance.value >= importance_filter.value
            ]

        self.logger.info(
            "Memory search completed",
            query=query,
            results_count=len(results),
            memory_types=[mt.value for mt in memory_types] if memory_types else "all",
        )

        return results

    async def get_context_memories(self, context_size: int = 20) -> List[MemoryEntry]:
        """Get relevant memories for current context."""
        # Get recent working memory
        working_memories = await self.primary_store.get_memories_by_type(
            MemoryType.WORKING, limit=context_size // 2
        )

        # Get important long-term memories
        important_memories = await self.primary_store.get_memories_by_type(
            MemoryType.LONG_TERM, limit=context_size // 2
        )

        # Combine and sort by importance and recency
        all_memories = working_memories + important_memories
        all_memories.sort(
            key=lambda m: (m.importance.value, m.accessed_at), reverse=True
        )

        return all_memories[:context_size]

    async def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[MemoryImportance] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update an existing memory."""
        updates = {}

        if content is not None:
            updates["content"] = content
        if importance is not None:
            updates["importance"] = importance.value
        if tags is not None:
            updates["tags"] = tags
        if metadata is not None:
            updates["metadata"] = metadata

        if not updates:
            return False

        success = await self.primary_store.update_memory(memory_id, updates)

        if success:
            self.logger.info(
                "Memory updated",
                memory_id=memory_id,
                updated_fields=list(updates.keys()),
            )

        return success

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        success = await self.primary_store.delete_memory(memory_id)

        if success:
            self.logger.info("Memory deleted", memory_id=memory_id)

        return success

    async def consolidate_memories(self) -> Dict[str, Any]:
        """Run memory consolidation process."""
        self.logger.info("Starting memory consolidation process")

        consolidation_results = {}

        # Consolidate each store
        for store_name, store in self.stores.items():
            try:
                result = await store.consolidate_memories()
                consolidation_results[store_name] = result
            except Exception as e:
                self.logger.error(
                    f"Consolidation failed for store",
                    store_name=store_name,
                    error=str(e),
                )
                consolidation_results[store_name] = {"error": str(e)}

        # Apply memory type limits
        await self._enforce_memory_limits()

        self.logger.info("Memory consolidation completed")
        return consolidation_results

    async def _enforce_memory_limits(self) -> None:
        """Enforce memory limits for each memory type."""
        for memory_type, config in self.memory_configs.items():
            max_entries = config.get("max_entries")
            if not max_entries:
                continue

            # Get all memories of this type
            memories = await self.primary_store.get_memories_by_type(memory_type)

            if len(memories) > max_entries:
                # Sort by importance and access patterns
                memories.sort(
                    key=lambda m: (m.importance.value, m.access_count, m.created_at),
                    reverse=True,
                )

                # Archive excess memories
                to_archive = memories[max_entries:]
                for memory in to_archive:
                    await self.primary_store.update_memory(
                        memory.id, {"status": MemoryStatus.ARCHIVED.value}
                    )

                self.logger.info(
                    f"Archived excess memories",
                    memory_type=memory_type.value,
                    archived_count=len(to_archive),
                )

    async def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory system summary."""
        stats = await self.primary_store.get_statistics()

        # Add memory type breakdown
        type_breakdown = {}
        for memory_type in MemoryType:
            memories = await self.primary_store.get_memories_by_type(
                memory_type, limit=None
            )
            type_breakdown[memory_type.value] = {
                "count": len(memories),
                "avg_importance": (
                    sum(m.importance.value for m in memories) / len(memories)
                    if memories
                    else 0
                ),
                "most_recent": (
                    max(m.created_at for m in memories).isoformat()
                    if memories
                    else None
                ),
            }

        return {
            "statistics": stats,
            "type_breakdown": type_breakdown,
            "system_health": await self._check_memory_health(),
        }

    async def _check_memory_health(self) -> Dict[str, Any]:
        """Check the health of the memory system."""
        health = {"status": "healthy", "issues": [], "recommendations": []}

        stats = await self.primary_store.get_statistics()
        total_memories = stats.get("total_memories", 0)

        # Check for issues
        if total_memories > 50000:
            health["issues"].append("High memory count - consider consolidation")
            health["recommendations"].append("Run memory consolidation process")

        if stats.get("active", 0) / max(total_memories, 1) < 0.7:
            health["issues"].append("Low ratio of active memories")
            health["recommendations"].append("Clean up archived/deleted memories")

        # Set overall status
        if health["issues"]:
            health["status"] = (
                "needs_attention" if len(health["issues"]) < 3 else "unhealthy"
            )

        return health

    async def backup_memories(self, backup_path: str) -> bool:
        """Create a backup of all memories."""
        try:
            import shutil
            from pathlib import Path

            # For SQLite, we can simply copy the database file
            source_path = self.primary_store.db_path
            backup_file = (
                Path(backup_path)
                / f"memory_backup_{self.user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.db"
            )

            backup_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, backup_file)

            self.logger.info(
                "Memory backup created",
                backup_path=str(backup_file),
                source_size=Path(source_path).stat().st_size,
            )

            return True

        except Exception as e:
            self.logger.error("Memory backup failed", error=str(e))
            return False

    async def restore_memories(self, backup_path: str) -> bool:
        """Restore memories from a backup."""
        try:
            import shutil
            from pathlib import Path

            if not Path(backup_path).exists():
                self.logger.error("Backup file not found", backup_path=backup_path)
                return False

            # Shutdown current store
            await self.primary_store.shutdown()

            # Replace database file
            shutil.copy2(backup_path, self.primary_store.db_path)

            # Reinitialize store
            self.primary_store = SQLiteMemoryStore(self.user_id, self.settings)
            await self.primary_store.initialize()
            self.stores["primary"] = self.primary_store

            self.logger.info("Memory restore completed", backup_path=backup_path)
            return True

        except Exception as e:
            self.logger.error("Memory restore failed", error=str(e))
            return False

    async def shutdown(self) -> None:
        """Shutdown memory manager and all stores."""
        self.logger.info("Shutting down memory manager")

        shutdown_tasks = []
        for store in self.stores.values():
            task = asyncio.create_task(store.shutdown())
            shutdown_tasks.append(task)

        await asyncio.gather(*shutdown_tasks, return_exceptions=True)

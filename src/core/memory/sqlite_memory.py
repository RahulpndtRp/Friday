import sqlite3
import aiosqlite
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from datetime import datetime, timedelta

from src.core.telemetry.logger import StructuredLogger, log_execution_time
from src.core.memory.base_memory import BaseMemoryStore
from src.core.memory.models import (
    MemoryEntry,
    MemoryType,
    MemoryImportance,
    MemoryStatus,
)
from src.core.config.settings import Settings


class SQLiteMemoryStore(BaseMemoryStore):
    """Improved SQLite-based memory store with proper connection handling."""

    def __init__(
        self, user_id: str, settings: "Settings", db_path: Optional[str] = None
    ):
        super().__init__(user_id, settings)
        self.db_path = db_path or f"{settings.memory_db_path}_{user_id}.db"
        self._connection = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize SQLite database with proper error handling."""
        self.logger.info(f"Initializing SQLite memory store", db_path=self.db_path)

        try:
            # Ensure directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

            # Remove any existing database locks
            await self._cleanup_database_locks()

            # Create single connection with proper configuration
            self._connection = await aiosqlite.connect(
                self.db_path, timeout=30.0  # 30 second timeout
            )

            # Configure SQLite for better concurrency
            await self._connection.execute("PRAGMA journal_mode = WAL")
            await self._connection.execute("PRAGMA synchronous = NORMAL")
            await self._connection.execute("PRAGMA cache_size = 10000")
            await self._connection.execute("PRAGMA temp_store = MEMORY")
            await self._connection.execute("PRAGMA busy_timeout = 30000")

            # Create tables
            await self._create_tables()

            self.is_initialized = True
            self.logger.info("SQLite memory store initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize SQLite memory store", error=str(e))
            if self._connection:
                await self._connection.close()
                self._connection = None
            raise

    async def _cleanup_database_locks(self) -> None:
        """Clean up any stale database locks."""
        try:
            # Remove SQLite temporary files that might cause locks
            for suffix in ["-wal", "-shm", "-journal"]:
                temp_file = Path(f"{self.db_path}{suffix}")
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                        self.logger.debug(
                            f"Removed stale lock file", file=str(temp_file)
                        )
                    except:
                        pass  # Ignore if we can't remove it
        except Exception as e:
            self.logger.warning(f"Could not cleanup database locks", error=str(e))

    async def _create_tables(self) -> None:
        """Create necessary database tables with proper FTS setup."""
        async with self._lock:
            try:
                # Main memories table
                await self._connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        memory_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata TEXT DEFAULT '{}',
                        importance INTEGER NOT NULL,
                        tags TEXT DEFAULT '[]',
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL,
                        accessed_at TIMESTAMP NOT NULL,
                        expires_at TIMESTAMP NULL,
                        related_memories TEXT DEFAULT '[]',
                        source_context TEXT DEFAULT '{}',
                        confidence REAL DEFAULT 1.0,
                        status TEXT DEFAULT 'active',
                        access_count INTEGER DEFAULT 0,
                        consolidation_count INTEGER DEFAULT 0
                    )
                """
                )

                # Create indexes
                await self._connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_memories_user_type 
                    ON memories(user_id, memory_type)
                """
                )

                await self._connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_memories_created_at 
                    ON memories(created_at DESC)
                """
                )

                await self._connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_memories_importance 
                    ON memories(importance DESC)
                """
                )

                await self._connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_memories_status 
                    ON memories(status, user_id)
                """
                )

                # Check if FTS table exists and create if needed
                cursor = await self._connection.execute(
                    """
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='memories_fts'
                """
                )
                fts_exists = await cursor.fetchone()

                if not fts_exists:
                    # Create FTS5 virtual table
                    await self._connection.execute(
                        """
                        CREATE VIRTUAL TABLE memories_fts USING fts5(
                            memory_id UNINDEXED,
                            content,
                            tags,
                            content='',
                            contentless_delete=1
                        )
                    """
                    )
                    self.logger.info("Created FTS virtual table")

                await self._connection.commit()
                self.logger.info("Database tables created successfully")

            except Exception as e:
                await self._connection.rollback()
                self.logger.error(f"Failed to create tables", error=str(e))
                raise

    async def _ensure_connection(self) -> None:
        """Ensure database connection is available."""
        if not self._connection or not self.is_initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")

    @log_execution_time
    async def store_memory(self, memory: MemoryEntry) -> str:
        """Store a memory entry in SQLite."""
        await self._ensure_connection()

        self.logger.info(
            f"Storing memory",
            memory_id=memory.id,
            memory_type=memory.memory_type.value,
            importance=memory.importance.value,
        )

        async with self._lock:
            try:
                # Insert into main table
                await self._connection.execute(
                    """
                    INSERT OR REPLACE INTO memories (
                        id, user_id, memory_type, content, metadata, importance, tags,
                        created_at, updated_at, accessed_at, expires_at, related_memories,
                        source_context, confidence, status, access_count, consolidation_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        memory.id,
                        memory.user_id,
                        memory.memory_type.value,
                        memory.content,
                        json.dumps(memory.metadata),
                        memory.importance.value,
                        json.dumps(memory.tags),
                        memory.created_at,
                        memory.updated_at,
                        memory.accessed_at,
                        memory.expires_at,
                        json.dumps(memory.related_memories),
                        json.dumps(memory.source_context),
                        memory.confidence,
                        memory.status.value,
                        memory.access_count,
                        memory.consolidation_count,
                    ),
                )

                # Insert into FTS table
                await self._connection.execute(
                    """
                    INSERT OR REPLACE INTO memories_fts (memory_id, content, tags)
                    VALUES (?, ?, ?)
                """,
                    (memory.id, memory.content, " ".join(memory.tags)),
                )

                await self._connection.commit()
                return memory.id

            except Exception as e:
                await self._connection.rollback()
                self.logger.error(
                    f"Failed to store memory", memory_id=memory.id, error=str(e)
                )
                raise

    async def retrieve_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory by ID."""
        await self._ensure_connection()

        async with self._lock:
            try:
                cursor = await self._connection.execute(
                    """
                    SELECT * FROM memories WHERE id = ? AND user_id = ?
                """,
                    (memory_id, self.user_id),
                )

                row = await cursor.fetchone()
                if not row:
                    return None

                # Update access tracking
                await self._connection.execute(
                    """
                    UPDATE memories 
                    SET accessed_at = ?, access_count = access_count + 1
                    WHERE id = ?
                """,
                    (datetime.utcnow(), memory_id),
                )
                await self._connection.commit()

                return self._row_to_memory(row)

            except Exception as e:
                self.logger.error(
                    f"Failed to retrieve memory", memory_id=memory_id, error=str(e)
                )
                return None

    async def search_memories(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> List[MemoryEntry]:
        """Search memories using content matching and filters."""
        await self._ensure_connection()

        async with self._lock:
            try:
                # Use simple LIKE search instead of FTS to avoid complexity
                # We can implement FTS later once basic functionality works
                sql = """
                    SELECT * FROM memories 
                    WHERE user_id = ? 
                    AND (content LIKE ? OR tags LIKE ?)
                    AND confidence >= ?
                    AND status = 'active'
                """

                params = [self.user_id, f"%{query}%", f"%{query}%", min_confidence]

                if memory_types:
                    type_placeholders = ",".join(["?" for _ in memory_types])
                    sql += f" AND memory_type IN ({type_placeholders})"
                    params.extend([mt.value for mt in memory_types])

                sql += " ORDER BY importance DESC, accessed_at DESC LIMIT ?"
                params.append(limit)

                cursor = await self._connection.execute(sql, params)
                rows = await cursor.fetchall()

                return [self._row_to_memory(row) for row in rows]

            except Exception as e:
                self.logger.error(
                    f"Failed to search memories", query=query, error=str(e)
                )
                return []

    async def get_memories_by_type(
        self, memory_type: MemoryType, limit: Optional[int] = None
    ) -> List[MemoryEntry]:
        """Get memories by type."""
        await self._ensure_connection()

        async with self._lock:
            try:
                sql = """
                    SELECT * FROM memories 
                    WHERE user_id = ? AND memory_type = ? AND status = 'active'
                    ORDER BY importance DESC, created_at DESC
                """
                params = [self.user_id, memory_type.value]

                if limit:
                    sql += " LIMIT ?"
                    params.append(limit)

                cursor = await self._connection.execute(sql, params)
                rows = await cursor.fetchall()

                return [self._row_to_memory(row) for row in rows]

            except Exception as e:
                self.logger.error(
                    f"Failed to get memories by type",
                    memory_type=memory_type.value,
                    error=str(e),
                )
                return []

    async def get_recent_memories(
        self, hours: int = 24, limit: int = 50
    ) -> List[MemoryEntry]:
        """Get recent memories within specified hours."""
        await self._ensure_connection()
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        async with self._lock:
            try:
                cursor = await self._connection.execute(
                    """
                    SELECT * FROM memories 
                    WHERE user_id = ? AND created_at >= ? AND status = 'active'
                    ORDER BY created_at DESC, importance DESC
                    LIMIT ?
                """,
                    (self.user_id, cutoff_time, limit),
                )

                rows = await cursor.fetchall()
                return [self._row_to_memory(row) for row in rows]

            except Exception as e:
                self.logger.error(f"Failed to get recent memories", error=str(e))
                return []

    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory entry."""
        if not updates:
            return False

        await self._ensure_connection()

        async with self._lock:
            try:
                # Build dynamic update query
                set_clauses = []
                params = []

                for key, value in updates.items():
                    if key in [
                        "metadata",
                        "tags",
                        "related_memories",
                        "source_context",
                    ]:
                        set_clauses.append(f"{key} = ?")
                        params.append(json.dumps(value))
                    else:
                        set_clauses.append(f"{key} = ?")
                        params.append(value)

                set_clauses.append("updated_at = ?")
                params.append(datetime.utcnow())

                params.extend([memory_id, self.user_id])

                sql = f"""
                    UPDATE memories 
                    SET {', '.join(set_clauses)}
                    WHERE id = ? AND user_id = ?
                """

                cursor = await self._connection.execute(sql, params)
                await self._connection.commit()

                return cursor.rowcount > 0

            except Exception as e:
                await self._connection.rollback()
                self.logger.error(
                    f"Failed to update memory", memory_id=memory_id, error=str(e)
                )
                return False

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        await self._ensure_connection()

        async with self._lock:
            try:
                # Soft delete by updating status
                cursor = await self._connection.execute(
                    """
                    UPDATE memories 
                    SET status = 'deleted', updated_at = ?
                    WHERE id = ? AND user_id = ?
                """,
                    (datetime.utcnow(), memory_id, self.user_id),
                )

                await self._connection.commit()
                return cursor.rowcount > 0

            except Exception as e:
                await self._connection.rollback()
                self.logger.error(
                    f"Failed to delete memory", memory_id=memory_id, error=str(e)
                )
                return False

    async def consolidate_memories(self) -> Dict[str, Any]:
        """Consolidate and optimize memory storage."""
        self.logger.info("Starting memory consolidation")

        await self._ensure_connection()

        async with self._lock:
            try:
                # Get statistics before consolidation
                stats_before = await self._get_memory_stats()

                # Archive old low-importance memories
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                await self._connection.execute(
                    """
                    UPDATE memories 
                    SET status = 'archived'
                    WHERE user_id = ? AND importance <= 2 AND created_at < ? AND status = 'active'
                """,
                    (self.user_id, cutoff_date),
                )

                # Clean up expired memories
                await self._connection.execute(
                    """
                    UPDATE memories 
                    SET status = 'deleted'
                    WHERE user_id = ? AND expires_at IS NOT NULL AND expires_at < ?
                """,
                    (self.user_id, datetime.utcnow()),
                )

                await self._connection.commit()

                stats_after = await self._get_memory_stats()

                consolidation_result = {
                    "consolidation_time": datetime.utcnow().isoformat(),
                    "before": stats_before,
                    "after": stats_after,
                    "archived": stats_before.get("active", 0)
                    - stats_after.get("active", 0),
                }

                self.logger.info(
                    "Memory consolidation completed",
                    archived_count=consolidation_result["archived"],
                )

                return consolidation_result

            except Exception as e:
                await self._connection.rollback()
                self.logger.error(f"Memory consolidation failed", error=str(e))
                return {"error": str(e)}

    async def get_statistics(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        await self._ensure_connection()

        async with self._lock:
            return await self._get_memory_stats()

    async def _get_memory_stats(self) -> Dict[str, Any]:
        """Get detailed memory statistics."""
        stats = {}

        try:
            # Count by status
            cursor = await self._connection.execute(
                """
                SELECT status, COUNT(*) FROM memories 
                WHERE user_id = ? GROUP BY status
            """,
                (self.user_id,),
            )
            status_counts = dict(await cursor.fetchall())
            stats.update(status_counts)

            # Count by type
            cursor = await self._connection.execute(
                """
                SELECT memory_type, COUNT(*) FROM memories 
                WHERE user_id = ? AND status = 'active' GROUP BY memory_type
            """,
                (self.user_id,),
            )
            type_counts = dict(await cursor.fetchall())
            stats["by_type"] = type_counts

            # Count by importance
            cursor = await self._connection.execute(
                """
                SELECT importance, COUNT(*) FROM memories 
                WHERE user_id = ? AND status = 'active' GROUP BY importance
            """,
                (self.user_id,),
            )
            importance_counts = dict(await cursor.fetchall())
            stats["by_importance"] = importance_counts

            # Total size
            cursor = await self._connection.execute(
                """
                SELECT COUNT(*), 
                       MIN(created_at) as oldest,
                       MAX(created_at) as newest,
                       AVG(access_count) as avg_access
                FROM memories WHERE user_id = ?
            """,
                (self.user_id,),
            )
            total_stats = await cursor.fetchone()

            stats.update(
                {
                    "total_memories": total_stats[0] if total_stats else 0,
                    "oldest_memory": total_stats[1] if total_stats else None,
                    "newest_memory": total_stats[2] if total_stats else None,
                    "average_access_count": total_stats[3] if total_stats else 0,
                }
            )

        except Exception as e:
            self.logger.error(f"Failed to get memory statistics", error=str(e))
            stats["error"] = str(e)

        return stats

    def _row_to_memory(self, row) -> MemoryEntry:
        """Convert database row to MemoryEntry."""
        try:
            return MemoryEntry(
                id=row[0],
                user_id=row[1],
                memory_type=MemoryType(row[2]),
                content=row[3],
                metadata=json.loads(row[4] or "{}"),
                importance=MemoryImportance(row[5]),
                tags=json.loads(row[6] or "[]"),
                created_at=(
                    datetime.fromisoformat(row[7])
                    if isinstance(row[7], str)
                    else row[7]
                ),
                updated_at=(
                    datetime.fromisoformat(row[8])
                    if isinstance(row[8], str)
                    else row[8]
                ),
                accessed_at=(
                    datetime.fromisoformat(row[9])
                    if isinstance(row[9], str)
                    else row[9]
                ),
                expires_at=(
                    datetime.fromisoformat(row[10])
                    if row[10] and isinstance(row[10], str)
                    else row[10]
                ),
                related_memories=json.loads(row[11] or "[]"),
                source_context=json.loads(row[12] or "{}"),
                confidence=row[13],
                status=MemoryStatus(row[14]),
                access_count=row[15],
                consolidation_count=row[16],
            )
        except Exception as e:
            self.logger.error(f"Failed to convert row to memory", error=str(e))
            # Return a basic memory entry if conversion fails
            return MemoryEntry(
                id=row[0],
                user_id=row[1],
                content=row[3],
                memory_type=MemoryType.WORKING,
            )

    async def shutdown(self) -> None:
        """Shutdown and cleanup connections."""
        await super().shutdown()

        async with self._lock:
            if self._connection:
                await self._connection.close()
                self._connection = None

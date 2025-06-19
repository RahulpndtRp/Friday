import sqlite3
import aiosqlite
import json
from typing import List, Dict, Any, Optional
import asyncio
from pathlib import Path

from src.core.chat.models import (
    Conversation,
    ChatMessage,
    MessageStatus,
    MessageType,
    ResponseType,
)
from src.core.utils.datetime_utils import utc_now, safe_parse_datetime
from src.core.telemetry.logger import StructuredLogger, log_execution_time
from src.core.config.settings import Settings


class ChatStore:
    """SQLite-based storage for chat conversations and messages."""

    def __init__(self, user_id: str, settings: "Settings"):
        self.user_id = user_id
        self.settings = settings
        self.logger = StructuredLogger("chat.store")
        self.db_path = f"{settings.memory_db_path}_chat_{user_id}.db"
        self._connection = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize chat database."""
        self.logger.info(f"Initializing chat store", db_path=self.db_path)

        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Create connection
        self._connection = await aiosqlite.connect(self.db_path, timeout=30.0)

        # Configure SQLite
        await self._connection.execute("PRAGMA journal_mode = WAL")
        await self._connection.execute("PRAGMA synchronous = NORMAL")
        await self._connection.execute("PRAGMA foreign_keys = ON")

        # Create tables
        await self._create_tables()

        self.logger.info("Chat store initialized successfully")

    async def _create_tables(self) -> None:
        """Create database tables."""
        # Conversations table
        await self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                message_count INTEGER DEFAULT 0,
                last_activity TIMESTAMP NOT NULL,
                created_at TIMESTAMP NOT NULL,
                settings TEXT DEFAULT '{}',
                tags TEXT DEFAULT '[]',
                is_active BOOLEAN DEFAULT 1,
                is_archived BOOLEAN DEFAULT 0
            )
        """
        )

        # Messages table
        await self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                message_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                response_type TEXT,
                sources TEXT DEFAULT '[]',
                processing_time REAL DEFAULT 0.0,
                token_count INTEGER,
                status TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                parent_message_id TEXT,
                context TEXT DEFAULT '{}',
                FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
            )
        """
        )

        # Create indexes
        await self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversations_user_id 
            ON conversations(user_id)
        """
        )

        await self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversations_last_activity 
            ON conversations(last_activity DESC)
        """
        )

        await self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_id 
            ON messages(conversation_id)
        """
        )

        await self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_created_at 
            ON messages(created_at DESC)
        """
        )

        # Full-text search for messages
        await self._connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                message_id UNINDEXED,
                content,
                content='',
                contentless_delete=1
            )
        """
        )

        await self._connection.commit()

    async def create_conversation(self, conversation: Conversation) -> str:
        """Create a new conversation."""
        async with self._lock:
            try:
                await self._connection.execute(
                    """
                    INSERT INTO conversations (
                        id, user_id, title, description, message_count, last_activity,
                        created_at, settings, tags, is_active, is_archived
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        conversation.id,
                        conversation.user_id,
                        conversation.title,
                        conversation.description,
                        conversation.message_count,
                        conversation.last_activity,
                        conversation.created_at,
                        json.dumps(conversation.settings),
                        json.dumps(conversation.tags),
                        conversation.is_active,
                        conversation.is_archived,
                    ),
                )

                await self._connection.commit()

                self.logger.info(
                    f"Conversation created",
                    conversation_id=conversation.id,
                    title=conversation.title,
                )

                return conversation.id

            except Exception as e:
                await self._connection.rollback()
                self.logger.error(f"Failed to create conversation", error=str(e))
                raise

    async def store_message(self, message: ChatMessage) -> str:
        """Store a chat message."""
        async with self._lock:
            try:
                await self._connection.execute(
                    """
                    INSERT OR REPLACE INTO messages (
                        id, conversation_id, user_id, message_type, content, metadata,
                        response_type, sources, processing_time, token_count, status,
                        created_at, updated_at, parent_message_id, context
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        message.id,
                        message.conversation_id,
                        message.user_id,
                        message.message_type.value,
                        message.content,
                        json.dumps(message.metadata),
                        message.response_type.value if message.response_type else None,
                        json.dumps(message.sources),
                        message.processing_time,
                        message.token_count,
                        message.status.value,
                        message.created_at,
                        message.updated_at,
                        message.parent_message_id,
                        json.dumps(message.context),
                    ),
                )

                # Update FTS index
                await self._connection.execute(
                    """
                    INSERT OR REPLACE INTO messages_fts (message_id, content)
                    VALUES (?, ?)
                """,
                    (message.id, message.content),
                )

                # Update conversation last activity and message count
                await self._connection.execute(
                    """
                    UPDATE conversations 
                    SET last_activity = ?, message_count = message_count + 1
                    WHERE id = ?
                """,
                    (utc_now(), message.conversation_id),
                )

                await self._connection.commit()

                return message.id

            except Exception as e:
                await self._connection.rollback()
                self.logger.error(f"Failed to store message", error=str(e))
                raise

    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID."""
        async with self._lock:
            try:
                cursor = await self._connection.execute(
                    """
                    SELECT * FROM conversations WHERE id = ? AND user_id = ?
                """,
                    (conversation_id, self.user_id),
                )

                row = await cursor.fetchone()
                if not row:
                    return None

                return self._row_to_conversation(row)

            except Exception as e:
                self.logger.error(f"Failed to get conversation", error=str(e))
                return None

    async def get_conversation_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        before_message_id: Optional[str] = None,
    ) -> List[ChatMessage]:
        """Get messages for a conversation."""
        async with self._lock:
            try:
                sql = """
                    SELECT * FROM messages 
                    WHERE conversation_id = ? AND user_id = ?
                """
                params = [conversation_id, self.user_id]

                if before_message_id:
                    sql += " AND created_at < (SELECT created_at FROM messages WHERE id = ?)"
                    params.append(before_message_id)

                sql += " ORDER BY created_at ASC"

                if limit:
                    sql += " LIMIT ?"
                    params.append(limit)

                cursor = await self._connection.execute(sql, params)
                rows = await cursor.fetchall()

                return [self._row_to_message(row) for row in rows]

            except Exception as e:
                self.logger.error(f"Failed to get conversation messages", error=str(e))
                return []

    async def get_user_conversations(
        self, limit: Optional[int] = None, include_archived: bool = False
    ) -> List[Conversation]:
        """Get user's conversations."""
        async with self._lock:
            try:
                sql = "SELECT * FROM conversations WHERE user_id = ?"
                params = [self.user_id]

                if not include_archived:
                    sql += " AND is_archived = 0"

                sql += " ORDER BY last_activity DESC"

                if limit:
                    sql += " LIMIT ?"
                    params.append(limit)

                cursor = await self._connection.execute(sql, params)
                rows = await cursor.fetchall()

                return [self._row_to_conversation(row) for row in rows]

            except Exception as e:
                self.logger.error(f"Failed to get user conversations", error=str(e))
                return []

    async def search_messages(
        self, query: str, conversation_id: Optional[str] = None, limit: int = 20
    ) -> List[ChatMessage]:
        """Search messages by content."""
        async with self._lock:
            try:
                sql = """
                    SELECT m.* FROM messages m
                    JOIN messages_fts fts ON m.id = fts.message_id
                    WHERE fts MATCH ? AND m.user_id = ?
                """
                params = [f'"{query}"', self.user_id]

                if conversation_id:
                    sql += " AND m.conversation_id = ?"
                    params.append(conversation_id)

                sql += " ORDER BY m.created_at DESC LIMIT ?"
                params.append(limit)

                cursor = await self._connection.execute(sql, params)
                rows = await cursor.fetchall()

                return [self._row_to_message(row) for row in rows]

            except Exception as e:
                self.logger.error(f"Failed to search messages", error=str(e))
                return []

    def _row_to_conversation(self, row) -> Conversation:
        """Convert database row to Conversation."""
        return Conversation(
            id=row[0],
            user_id=row[1],
            title=row[2],
            description=row[3] or "",
            message_count=row[4] or 0,
            last_activity=safe_parse_datetime(row[5]) or utc_now(),
            created_at=safe_parse_datetime(row[6]) or utc_now(),
            settings=json.loads(row[7] or "{}"),
            tags=json.loads(row[8] or "[]"),
            is_active=bool(row[9]),
            is_archived=bool(row[10]),
        )

    def _row_to_message(self, row) -> ChatMessage:
        """Convert database row to ChatMessage."""
        return ChatMessage(
            id=row[0],
            conversation_id=row[1],
            user_id=row[2],
            message_type=MessageType(row[3]),
            content=row[4],
            metadata=json.loads(row[5] or "{}"),
            response_type=ResponseType(row[6]) if row[6] else None,
            sources=json.loads(row[7] or "[]"),
            processing_time=row[8] or 0.0,
            token_count=row[9],
            status=MessageStatus(row[10]),
            created_at=safe_parse_datetime(row[11]) or utc_now(),
            updated_at=safe_parse_datetime(row[12]) or utc_now(),
            parent_message_id=row[13],
            context=json.loads(row[14] or "{}"),
        )

    async def shutdown(self) -> None:
        """Shutdown chat store."""
        async with self._lock:
            if self._connection:
                await self._connection.close()
                self._connection = None

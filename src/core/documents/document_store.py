import sqlite3
import aiosqlite
import json
from typing import List, Dict, Any, Optional
import asyncio
import datetime

from core.utils.datetime_utils import utc_now, safe_parse_datetime
from src.core.config.settings import Settings
from src.core.documents.models import (
    ChunkType,
    DocumentChunk,
    DocumentCategory,
    DocumentType,
    Path,
    ProcessedDocument,
    ProcessingStatus,
)
from src.core.telemetry.logger import StructuredLogger


class DocumentStore:
    """SQLite-based storage for processed documents."""

    def __init__(self, user_id: str, settings: Settings):
        self.user_id = user_id
        self.settings = settings
        self.logger = StructuredLogger("document.store")
        self.db_path = f"{settings.memory_db_path}_documents_{user_id}.db"
        self._connection = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize document database."""
        self.logger.info(f"Initializing document store", db_path=self.db_path)

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

        self.logger.info("Document store initialized successfully")

    async def _create_tables(self) -> None:
        """Create database tables."""
        # Documents table
        await self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                document_type TEXT NOT NULL,
                category TEXT NOT NULL,
                title TEXT,
                author TEXT,
                subject TEXT,
                description TEXT,
                language TEXT DEFAULT 'en',
                file_size INTEGER NOT NULL,
                page_count INTEGER,
                word_count INTEGER DEFAULT 0,
                char_count INTEGER DEFAULT 0,
                status TEXT NOT NULL,
                processing_time REAL DEFAULT 0.0,
                error_message TEXT,
                chunk_count INTEGER DEFAULT 0,
                created_at TIMESTAMP NOT NULL,
                processed_at TIMESTAMP,
                last_accessed TIMESTAMP NOT NULL,
                metadata TEXT DEFAULT '{}',
                tags TEXT DEFAULT '[]'
            )
        """
        )

        # Document chunks table
        await self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                page_number INTEGER,
                start_position INTEGER,
                end_position INTEGER,
                parent_section TEXT,
                word_count INTEGER DEFAULT 0,
                char_count INTEGER DEFAULT 0,
                language TEXT DEFAULT 'en',
                confidence REAL DEFAULT 1.0,
                embedding TEXT,
                keywords TEXT DEFAULT '[]',
                entities TEXT DEFAULT '[]',
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
            )
        """
        )

        # Create indexes
        await self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id)
        """
        )

        await self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type)
        """
        )

        await self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category)
        """
        )

        await self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id)
        """
        )

        await self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_type ON document_chunks(chunk_type)
        """
        )

        # Full-text search for chunks
        await self._connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                content,
                keywords,
                content='',
                contentless_delete=1
            )
        """
        )

        await self._connection.commit()

    async def store_document(self, document: ProcessedDocument) -> str:
        """Store a processed document and its chunks."""
        async with self._lock:
            try:
                # Store main document record
                await self._connection.execute(
                    """
                    INSERT OR REPLACE INTO documents (
                        id, user_id, original_filename, file_path, document_type, category,
                        title, author, subject, description, language, file_size, page_count,
                        word_count, char_count, status, processing_time, error_message,
                        chunk_count, created_at, processed_at, last_accessed, metadata, tags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        document.id,
                        document.user_id,
                        document.original_filename,
                        document.file_path,
                        document.document_type.value,
                        document.category.value,
                        document.title,
                        document.author,
                        document.subject,
                        document.description,
                        document.language,
                        document.file_size,
                        document.page_count,
                        document.word_count,
                        document.char_count,
                        document.status.value,
                        document.processing_time,
                        document.error_message,
                        document.chunk_count,
                        document.created_at,
                        document.processed_at,
                        document.last_accessed,
                        json.dumps(document.metadata),
                        json.dumps(document.tags),
                    ),
                )

                # Store chunks
                for chunk in document.chunks:
                    await self._store_chunk(chunk)

                await self._connection.commit()

                self.logger.info(
                    f"Document stored",
                    document_id=document.id,
                    chunks=len(document.chunks),
                )

                return document.id

            except Exception as e:
                await self._connection.rollback()
                self.logger.error(f"Failed to store document", error=str(e))
                raise

    async def _store_chunk(self, chunk: DocumentChunk) -> None:
        """Store a document chunk."""
        # Store chunk data
        await self._connection.execute(
            """
            INSERT OR REPLACE INTO document_chunks (
                id, document_id, chunk_index, chunk_type, content, metadata,
                page_number, start_position, end_position, parent_section,
                word_count, char_count, language, confidence, embedding,
                keywords, entities, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                chunk.id,
                chunk.document_id,
                chunk.chunk_index,
                chunk.chunk_type.value,
                chunk.content,
                json.dumps(chunk.metadata),
                chunk.page_number,
                chunk.start_position,
                chunk.end_position,
                chunk.parent_section,
                chunk.word_count,
                chunk.char_count,
                chunk.language,
                chunk.confidence,
                json.dumps(chunk.embedding) if chunk.embedding else None,
                json.dumps(chunk.keywords),
                json.dumps(chunk.entities),
                chunk.created_at,
            ),
        )

        # Store in FTS table
        await self._connection.execute(
            """
            INSERT OR REPLACE INTO chunks_fts (chunk_id, content, keywords)
            VALUES (?, ?, ?)
        """,
            (chunk.id, chunk.content, " ".join(chunk.keywords)),
        )

    async def get_document(self, document_id: str) -> Optional[ProcessedDocument]:
        """Retrieve a document by ID."""
        async with self._lock:
            try:
                # Get document data
                cursor = await self._connection.execute(
                    """
                    SELECT * FROM documents WHERE id = ? AND user_id = ?
                """,
                    (document_id, self.user_id),
                )

                row = await cursor.fetchone()
                if not row:
                    return None

                # Convert row to document
                document = self._row_to_document(row)

                # Get chunks
                chunks = await self._get_document_chunks(document_id)
                document.chunks = chunks

                # Update access time
                await self._connection.execute(
                    """
                    UPDATE documents SET last_accessed = ? WHERE id = ?
                """,
                    (utc_now(), document_id),
                )  # FIXED
                await self._connection.commit()

                return document

            except Exception as e:
                self.logger.error(
                    f"Failed to get document", document_id=document_id, error=str(e)
                )
                return None

    async def _get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document."""
        cursor = await self._connection.execute(
            """
            SELECT * FROM document_chunks 
            WHERE document_id = ? 
            ORDER BY chunk_index
        """,
            (document_id,),
        )

        rows = await cursor.fetchall()
        return [self._row_to_chunk(row) for row in rows]

    async def search_documents(
        self,
        query: str,
        document_types: Optional[List[DocumentType]] = None,
        categories: Optional[List[DocumentCategory]] = None,
        limit: int = 10,
    ) -> List[ProcessedDocument]:
        """Search documents by content."""
        async with self._lock:
            try:
                sql = """
                    SELECT DISTINCT d.* FROM documents d
                    JOIN document_chunks c ON d.id = c.document_id
                    WHERE d.user_id = ? 
                    AND (d.title LIKE ? OR d.description LIKE ? OR c.content LIKE ?)
                    AND d.status = 'completed'
                """

                params = [self.user_id, f"%{query}%", f"%{query}%", f"%{query}%"]

                if document_types:
                    type_placeholders = ",".join(["?" for _ in document_types])
                    sql += f" AND d.document_type IN ({type_placeholders})"
                    params.extend([dt.value for dt in document_types])

                if categories:
                    cat_placeholders = ",".join(["?" for _ in categories])
                    sql += f" AND d.category IN ({cat_placeholders})"
                    params.extend([cat.value for cat in categories])

                sql += " ORDER BY d.last_accessed DESC LIMIT ?"
                params.append(limit)

                cursor = await self._connection.execute(sql, params)
                rows = await cursor.fetchall()

                documents = []
                for row in rows:
                    doc = self._row_to_document(row)
                    # Get first few chunks for preview
                    preview_chunks = await self._get_document_chunks_limited(doc.id, 3)
                    doc.chunks = preview_chunks
                    documents.append(doc)

                return documents

            except Exception as e:
                self.logger.error(f"Failed to search documents", error=str(e))
                return []

    async def _get_document_chunks_limited(
        self, document_id: str, limit: int
    ) -> List[DocumentChunk]:
        """Get limited number of chunks for preview."""
        cursor = await self._connection.execute(
            """
            SELECT * FROM document_chunks 
            WHERE document_id = ? 
            ORDER BY chunk_index 
            LIMIT ?
        """,
            (document_id, limit),
        )

        rows = await cursor.fetchall()
        return [self._row_to_chunk(row) for row in rows]

    async def search_chunks(
        self,
        query: str,
        document_id: Optional[str] = None,
        chunk_types: Optional[List[ChunkType]] = None,
        limit: int = 20,
    ) -> List[DocumentChunk]:
        """Search document chunks by content."""
        async with self._lock:
            try:
                sql = """
                    SELECT c.* FROM document_chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE d.user_id = ? AND c.content LIKE ?
                """

                params = [self.user_id, f"%{query}%"]

                if document_id:
                    sql += " AND c.document_id = ?"
                    params.append(document_id)

                if chunk_types:
                    type_placeholders = ",".join(["?" for _ in chunk_types])
                    sql += f" AND c.chunk_type IN ({type_placeholders})"
                    params.extend([ct.value for ct in chunk_types])

                sql += " ORDER BY c.chunk_index LIMIT ?"
                params.append(limit)

                cursor = await self._connection.execute(sql, params)
                rows = await cursor.fetchall()

                return [self._row_to_chunk(row) for row in rows]

            except Exception as e:
                self.logger.error(f"Failed to search chunks", error=str(e))
                return []

    async def get_user_documents(
        self,
        status: Optional[ProcessingStatus] = None,
        category: Optional[DocumentCategory] = None,
        limit: Optional[int] = None,
    ) -> List[ProcessedDocument]:
        """Get documents for a user with filters."""
        async with self._lock:
            try:
                sql = "SELECT * FROM documents WHERE user_id = ?"
                params = [self.user_id]

                if status:
                    sql += " AND status = ?"
                    params.append(status.value)

                if category:
                    sql += " AND category = ?"
                    params.append(category.value)

                sql += " ORDER BY last_accessed DESC"

                if limit:
                    sql += " LIMIT ?"
                    params.append(limit)

                cursor = await self._connection.execute(sql, params)
                rows = await cursor.fetchall()

                return [self._row_to_document(row) for row in rows]

            except Exception as e:
                self.logger.error(f"Failed to get user documents", error=str(e))
                return []

    async def get_statistics(self) -> Dict[str, Any]:
        """Get document store statistics."""
        async with self._lock:
            try:
                stats = {}

                # Document counts
                cursor = await self._connection.execute(
                    """
                    SELECT status, COUNT(*) FROM documents 
                    WHERE user_id = ? GROUP BY status
                """,
                    (self.user_id,),
                )
                status_counts = dict(await cursor.fetchall())
                stats["by_status"] = status_counts

                # Type counts
                cursor = await self._connection.execute(
                    """
                    SELECT document_type, COUNT(*) FROM documents 
                    WHERE user_id = ? GROUP BY document_type
                """,
                    (self.user_id,),
                )
                type_counts = dict(await cursor.fetchall())
                stats["by_type"] = type_counts

                # Category counts
                cursor = await self._connection.execute(
                    """
                    SELECT category, COUNT(*) FROM documents 
                    WHERE user_id = ? GROUP BY category
                """,
                    (self.user_id,),
                )
                category_counts = dict(await cursor.fetchall())
                stats["by_category"] = category_counts

                # Total statistics
                cursor = await self._connection.execute(
                    """
                    SELECT 
                        COUNT(*) as total_docs,
                        SUM(word_count) as total_words,
                        SUM(char_count) as total_chars,
                        SUM(chunk_count) as total_chunks,
                        AVG(processing_time) as avg_processing_time
                    FROM documents WHERE user_id = ?
                """,
                    (self.user_id,),
                )

                total_stats = await cursor.fetchone()
                if total_stats:
                    stats.update(
                        {
                            "total_documents": total_stats[0] or 0,
                            "total_words": total_stats[1] or 0,
                            "total_characters": total_stats[2] or 0,
                            "total_chunks": total_stats[3] or 0,
                            "average_processing_time": total_stats[4] or 0,
                        }
                    )

                return stats

            except Exception as e:
                self.logger.error(f"Failed to get statistics", error=str(e))
                return {}

    def _row_to_document(self, row) -> ProcessedDocument:
        """Convert database row to ProcessedDocument."""
        return ProcessedDocument(
            id=row[0],
            user_id=row[1],
            original_filename=row[2],
            file_path=row[3],
            document_type=DocumentType(row[4]),
            category=DocumentCategory(row[5]),
            title=row[6] or "",
            author=row[7],
            subject=row[8],
            description=row[9] or "",
            language=row[10] or "en",
            file_size=row[11],
            page_count=row[12],
            word_count=row[13] or 0,
            char_count=row[14] or 0,
            status=ProcessingStatus(row[15]),
            processing_time=row[16] or 0.0,
            error_message=row[17],
            chunk_count=row[18] or 0,
            created_at=safe_parse_datetime(row[19]) or utc_now(),
            processed_at=safe_parse_datetime(row[20]),
            last_accessed=safe_parse_datetime(row[21]) or utc_now(),
            metadata=json.loads(row[22] or "{}"),
            tags=json.loads(row[23] or "[]"),
        )

    def _row_to_chunk(self, row) -> DocumentChunk:
        """Convert database row to DocumentChunk."""
        return DocumentChunk(
            id=row[0],
            document_id=row[1],
            chunk_index=row[2],
            chunk_type=ChunkType(row[3]),
            content=row[4],
            metadata=json.loads(row[5] or "{}"),
            page_number=row[6],
            start_position=row[7],
            end_position=row[8],
            parent_section=row[9],
            word_count=row[10] or 0,
            char_count=row[11] or 0,
            language=row[12] or "en",
            confidence=row[13] or 1.0,
            embedding=json.loads(row[14]) if row[14] else None,
            keywords=json.loads(row[15] or "[]"),
            entities=json.loads(row[16] or "[]"),
            created_at=safe_parse_datetime(row[17]) or utc_now(),
        )

    async def shutdown(self) -> None:
        """Shutdown document store."""
        async with self._lock:
            if self._connection:
                await self._connection.close()
                self._connection = None

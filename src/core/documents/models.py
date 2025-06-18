from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
import uuid
import json
from pathlib import Path

from src.core.utils.datetime_utils import utc_now


def utc_now() -> datetime:
    """Get current UTC datetime - Python 3.12 compatible."""
    return datetime.now(timezone.utc)


class DocumentType(Enum):
    """Types of documents that can be processed."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"
    HTML = "html"
    CSV = "csv"
    XLSX = "xlsx"
    UNKNOWN = "unknown"


class DocumentCategory(Enum):
    """Categories for document classification."""

    GOVERNMENT = "government"
    ACADEMIC = "academic"
    PERSONAL = "personal"
    PROFESSIONAL = "professional"
    REFERENCE = "reference"
    TEMPORARY = "temporary"
    UNKNOWN = "unknown"


class ProcessingStatus(Enum):
    """Status of document processing."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class ChunkType(Enum):
    """Types of document chunks."""

    PARAGRAPH = "paragraph"
    SECTION = "section"
    TABLE = "table"
    LIST = "list"
    HEADER = "header"
    FOOTER = "footer"
    METADATA = "metadata"
    IMAGE_CAPTION = "image_caption"


@dataclass
class DocumentChunk:
    """A chunk of processed document content."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    chunk_index: int = 0
    chunk_type: ChunkType = ChunkType.PARAGRAPH
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Position and context
    page_number: Optional[int] = None
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    parent_section: Optional[str] = None

    # Processing information
    word_count: int = 0
    char_count: int = 0
    language: str = "en"
    confidence: float = 1.0

    # Embeddings and search
    embedding: Optional[List[float]] = None
    keywords: List[str] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)

    created_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "chunk_type": self.chunk_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "page_number": self.page_number,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "parent_section": self.parent_section,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "language": self.language,
            "confidence": self.confidence,
            "embedding": self.embedding,
            "keywords": self.keywords,
            "entities": self.entities,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ProcessedDocument:
    """A fully processed document with metadata and chunks."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    original_filename: str = ""
    file_path: str = ""
    document_type: DocumentType = DocumentType.UNKNOWN
    category: DocumentCategory = DocumentCategory.UNKNOWN

    # Content information
    title: str = ""
    author: Optional[str] = None
    subject: Optional[str] = None
    description: str = ""
    language: str = "en"

    # File metadata
    file_size: int = 0
    page_count: Optional[int] = None
    word_count: int = 0
    char_count: int = 0

    # Processing information
    status: ProcessingStatus = ProcessingStatus.PENDING
    processing_time: float = 0.0
    error_message: Optional[str] = None

    # Content chunks
    chunks: List[DocumentChunk] = field(default_factory=list)
    chunk_count: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=utc_now)
    processed_at: Optional[datetime] = None
    last_accessed: datetime = field(default_factory=utc_now)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "original_filename": self.original_filename,
            "file_path": self.file_path,
            "document_type": self.document_type.value,
            "category": self.category.value,
            "title": self.title,
            "author": self.author,
            "subject": self.subject,
            "description": self.description,
            "language": self.language,
            "file_size": self.file_size,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "status": self.status.value,
            "processing_time": self.processing_time,
            "error_message": self.error_message,
            "chunk_count": self.chunk_count,
            "created_at": self.created_at.isoformat(),
            "processed_at": (
                self.processed_at.isoformat() if self.processed_at else None
            ),
            "last_accessed": self.last_accessed.isoformat(),
            "metadata": self.metadata,
            "tags": self.tags,
        }

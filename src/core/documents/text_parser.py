from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
import chardet
import re

from src.core.config.settings import Settings
from core.documents.base_parser import BaseDocumentParser
from core.documents.models import (
    ProcessedDocument,
    DocumentChunk,
    DocumentType,
    DocumentCategory,
    ProcessingStatus,
    ChunkType,
    utc_now,
)


class TextParser(BaseDocumentParser):
    """Parser for plain text documents."""

    def __init__(self, settings: "Settings"):
        super().__init__(settings)
        self.supported_types = [DocumentType.TXT, DocumentType.MD]

    async def can_parse(self, file_path: str) -> bool:
        """Check if file is a text document."""
        extension = Path(file_path).suffix.lower()
        return extension in [".txt", ".md", ".markdown", ".text"]

    async def parse_document(
        self, file_path: str, user_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """Parse text document."""
        self.logger.info(f"Parsing text document", file_path=file_path)

        start_time = utc_now()  # FIXED

        try:
            # Detect encoding
            encoding = await self._detect_encoding(file_path)

            # Create document object
            doc = ProcessedDocument(
                user_id=user_id,
                original_filename=Path(file_path).name,
                file_path=file_path,
                document_type=self._detect_document_type(file_path),
                file_size=Path(file_path).stat().st_size,
            )

            doc.status = ProcessingStatus.PROCESSING

            # Read and process content
            with open(file_path, "r", encoding=encoding) as file:
                content = file.read()

            # Create chunks
            chunks = await self._extract_text_content(content, doc.id)
            doc.chunks = chunks
            doc.chunk_count = len(chunks)

            # Calculate statistics
            doc.word_count = sum(chunk.word_count for chunk in chunks)
            doc.char_count = sum(chunk.char_count for chunk in chunks)

            # Set title and classification
            doc.title = Path(file_path).stem
            doc.category = self._classify_document(
                doc.original_filename, content[:1000]
            )

            # Add metadata
            doc.metadata["encoding"] = encoding
            doc.metadata["line_count"] = content.count("\n") + 1

            # Mark as completed
            doc.status = ProcessingStatus.COMPLETED
            doc.processed_at = utc_now()  # FIXED
            doc.processing_time = (doc.processed_at - start_time).total_seconds()

            self.logger.info(
                f"Text parsing completed",
                document_id=doc.id,
                chunks=len(chunks),
                processing_time=doc.processing_time,
            )

            return doc

        except Exception as e:
            self.logger.error(f"Text parsing failed", file_path=file_path, error=str(e))

            doc = ProcessedDocument(
                user_id=user_id,
                original_filename=Path(file_path).name,
                file_path=file_path,
                document_type=self._detect_document_type(file_path),
                status=ProcessingStatus.FAILED,
                error_message=str(e),
                processing_time=(utc_now() - start_time).total_seconds(),  # FIXED
            )
            return doc

    async def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from text file."""
        metadata = {}

        try:
            file_stat = Path(file_path).stat()
            metadata["creation_date"] = datetime.fromtimestamp(
                file_stat.st_ctime, tz=timezone.utc
            ).isoformat()  # FIXED
            metadata["modification_date"] = datetime.fromtimestamp(
                file_stat.st_mtime, tz=timezone.utc
            ).isoformat()  # FIXED

            # Detect encoding
            encoding = await self._detect_encoding(file_path)
            metadata["encoding"] = encoding

        except Exception as e:
            self.logger.warning(f"Failed to extract text metadata", error=str(e))

        return metadata

    async def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding."""
        try:
            with open(file_path, "rb") as file:
                raw_data = file.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                return result.get("encoding", "utf-8")
        except:
            return "utf-8"

    async def _extract_text_content(
        self, content: str, document_id: str
    ) -> List[DocumentChunk]:
        """Extract content from text and create chunks."""
        chunks = []

        # Split into paragraphs first
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        current_section = None

        for para_idx, paragraph in enumerate(paragraphs):
            # Detect if this looks like a header (short line, all caps, etc.)
            chunk_type = ChunkType.PARAGRAPH
            if self._is_likely_header(paragraph):
                chunk_type = ChunkType.HEADER
                current_section = paragraph

            # If paragraph is too long, chunk it further
            if len(paragraph) > 800:
                sub_chunks = self._chunk_text(paragraph, chunk_size=600, overlap=50)
                for sub_chunk in sub_chunks:
                    chunk = DocumentChunk(
                        document_id=document_id,
                        chunk_index=len(chunks),
                        chunk_type=chunk_type,
                        content=sub_chunk,
                        parent_section=current_section,
                        word_count=len(sub_chunk.split()),
                        char_count=len(sub_chunk),
                        metadata={"paragraph_index": para_idx, "is_sub_chunk": True},
                    )
                    chunks.append(chunk)
            else:
                chunk = DocumentChunk(
                    document_id=document_id,
                    chunk_index=len(chunks),
                    chunk_type=chunk_type,
                    content=paragraph,
                    parent_section=current_section,
                    word_count=len(paragraph.split()),
                    char_count=len(paragraph),
                    metadata={"paragraph_index": para_idx},
                )
                chunks.append(chunk)

        return chunks

    def _is_likely_header(self, text: str) -> bool:
        """Heuristic to detect if text is likely a header."""
        # Short lines are more likely to be headers
        if len(text) < 50 and "\n" not in text:
            # All caps or title case
            if text.isupper() or text.istitle():
                return True
            # Starts with numbers (like "1. Introduction")
            if re.match(r"^\d+\.?\s+", text):
                return True
            # Contains markdown header syntax
            if text.startswith("#"):
                return True
        return False

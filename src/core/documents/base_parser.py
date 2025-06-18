from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from src.core.config.settings import Settings
from src.core.telemetry.logger import StructuredLogger
from src.core.documents.models import (
    DocumentCategory,
    DocumentType,
    ProcessedDocument,
    Path,
)


class BaseDocumentParser(ABC):
    """Abstract base class for document parsers."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = StructuredLogger(f"parser.{self.__class__.__name__.lower()}")
        self.supported_types: List[DocumentType] = []

    @abstractmethod
    async def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the given file."""
        pass

    @abstractmethod
    async def parse_document(
        self, file_path: str, user_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """Parse a document and return processed result."""
        pass

    @abstractmethod
    async def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from document."""
        pass

    def _detect_document_type(self, file_path: str) -> DocumentType:
        """Detect document type from file extension."""
        extension = Path(file_path).suffix.lower()
        type_mapping = {
            ".pdf": DocumentType.PDF,
            ".docx": DocumentType.DOCX,
            ".doc": DocumentType.DOCX,
            ".txt": DocumentType.TXT,
            ".md": DocumentType.MD,
            ".html": DocumentType.HTML,
            ".htm": DocumentType.HTML,
            ".csv": DocumentType.CSV,
            ".xlsx": DocumentType.XLSX,
            ".xls": DocumentType.XLSX,
        }
        return type_mapping.get(extension, DocumentType.UNKNOWN)

    def _classify_document(self, filename: str, content: str) -> DocumentCategory:
        """Classify document based on filename and content."""
        filename_lower = filename.lower()
        content_lower = content.lower()

        # Government documents
        gov_keywords = [
            "aadhar",
            "pan",
            "passport",
            "license",
            "certificate",
            "voter",
            "ration",
        ]
        if any(keyword in filename_lower for keyword in gov_keywords):
            return DocumentCategory.GOVERNMENT

        # Academic documents
        academic_keywords = [
            "research",
            "paper",
            "study",
            "thesis",
            "dissertation",
            "journal",
        ]
        if any(keyword in filename_lower for keyword in academic_keywords):
            return DocumentCategory.ACADEMIC

        # Professional documents
        prof_keywords = ["resume", "cv", "report", "proposal", "contract", "agreement"]
        if any(keyword in filename_lower for keyword in prof_keywords):
            return DocumentCategory.PROFESSIONAL

        # Reference documents
        ref_keywords = ["manual", "guide", "documentation", "readme", "instructions"]
        if any(keyword in filename_lower for keyword in ref_keywords):
            return DocumentCategory.REFERENCE

        return DocumentCategory.UNKNOWN

    def _chunk_text(
        self, text: str, chunk_size: int = 1000, overlap: int = 200
    ) -> List[str]:
        """Chunk text into overlapping segments."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                search_start = max(start + chunk_size - overlap, start)
                search_end = min(end + overlap, len(text))

                for i in range(search_end - 1, search_start - 1, -1):
                    if text[i] in ".!?":
                        end = i + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap
            if start >= len(text):
                break

        return chunks

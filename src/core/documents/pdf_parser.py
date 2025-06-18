import PyPDF2
import pdfplumber
from typing import List, Dict, Any, Optional
import re
from datetime import datetime, timezone

from src.core.utils.datetime_utils import utc_now
from src.core.config.settings import Settings
from src.core.documents.models import (
    ChunkType,
    DocumentChunk,
    DocumentType,
    Path,
    ProcessedDocument,
    ProcessingStatus,
)
from src.core.documents.base_parser import BaseDocumentParser


class PDFParser(BaseDocumentParser):
    """Parser for PDF documents."""

    def __init__(self, settings: "Settings"):
        super().__init__(settings)
        self.supported_types = [DocumentType.PDF]

    async def can_parse(self, file_path: str) -> bool:
        """Check if file is a PDF."""
        return Path(file_path).suffix.lower() == ".pdf"

    async def parse_document(
        self, file_path: str, user_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """Parse PDF document."""
        self.logger.info(f"Parsing PDF document", file_path=file_path)

        start_time = datetime.now(timezone.utc)

        try:
            # Create document object
            doc = ProcessedDocument(
                user_id=user_id,
                original_filename=Path(file_path).name,
                file_path=file_path,
                document_type=DocumentType.PDF,
                file_size=Path(file_path).stat().st_size,
            )

            doc.status = ProcessingStatus.PROCESSING

            # Extract metadata
            doc_metadata = await self.extract_metadata(file_path)
            doc.metadata.update(doc_metadata)

            # Extract text and create chunks
            chunks = await self._extract_pdf_content(file_path, doc.id)
            doc.chunks = chunks
            doc.chunk_count = len(chunks)

            # Calculate statistics
            doc.word_count = sum(chunk.word_count for chunk in chunks)
            doc.char_count = sum(chunk.char_count for chunk in chunks)
            doc.page_count = doc_metadata.get("page_count", 0)

            # Set title and classification
            doc.title = doc_metadata.get("title", Path(file_path).stem)
            doc.author = doc_metadata.get("author")
            doc.subject = doc_metadata.get("subject")
            doc.category = self._classify_document(
                doc.original_filename, " ".join(chunk.content for chunk in chunks[:3])
            )

            # Mark as completed
            doc.status = ProcessingStatus.COMPLETED
            doc.processed_at = datetime.now(timezone.utc)
            doc.processing_time = (doc.processed_at - start_time).total_seconds()

            self.logger.info(
                f"PDF parsing completed",
                document_id=doc.id,
                chunks=len(chunks),
                processing_time=doc.processing_time,
            )

            return doc

        except Exception as e:
            self.logger.error(f"PDF parsing failed", file_path=file_path, error=str(e))

            # Return failed document
            doc = ProcessedDocument(
                user_id=user_id,
                original_filename=Path(file_path).name,
                file_path=file_path,
                document_type=DocumentType.PDF,
                status=ProcessingStatus.FAILED,
                error_message=str(e),
                processing_time=(utc_now() - start_time).total_seconds(),
            )
            return doc

    async def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from PDF."""
        metadata = {}

        try:
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)

                # Basic metadata
                metadata["page_count"] = len(pdf_reader.pages)

                # Document information
                if pdf_reader.metadata:
                    info = pdf_reader.metadata
                    metadata["title"] = getattr(info, "title", None)
                    metadata["author"] = getattr(info, "author", None)
                    metadata["subject"] = getattr(info, "subject", None)
                    metadata["creator"] = getattr(info, "creator", None)
                    metadata["producer"] = getattr(info, "producer", None)

                    # Creation and modification dates
                    if hasattr(info, "creation_date"):
                        metadata["creation_date"] = str(info.creation_date)
                    if hasattr(info, "modification_date"):
                        metadata["modification_date"] = str(info.modification_date)

        except Exception as e:
            self.logger.warning(f"Failed to extract PDF metadata", error=str(e))

        return metadata

    async def _extract_pdf_content(
        self, file_path: str, document_id: str
    ) -> List[DocumentChunk]:
        """Extract content from PDF and create chunks."""
        chunks = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract text from page
                    text = page.extract_text()
                    if not text:
                        continue

                    # Clean and normalize text
                    text = self._clean_text(text)

                    # Create chunks from page text
                    page_chunks = self._chunk_text(text, chunk_size=800, overlap=100)

                    for chunk_idx, chunk_text in enumerate(page_chunks):
                        if not chunk_text.strip():
                            continue

                        chunk = DocumentChunk(
                            document_id=document_id,
                            chunk_index=len(chunks),
                            chunk_type=ChunkType.PARAGRAPH,
                            content=chunk_text,
                            page_number=page_num + 1,
                            word_count=len(chunk_text.split()),
                            char_count=len(chunk_text),
                            metadata={
                                "page_source": page_num + 1,
                                "chunk_on_page": chunk_idx,
                            },
                        )

                        chunks.append(chunk)

        except Exception as e:
            self.logger.error(f"Failed to extract PDF content", error=str(e))
            raise

        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove page numbers and headers/footers (simple heuristic)
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            line = line.strip()

            # Skip very short lines that might be page numbers
            if len(line) < 3:
                continue

            # Skip lines that are just numbers (page numbers)
            if line.isdigit():
                continue

            cleaned_lines.append(line)

        return " ".join(cleaned_lines).strip()

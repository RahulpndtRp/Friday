from typing import List, Optional

from src.core.config.settings import Settings
from src.core.documents.models import (
    DocumentType,
    Path,
)
from src.core.telemetry.logger import StructuredLogger
from src.core.documents.base_parser import BaseDocumentParser
from src.core.documents.pdf_parser import PDFParser
from src.core.documents.docx_parser import DOCXParser
from src.core.documents.text_parser import TextParser


class DocumentParserFactory:
    """Factory for creating appropriate document parsers."""

    def __init__(self, settings: "Settings"):
        self.settings = settings
        self.logger = StructuredLogger("parser.factory")

        # Initialize all parsers
        self.parsers = {
            DocumentType.PDF: PDFParser(settings),
            DocumentType.DOCX: DOCXParser(settings),
            DocumentType.TXT: TextParser(settings),
            DocumentType.MD: TextParser(settings),
        }

    async def get_parser(self, file_path: str) -> Optional[BaseDocumentParser]:
        """Get appropriate parser for file."""
        # Detect document type
        doc_type = self._detect_document_type(file_path)

        # Find compatible parser
        for parser_type, parser in self.parsers.items():
            if await parser.can_parse(file_path):
                self.logger.info(
                    f"Selected parser",
                    file_path=file_path,
                    parser_type=parser_type.value,
                )
                return parser

        self.logger.warning(f"No parser available for file", file_path=file_path)
        return None

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

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return [".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".csv", ".xlsx"]

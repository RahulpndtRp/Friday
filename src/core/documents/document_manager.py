from typing import List, Dict, Any, Optional
import asyncio
from pathlib import Path

from src.core.config.settings import Settings
from src.core.documents.models import (
    DocumentChunk,
    DocumentCategory,
    DocumentType,
    Path,
    ProcessedDocument,
    ProcessingStatus,
)
from src.core.telemetry.logger import StructuredLogger
from src.core.documents.parser_factory import DocumentParserFactory
from src.core.documents.document_store import DocumentStore


class DocumentManager:
    """High-level document processing and management interface."""

    def __init__(self, user_id: str, settings: "Settings"):
        self.user_id = user_id
        self.settings = settings
        self.logger = StructuredLogger("document.manager")

        # Initialize components
        self.parser_factory = DocumentParserFactory(settings)
        self.document_store = DocumentStore(user_id, settings)

        # Processing configuration
        self.max_concurrent_processing = 3
        self.processing_semaphore = asyncio.Semaphore(self.max_concurrent_processing)

    async def initialize(self) -> None:
        """Initialize document manager."""
        self.logger.info(f"Initializing document manager for user {self.user_id}")
        await self.document_store.initialize()
        self.logger.info("Document manager initialized successfully")

    async def process_document(
        self,
        file_path: str,
        category: Optional[DocumentCategory] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProcessedDocument:
        """Process a single document."""
        async with self.processing_semaphore:
            self.logger.info(f"Processing document", file_path=file_path)

            try:
                # Validate file exists
                if not Path(file_path).exists():
                    raise FileNotFoundError(f"File not found: {file_path}")

                # Get appropriate parser
                parser = await self.parser_factory.get_parser(file_path)
                if not parser:
                    raise ValueError(f"No parser available for file: {file_path}")

                # Parse document
                processed_doc = await parser.parse_document(
                    file_path, self.user_id, metadata
                )

                # Override category if provided
                if category:
                    processed_doc.category = category

                # Add tags if provided
                if tags:
                    processed_doc.tags.extend(tags)

                # Store in database
                await self.document_store.store_document(processed_doc)

                self.logger.info(
                    f"Document processing completed",
                    document_id=processed_doc.id,
                    status=processed_doc.status.value,
                    chunks=len(processed_doc.chunks),
                )

                return processed_doc

            except Exception as e:
                self.logger.error(
                    f"Document processing failed", file_path=file_path, error=str(e)
                )
                raise

    async def process_multiple_documents(
        self,
        file_paths: List[str],
        category: Optional[DocumentCategory] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ProcessedDocument]:
        """Process multiple documents concurrently."""
        self.logger.info(f"Processing {len(file_paths)} documents")

        # Create processing tasks
        tasks = []
        for file_path in file_paths:
            task = asyncio.create_task(self.process_document(file_path, category, tags))
            tasks.append(task)

        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate successful and failed results
        successful = []
        failed = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed.append({"file_path": file_paths[i], "error": str(result)})
            else:
                successful.append(result)

        self.logger.info(
            f"Batch processing completed",
            successful=len(successful),
            failed=len(failed),
            total=len(file_paths),
        )

        if failed:
            self.logger.warning(
                f"Some documents failed to process", failed_files=failed
            )

        return successful

    async def get_document(self, document_id: str) -> Optional[ProcessedDocument]:
        """Get a document by ID."""
        return await self.document_store.get_document(document_id)

    async def search_documents(
        self,
        query: str,
        document_types: Optional[List[DocumentType]] = None,
        categories: Optional[List[DocumentCategory]] = None,
        limit: int = 10,
    ) -> List[ProcessedDocument]:
        """Search documents."""
        return await self.document_store.search_documents(
            query, document_types, categories, limit
        )

    async def search_content(
        self, query: str, document_id: Optional[str] = None, limit: int = 20
    ) -> List[DocumentChunk]:
        """Search document content chunks."""
        return await self.document_store.search_chunks(query, document_id, None, limit)

    async def get_user_documents(
        self,
        status: Optional[ProcessingStatus] = None,
        category: Optional[DocumentCategory] = None,
        limit: Optional[int] = None,
    ) -> List[ProcessedDocument]:
        """Get user's documents with filters."""
        return await self.document_store.get_user_documents(status, category, limit)

    async def get_document_statistics(self) -> Dict[str, Any]:
        """Get document processing statistics."""
        return await self.document_store.get_statistics()

    async def shutdown(self) -> None:
        """Shutdown document manager."""
        self.logger.info("Shutting down document manager")
        await self.document_store.shutdown()

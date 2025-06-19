import time
from typing import Dict, Any


from core.brain.base_brain import BaseBrain, BrainType, BrainRequest, BrainResponse
from core.documents.document_manager import DocumentManager
from core.documents.models import DocumentCategory, DocumentType, ProcessingStatus
from src.core.config.settings import Settings
from src.core.documents.models import (
    DocumentCategory,
    DocumentType,
    ProcessingStatus,
)


class DocumentBrain(BaseBrain):
    """Brain component responsible for document operations."""

    def __init__(self, settings: "Settings"):
        super().__init__(
            BrainType.KNOWLEDGE, settings
        )  # Using KNOWLEDGE type for documents
        self.document_managers: Dict[str, DocumentManager] = {}

    async def initialize(self) -> None:
        """Initialize the document brain."""
        self.logger.info("Initializing Document Brain")
        self.is_initialized = True

    async def process(self, request: BrainRequest) -> BrainResponse:
        """Process document-related requests."""
        start_time = time.time()

        if not self._validate_request(request):
            return BrainResponse(
                request_id=request.request_id,
                brain_type=self.brain_type,
                response={},
                confidence=0.0,
                processing_time=time.time() - start_time,
                metadata={},
                success=False,
                error="Invalid request format",
            )

        try:
            # Get or create document manager for user
            doc_manager = await self._get_document_manager(request.user_id)

            # Determine operation type
            operation = request.context.get("operation", "search")

            if operation == "process":
                result = await self._process_document(doc_manager, request)
            elif operation == "search":
                result = await self._search_documents(doc_manager, request)
            elif operation == "get":
                result = await self._get_document(doc_manager, request)
            elif operation == "list":
                result = await self._list_documents(doc_manager, request)
            elif operation == "stats":
                result = await self._get_statistics(doc_manager, request)
            else:
                result = {"error": f"Unknown operation: {operation}"}

            processing_time = time.time() - start_time

            return BrainResponse(
                request_id=request.request_id,
                brain_type=self.brain_type,
                response=result,
                confidence=1.0 if "error" not in result else 0.0,
                processing_time=processing_time,
                metadata={"operation": operation},
                success="error" not in result,
            )

        except Exception as e:
            self.logger.error(
                "Document processing failed",
                request_id=request.request_id,
                error=str(e),
            )

            return BrainResponse(
                request_id=request.request_id,
                brain_type=self.brain_type,
                response={},
                confidence=0.0,
                processing_time=time.time() - start_time,
                metadata={},
                success=False,
                error=str(e),
            )

    async def _get_document_manager(self, user_id: str) -> DocumentManager:
        """Get or create document manager for user."""
        if user_id not in self.document_managers:
            manager = DocumentManager(user_id, self.settings)
            await manager.initialize()
            self.document_managers[user_id] = manager

        return self.document_managers[user_id]

    async def _process_document(
        self, manager: DocumentManager, request: BrainRequest
    ) -> Dict[str, Any]:
        """Process a document."""
        file_path = request.context.get("file_path")
        if not file_path:
            return {"error": "file_path required for process operation"}

        category = request.context.get("category")
        if category:
            category = DocumentCategory(category)

        tags = request.context.get("tags", [])
        metadata = request.context.get("metadata", {})

        try:
            processed_doc = await manager.process_document(
                file_path=file_path, category=category, tags=tags, metadata=metadata
            )

            return {
                "document": processed_doc.to_dict(),
                "operation": "process",
                "success": True,
            }

        except Exception as e:
            return {"error": str(e), "operation": "process"}

    async def _search_documents(
        self, manager: DocumentManager, request: BrainRequest
    ) -> Dict[str, Any]:
        """Search documents."""
        query = request.message

        document_types = request.context.get("document_types")
        if document_types:
            document_types = [DocumentType(dt) for dt in document_types]

        categories = request.context.get("categories")
        if categories:
            categories = [DocumentCategory(cat) for cat in categories]

        limit = request.context.get("limit", 10)

        documents = await manager.search_documents(
            query=query,
            document_types=document_types,
            categories=categories,
            limit=limit,
        )

        return {
            "documents": [doc.to_dict() for doc in documents],
            "count": len(documents),
            "query": query,
            "operation": "search",
            "success": True,
        }

    async def _get_document(
        self, manager: DocumentManager, request: BrainRequest
    ) -> Dict[str, Any]:
        """Get a specific document."""
        document_id = request.context.get("document_id")
        if not document_id:
            return {"error": "document_id required for get operation"}

        document = await manager.get_document(document_id)
        if not document:
            return {"error": "Document not found"}

        return {
            "document": document.to_dict(),
            "chunks": [chunk.to_dict() for chunk in document.chunks],
            "operation": "get",
            "success": True,
        }

    async def _list_documents(
        self, manager: DocumentManager, request: BrainRequest
    ) -> Dict[str, Any]:
        """List user documents."""
        status = request.context.get("status")
        if status:
            status = ProcessingStatus(status)

        category = request.context.get("category")
        if category:
            category = DocumentCategory(category)

        limit = request.context.get("limit")

        documents = await manager.get_user_documents(
            status=status, category=category, limit=limit
        )

        return {
            "documents": [doc.to_dict() for doc in documents],
            "count": len(documents),
            "operation": "list",
            "success": True,
        }

    async def _get_statistics(
        self, manager: DocumentManager, request: BrainRequest
    ) -> Dict[str, Any]:
        """Get document statistics."""
        stats = await manager.get_document_statistics()

        return {"statistics": stats, "operation": "stats", "success": True}

    async def health_check(self) -> Dict[str, Any]:
        """Check health of document brain."""
        health = {
            "status": "healthy",
            "active_users": len(self.document_managers),
            "initialized": self.is_initialized,
        }

        # Check health of each document manager
        user_health = {}
        for user_id, manager in self.document_managers.items():
            try:
                stats = await manager.get_document_statistics()
                user_health[user_id] = {
                    "status": "healthy",
                    "total_documents": stats.get("total_documents", 0),
                    "completed_documents": stats.get("by_status", {}).get(
                        "completed", 0
                    ),
                }
            except Exception as e:
                user_health[user_id] = {"status": "error", "error": str(e)}

        health["users"] = user_health
        return health

    async def shutdown(self) -> None:
        """Shutdown document brain and all managers."""
        await super().shutdown()

        for manager in self.document_managers.values():
            await manager.shutdown()

        self.document_managers.clear()

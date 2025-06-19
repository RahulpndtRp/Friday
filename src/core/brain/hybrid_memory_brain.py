"""
Enhanced Memory Brain using memrp's proven memory core
Replaces FRIDAY's SQLite-based memory with semantic vector memory
"""

import time
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.core.brain.base_brain import BaseBrain, BrainType, BrainRequest, BrainResponse
from src.core.config.settings import Settings
from src.core.telemetry.logger import StructuredLogger

# Import memrp components
from my_mem.memory.main import Memory, AsyncMemory
from my_mem.rag.rag_pipeline import RAGPipeline, AsyncRAGPipeline
from my_mem.configs.base import MemoryConfig


class HybridMemoryBrain(BaseBrain):
    """
    Enhanced Memory Brain that combines:
    - memrp's semantic memory (short-term + long-term with FAISS)
    - FRIDAY's brain architecture
    - Intelligent fact extraction and reconciliation
    """

    def __init__(self, settings: Settings):
        super().__init__(BrainType.MEMORY, settings)

        # Configure memrp for FRIDAY integration
        memory_config = self._build_memory_config(settings)

        # Initialize memrp components
        self.memory_core = AsyncMemory(memory_config)
        self.rag_pipeline = AsyncRAGPipeline(self.memory_core, top_k=5)

        # User-specific memory instances (if needed for isolation)
        self.user_memories: Dict[str, AsyncMemory] = {}

        self.logger = StructuredLogger("brain.hybrid_memory")

    def _build_memory_config(self, settings: Settings) -> MemoryConfig:
        """Build memrp config from FRIDAY settings"""

        # Configure LLM based on FRIDAY's settings
        llm_config = {
            "provider": "openai_async",  # Use async for brain architecture
            "config": {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "api_key": settings.openai_api_key,
                "max_tokens": 2000,
            },
        }

        # Configure vector store for local storage
        vector_config = {
            "provider": "faiss",
            "config": {
                "path": f"{settings.memory_db_path}_vectors",
                "collection_name": "friday_memory",
                "embedding_model_dims": 1536,
                "metric_type": "IP",
            },
        }

        # Configure embeddings
        embedder_config = {
            "provider": "openai",
            "config": {
                "model": "text-embedding-ada-002",
                "api_key": settings.openai_api_key,
            },
        }

        return MemoryConfig(
            llm=llm_config,
            vector_store=vector_config,
            embedder=embedder_config,
            history_db_path=f"{settings.memory_db_path}_history.db",
        )

    async def initialize(self) -> None:
        """Initialize the hybrid memory brain"""
        self.logger.info("Initializing Hybrid Memory Brain with memrp core")

        try:
            # memrp components are initialized on first use
            self.is_initialized = True
            self.logger.info("Hybrid Memory Brain initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Hybrid Memory Brain", error=str(e))
            raise

    async def process(self, request: BrainRequest) -> BrainResponse:
        """Process memory-related requests using memrp's proven logic"""
        start_time = time.time()

        if not self._validate_request(request):
            return self._error_response(request, "Invalid request format", start_time)

        try:
            # Get user-specific memory if needed
            memory_core = await self._get_user_memory(request.user_id)

            # Determine operation type
            operation = request.context.get("operation", "store")

            # Route to appropriate handler
            if operation == "store":
                result = await self._store_memory(memory_core, request)
            elif operation == "recall":
                result = await self._recall_memory(memory_core, request)
            elif operation == "search":
                result = await self._search_memories(memory_core, request)
            elif operation == "rag_query":
                result = await self._rag_query(memory_core, request)
            elif operation == "get_context":
                result = await self._get_context(memory_core, request)
            elif operation == "consolidate":
                result = await self._consolidate_memories(memory_core, request)
            else:
                result = {"error": f"Unknown operation: {operation}"}

            processing_time = time.time() - start_time
            confidence = 1.0 if "error" not in result else 0.0

            return BrainResponse(
                request_id=request.request_id,
                brain_type=self.brain_type,
                response=result,
                confidence=confidence,
                processing_time=processing_time,
                metadata={"operation": operation, "user_id": request.user_id},
                success="error" not in result,
            )

        except Exception as e:
            self.logger.error(
                "Memory processing failed",
                request_id=request.request_id,
                user_id=request.user_id,
                error=str(e),
            )
            return self._error_response(request, str(e), start_time)

    async def _get_user_memory(self, user_id: str) -> AsyncMemory:
        """Get or create user-specific memory instance"""
        if user_id not in self.user_memories:
            # For now, use shared memory core
            # Later can implement user isolation if needed
            return self.memory_core
        return self.user_memories[user_id]

    async def _store_memory(
        self, memory_core: AsyncMemory, request: BrainRequest
    ) -> Dict[str, Any]:
        """Store new memory using memrp's fact extraction"""
        message = request.message
        infer = request.context.get("infer", True)

        try:
            # Use memrp's add method with fact extraction
            result = await memory_core.add(
                message=message, user_id=request.user_id, infer=infer
            )

            return {
                "operation": "store",
                "success": True,
                "results": result.get("results", []),
                "message": f"Stored {len(result.get('results', []))} memory items",
            }

        except Exception as e:
            return {"error": f"Failed to store memory: {str(e)}"}

    async def _recall_memory(
        self, memory_core: AsyncMemory, request: BrainRequest
    ) -> Dict[str, Any]:
        """Recall specific memories by search"""
        query = request.message
        limit = request.context.get("limit", 5)

        try:
            # Use memrp's search for recall
            results = await memory_core.search(
                query=query, user_id=request.user_id, limit=limit
            )

            return {
                "operation": "recall",
                "success": True,
                "memories": results.get("results", []),
                "count": len(results.get("results", [])),
                "query": query,
            }

        except Exception as e:
            return {"error": f"Failed to recall memories: {str(e)}"}

    async def _search_memories(
        self, memory_core: AsyncMemory, request: BrainRequest
    ) -> Dict[str, Any]:
        """Search memories with advanced filtering"""
        query = request.message
        limit = request.context.get("limit", 10)
        ltm_threshold = request.context.get("ltm_threshold", 0.75)

        try:
            results = await memory_core.search(
                query=query,
                user_id=request.user_id,
                limit=limit,
                ltm_threshold=ltm_threshold,
            )

            return {
                "operation": "search",
                "success": True,
                "memories": results.get("results", []),
                "count": len(results.get("results", [])),
                "query": query,
                "threshold": ltm_threshold,
            }

        except Exception as e:
            return {"error": f"Failed to search memories: {str(e)}"}

    async def _rag_query(
        self, memory_core: AsyncMemory, request: BrainRequest
    ) -> Dict[str, Any]:
        """Perform RAG query for context-aware responses"""
        query = request.message

        try:
            # Get RAG pipeline for this memory core
            rag = AsyncRAGPipeline(memory_core, top_k=5)

            # Perform RAG query
            result = await rag.query(query, user_id=request.user_id)

            return {
                "operation": "rag_query",
                "success": True,
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "query": query,
            }

        except Exception as e:
            return {"error": f"Failed RAG query: {str(e)}"}

    async def _get_context(
        self, memory_core: AsyncMemory, request: BrainRequest
    ) -> Dict[str, Any]:
        """Get contextual memories for conversation"""
        context_size = request.context.get("context_size", 10)

        try:
            # Use search with recent bias
            results = await memory_core.search(
                query=request.message,
                user_id=request.user_id,
                limit=context_size,
                ltm_threshold=0.5,  # Lower threshold for context
            )

            return {
                "operation": "get_context",
                "success": True,
                "context_memories": results.get("results", []),
                "count": len(results.get("results", [])),
                "context_size": context_size,
            }

        except Exception as e:
            return {"error": f"Failed to get context: {str(e)}"}

    async def _consolidate_memories(
        self, memory_core: AsyncMemory, request: BrainRequest
    ) -> Dict[str, Any]:
        """Consolidate and optimize memory storage"""
        try:
            # For memrp, consolidation happens automatically during add operations
            # We can add custom consolidation logic here if needed

            return {
                "operation": "consolidate",
                "success": True,
                "message": "Memory consolidation completed (automatic with memrp)",
            }

        except Exception as e:
            return {"error": f"Failed to consolidate memories: {str(e)}"}

    def _error_response(
        self, request: BrainRequest, error_msg: str, start_time: float
    ) -> BrainResponse:
        """Create standardized error response"""
        return BrainResponse(
            request_id=request.request_id,
            brain_type=self.brain_type,
            response={"error": error_msg},
            confidence=0.0,
            processing_time=time.time() - start_time,
            metadata={},
            success=False,
            error=error_msg,
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check health of hybrid memory brain"""
        health = {
            "status": "healthy",
            "memory_core": "memrp_async",
            "initialized": self.is_initialized,
            "active_users": len(self.user_memories),
        }

        try:
            # Test basic memory operation
            test_result = await self.memory_core.search(
                "test", user_id="health_check", limit=1
            )
            health["memory_responsive"] = True
            health["vector_store"] = "operational"

        except Exception as e:
            health["status"] = "unhealthy"
            health["memory_responsive"] = False
            health["error"] = str(e)

        return health

    async def shutdown(self) -> None:
        """Shutdown hybrid memory brain"""
        await super().shutdown()

        # Clean shutdown of memrp components
        try:
            if hasattr(self.memory_core, "shutdown"):
                await self.memory_core.shutdown()
        except Exception as e:
            self.logger.warning(f"Error during memory core shutdown: {e}")

        self.user_memories.clear()
        self.logger.info("Hybrid Memory Brain shutdown complete")

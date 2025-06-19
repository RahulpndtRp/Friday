"""
Direct fix for FRIDAY's memory system - replace existing memory components
This fixes the integration by directly modifying the existing chat manager
"""

# Step 1: Replace src/core/chat/chat_manager.py with this enhanced version

import time
import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator, Union

from src.core.chat.models import (
    Conversation,
    ChatMessage,
    MessageStatus,
    MessageType,
    ResponseType,
    ChatMessageResponse,
    StreamChunk,
)
from src.core.chat.chat_store import ChatStore
from src.core.llm.model_router import ModelRouter
from src.core.utils.datetime_utils import utc_now
from src.core.telemetry.logger import StructuredLogger, log_execution_time
from src.core.config.settings import Settings

# Import memrp components directly
try:
    from my_mem.memory.main import AsyncMemory
    from my_mem.rag.rag_pipeline import AsyncRAGPipeline
    from my_mem.configs.base import MemoryConfig

    MEMRP_AVAILABLE = True
except ImportError:
    MEMRP_AVAILABLE = False
    print("⚠️  memrp not available, using fallback memory")


class EnhancedChatManager:
    """
    Enhanced Chat Manager with direct memrp integration
    This replaces your existing ChatManager
    """

    def __init__(self, user_id: str, settings: Settings):
        self.user_id = user_id
        self.settings = settings
        self.logger = StructuredLogger("chat.manager")

        # Initialize components
        self.chat_store = ChatStore(user_id, settings)
        self.model_router = ModelRouter(settings)

        # Initialize memrp memory system
        self.memory_system = None
        self.rag_pipeline = None
        self._initialize_memory_system()

        # Legacy components (kept for compatibility)
        self.brain_orchestrator = None
        self.memory_manager = None
        self.document_manager = None

        # Configuration
        self.max_context_length = 4000
        self.default_conversation_title = "New Conversation"

    def _initialize_memory_system(self):
        """Initialize the memrp memory system"""
        if not MEMRP_AVAILABLE:
            self.logger.warning("memrp not available, memory features disabled")
            return

        try:
            # Configure memrp for FRIDAY
            memory_config = MemoryConfig(
                llm={
                    "provider": "openai_async",
                    "config": {
                        "model": "gpt-4o-mini",
                        "temperature": 0.1,
                        "api_key": self.settings.openai_api_key,
                        "max_tokens": 2000,
                    },
                },
                vector_store={
                    "provider": "faiss",
                    "config": {
                        "path": ".friday_memory",
                        "collection_name": "friday_memories",
                        "embedding_model_dims": 1536,
                    },
                },
                embedder={
                    "provider": "openai",
                    "config": {
                        "model": "text-embedding-ada-002",
                        "api_key": self.settings.openai_api_key,
                    },
                },
            )

            # Initialize memory and RAG
            self.memory_system = AsyncMemory(memory_config)
            self.rag_pipeline = AsyncRAGPipeline(self.memory_system, top_k=5)

            self.logger.info("✅ memrp memory system initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize memrp memory system: {e}")
            self.memory_system = None
            self.rag_pipeline = None

    async def initialize(self) -> None:
        """Initialize chat manager."""
        self.logger.info(f"Initializing enhanced chat manager for user {self.user_id}")

        # Initialize storage and LLM
        await self.chat_store.initialize()
        await self.model_router.initialize()

        self.logger.info("Enhanced chat manager initialized successfully")

    def set_dependencies(
        self, brain_orchestrator=None, memory_manager=None, document_manager=None
    ):
        """Set dependencies (kept for compatibility)"""
        self.brain_orchestrator = brain_orchestrator
        self.memory_manager = memory_manager
        self.document_manager = document_manager

    async def send_message(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        stream: bool = True,
        **kwargs,
    ) -> Union[ChatMessageResponse, AsyncGenerator[StreamChunk, None]]:
        """Send a message and get response with memory integration."""
        start_time = time.time()

        try:
            # Create or get conversation
            if not conversation_id:
                conversation = await self._create_new_conversation(message)
                conversation_id = conversation.id
            else:
                conversation = await self.chat_store.get_conversation(conversation_id)
                if not conversation:
                    raise ValueError(f"Conversation {conversation_id} not found")

            # Store user message in memory system FIRST
            if self.memory_system:
                try:
                    await self.memory_system.add(
                        message=message,
                        user_id=self.user_id,
                        infer=True,  # Enable fact extraction
                    )
                    self.logger.info("✅ Message stored in memory system")
                except Exception as e:
                    self.logger.warning(f"Failed to store in memory system: {e}")

            # Create user message record
            user_message = ChatMessage(
                conversation_id=conversation_id,
                user_id=self.user_id,
                message_type=MessageType.USER,
                content=message,
                status=MessageStatus.COMPLETED,
            )
            await self.chat_store.store_message(user_message)

            # Create assistant message
            assistant_message = ChatMessage(
                conversation_id=conversation_id,
                user_id=self.user_id,
                message_type=MessageType.ASSISTANT,
                content="",
                status=MessageStatus.PROCESSING,
                parent_message_id=user_message.id,
            )

            if stream:
                return self._stream_response_with_memory(
                    assistant_message, message, conversation_id, start_time
                )
            else:
                return await self._complete_response_with_memory(
                    assistant_message, message, conversation_id, start_time
                )

        except Exception as e:
            self.logger.error(f"Failed to send message", error=str(e))
            raise

    async def _complete_response_with_memory(
        self,
        assistant_message: ChatMessage,
        user_message: str,
        conversation_id: str,
        start_time: float,
    ) -> ChatMessageResponse:
        """Generate complete response using memory system."""
        try:
            # Get memory-enhanced context
            context = await self._build_memory_context(user_message, conversation_id)

            # Use RAG if available, otherwise fallback to standard LLM
            if self.rag_pipeline:
                try:
                    # Use RAG pipeline for memory-aware response
                    rag_result = await self.rag_pipeline.query(
                        user_message, user_id=self.user_id
                    )
                    response_text = rag_result.get("answer", "")
                    sources = rag_result.get("sources", [])

                    self.logger.info(
                        f"✅ RAG response generated with {len(sources)} sources"
                    )

                except Exception as e:
                    self.logger.warning(f"RAG failed, using fallback: {e}")
                    response_text = await self._fallback_response(user_message, context)
                    sources = []
            else:
                response_text = await self._fallback_response(user_message, context)
                sources = []

            # Update assistant message
            assistant_message.content = response_text
            assistant_message.status = MessageStatus.COMPLETED
            assistant_message.processing_time = time.time() - start_time
            assistant_message.response_type = ResponseType.TEXT
            assistant_message.sources = [
                {"type": "memory", "content": s.get("text", "")} for s in sources
            ]
            assistant_message.token_count = await self.model_router.count_tokens(
                response_text
            )
            assistant_message.updated_at = utc_now()

            # Store assistant message
            await self.chat_store.store_message(assistant_message)

            return ChatMessageResponse(
                message_id=assistant_message.id,
                conversation_id=conversation_id,
                response=response_text,
                response_type=assistant_message.response_type.value,
                sources=assistant_message.sources,
                metadata=assistant_message.metadata,
                processing_time=assistant_message.processing_time,
                token_count=assistant_message.token_count,
                status=assistant_message.status.value,
            )

        except Exception as e:
            assistant_message.status = MessageStatus.FAILED
            assistant_message.metadata["error"] = str(e)
            await self.chat_store.store_message(assistant_message)
            raise

    async def _stream_response_with_memory(
        self,
        assistant_message: ChatMessage,
        user_message: str,
        conversation_id: str,
        start_time: float,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Generate streaming response with memory integration."""
        try:
            assistant_message.status = MessageStatus.STREAMING
            await self.chat_store.store_message(assistant_message)

            # Use RAG streaming if available
            if self.rag_pipeline:
                try:
                    full_response = ""
                    chunk_count = 0

                    async for chunk_content in self.rag_pipeline.stream_query(
                        user_message, user_id=self.user_id
                    ):
                        chunk_count += 1
                        full_response += chunk_content

                        yield StreamChunk(
                            chunk_id=f"{assistant_message.id}_{chunk_count}",
                            message_id=assistant_message.id,
                            content=chunk_content,
                            is_final=False,
                            metadata={
                                "chunk_number": chunk_count,
                                "source": "memory_rag",
                            },
                        )

                    # Send final chunk
                    processing_time = time.time() - start_time
                    yield StreamChunk(
                        chunk_id=f"{assistant_message.id}_final",
                        message_id=assistant_message.id,
                        content="",
                        is_final=True,
                        metadata={
                            "total_chunks": chunk_count,
                            "processing_time": processing_time,
                            "memory_enhanced": True,
                        },
                    )

                    # Store final message
                    assistant_message.content = full_response
                    assistant_message.status = MessageStatus.COMPLETED
                    assistant_message.processing_time = processing_time
                    assistant_message.response_type = ResponseType.TEXT
                    assistant_message.updated_at = utc_now()
                    await self.chat_store.store_message(assistant_message)

                    return

                except Exception as e:
                    self.logger.warning(f"Memory streaming failed: {e}")

            # Fallback to standard streaming
            context = await self._build_memory_context(user_message, conversation_id)
            llm_messages = await self._prepare_llm_messages(
                user_message, conversation_id, context
            )

            response_stream = await self.model_router.generate_response(
                messages=llm_messages, context=context, stream=True
            )

            full_response = ""
            chunk_count = 0

            async for chunk_content in response_stream:
                chunk_count += 1
                full_response += chunk_content

                yield StreamChunk(
                    chunk_id=f"{assistant_message.id}_{chunk_count}",
                    message_id=assistant_message.id,
                    content=chunk_content,
                    is_final=False,
                    metadata={"chunk_number": chunk_count},
                )

            # Final chunk
            processing_time = time.time() - start_time
            yield StreamChunk(
                chunk_id=f"{assistant_message.id}_final",
                message_id=assistant_message.id,
                content="",
                is_final=True,
                metadata={
                    "total_chunks": chunk_count,
                    "processing_time": processing_time,
                },
            )

            # Store final message
            assistant_message.content = full_response
            assistant_message.status = MessageStatus.COMPLETED
            assistant_message.processing_time = processing_time
            assistant_message.response_type = ResponseType.TEXT
            assistant_message.updated_at = utc_now()
            await self.chat_store.store_message(assistant_message)

        except Exception as e:
            assistant_message.status = MessageStatus.FAILED
            assistant_message.metadata["error"] = str(e)
            await self.chat_store.store_message(assistant_message)

            yield StreamChunk(
                chunk_id=f"{assistant_message.id}_error",
                message_id=assistant_message.id,
                content=f"Error: {str(e)}",
                is_final=True,
                metadata={"error": True},
            )

    async def _build_memory_context(
        self, user_message: str, conversation_id: str
    ) -> Dict[str, Any]:
        """Build context using memory system."""
        context = {
            "sources": [],
            "memory_context": [],
            "conversation_context": [],
        }

        try:
            # Get conversation history
            recent_messages = await self.chat_store.get_conversation_messages(
                conversation_id, limit=10
            )
            context["conversation_context"] = [
                {"role": msg.message_type.value, "content": msg.content}
                for msg in recent_messages
            ]

            # Get memory context
            if self.memory_system:
                try:
                    memory_results = await self.memory_system.search(
                        query=user_message, user_id=self.user_id, limit=5
                    )

                    memories = memory_results.get("results", [])
                    context["memory_context"] = [
                        {
                            "type": "memory",
                            "content": mem.get("memory", ""),
                            "score": mem.get("score", 0.0),
                        }
                        for mem in memories
                    ]

                    context["sources"] = [
                        {
                            "type": "memory",
                            "content": mem.get("memory", "")[:100] + "...",
                            "score": mem.get("score", 0.0),
                        }
                        for mem in memories
                    ]

                    self.logger.info(f"✅ Retrieved {len(memories)} memory contexts")

                except Exception as e:
                    self.logger.warning(f"Failed to get memory context: {e}")

        except Exception as e:
            self.logger.warning(f"Failed to build memory context", error=str(e))

        return context

    async def _fallback_response(
        self, user_message: str, context: Dict[str, Any]
    ) -> str:
        """Fallback response when RAG is not available."""
        messages = [
            {"role": "system", "content": self._build_system_prompt(context)},
            {"role": "user", "content": user_message},
        ]

        return await self.model_router.generate_response(
            messages=messages, context=context, stream=False
        )

    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        """Build system prompt with memory context."""
        prompt_parts = [
            "You are FRIDAY, a highly intelligent personal AI assistant with memory.",
            "You remember past conversations and user preferences.",
            "Provide helpful, accurate, and personalized responses.",
        ]

        # Add memory context
        memory_context = context.get("memory_context", [])
        if memory_context:
            prompt_parts.append("\nRELEVANT MEMORIES:")
            for mem in memory_context:
                prompt_parts.append(f"- {mem['content']}")

        prompt_parts.append("\nUse this context to provide personalized responses.")
        return "\n".join(prompt_parts)

    async def _prepare_llm_messages(
        self, user_message: str, conversation_id: str, context: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Prepare messages for LLM with context."""
        system_prompt = self._build_system_prompt(context)
        messages = [{"role": "system", "content": system_prompt}]

        # Add recent conversation history
        conversation_context = context.get("conversation_context", [])
        for msg in conversation_context[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Add current user message
        messages.append({"role": "user", "content": user_message})
        return messages

    async def _create_new_conversation(self, first_message: str) -> Conversation:
        """Create a new conversation."""
        title = first_message[:50] + "..." if len(first_message) > 50 else first_message
        title = title.strip() or self.default_conversation_title

        conversation = Conversation(
            user_id=self.user_id,
            title=title,
            description=f"Started with: {first_message[:100]}...",
        )

        await self.chat_store.create_conversation(conversation)
        return conversation

    # Keep existing methods for compatibility
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        return await self.chat_store.get_conversation(conversation_id)

    async def get_conversation_messages(
        self, conversation_id: str, limit: Optional[int] = None
    ) -> List[ChatMessage]:
        return await self.chat_store.get_conversation_messages(conversation_id, limit)

    async def get_user_conversations(
        self, limit: Optional[int] = None
    ) -> List[Conversation]:
        return await self.chat_store.get_user_conversations(limit)

    async def search_conversations(
        self, query: str, limit: int = 10
    ) -> List[ChatMessage]:
        return await self.chat_store.search_messages(query, limit=limit)

    async def shutdown(self) -> None:
        """Shutdown chat manager."""
        self.logger.info("Shutting down enhanced chat manager")
        await self.chat_store.shutdown()


# Step 2: Create ChatManager alias for compatibility
ChatManager = EnhancedChatManager

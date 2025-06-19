from typing import Dict, Any, List, Optional, AsyncGenerator, Union
import time


from src.core.chat.models import (
    Conversation,
    ChatMessage,
    MessageStatus,
    MessageType,
    ResponseType,
    ChatMessageResponse,
    StreamChunk,
)
from src.core.memory.models import MemoryType, MemoryImportance
from src.core.chat.chat_store import ChatStore
from src.core.llm.model_router import ModelRouter
from src.core.utils.datetime_utils import utc_now, safe_parse_datetime
from src.core.telemetry.logger import StructuredLogger, log_execution_time
from src.core.config.settings import Settings


class ChatManager:
    """High-level chat management and orchestration."""

    def __init__(self, user_id: str, settings: "Settings"):
        self.user_id = user_id
        self.settings = settings
        self.logger = StructuredLogger("chat.manager")

        # Initialize components
        self.chat_store = ChatStore(user_id, settings)
        self.model_router = ModelRouter(settings)

        # Import brain orchestrator and other managers
        self.brain_orchestrator = None  # Will be injected
        self.memory_manager = None  # Will be injected
        self.document_manager = None  # Will be injected

        # Configuration
        self.max_context_length = 4000
        self.default_conversation_title = "New Conversation"

    async def initialize(self) -> None:
        """Initialize chat manager."""
        self.logger.info(f"Initializing chat manager for user {self.user_id}")

        # Initialize storage and LLM
        await self.chat_store.initialize()
        await self.model_router.initialize()

        self.logger.info("Chat manager initialized successfully")

    def set_dependencies(
        self, brain_orchestrator=None, memory_manager=None, document_manager=None
    ):
        """Inject dependencies from main application."""
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
        """Send a message and get response."""
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

            # Create user message
            user_message = ChatMessage(
                conversation_id=conversation_id,
                user_id=self.user_id,
                message_type=MessageType.USER,
                content=message,
                status=MessageStatus.COMPLETED,
            )

            # Store user message
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
                return self._stream_response(
                    assistant_message, message, conversation_id, start_time
                )
            else:
                return await self._complete_response(
                    assistant_message, message, conversation_id, start_time
                )

        except Exception as e:
            self.logger.error(f"Failed to send message", error=str(e))
            raise

    async def _create_new_conversation(self, first_message: str) -> Conversation:
        """Create a new conversation with auto-generated title."""
        # Generate title from first message (first 50 chars)
        title = first_message[:50] + "..." if len(first_message) > 50 else first_message
        title = title.strip() or self.default_conversation_title

        conversation = Conversation(
            user_id=self.user_id,
            title=title,
            description=f"Started with: {first_message[:100]}...",
        )

        await self.chat_store.create_conversation(conversation)
        return conversation

    async def _complete_response(
        self,
        assistant_message: ChatMessage,
        user_message: str,
        conversation_id: str,
        start_time: float,
    ) -> ChatMessageResponse:
        """Generate complete response."""
        try:
            # Get conversation context
            context = await self._build_context(user_message, conversation_id)

            # Prepare messages for LLM
            llm_messages = await self._prepare_llm_messages(
                user_message, conversation_id, context
            )

            # Generate response
            response_text = await self.model_router.generate_response(
                messages=llm_messages, context=context, stream=False
            )

            # Update assistant message
            assistant_message.content = response_text
            assistant_message.status = MessageStatus.COMPLETED
            assistant_message.processing_time = time.time() - start_time
            assistant_message.response_type = ResponseType.TEXT
            assistant_message.sources = context.get("sources", [])
            assistant_message.token_count = await self.model_router.count_tokens(
                response_text
            )
            assistant_message.updated_at = utc_now()

            # Store assistant message
            await self.chat_store.store_message(assistant_message)

            # Store relevant information in memory
            await self._store_conversation_memory(user_message, response_text, context)

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

    async def _stream_response(
        self,
        assistant_message: ChatMessage,
        user_message: str,
        conversation_id: str,
        start_time: float,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Generate streaming response."""
        try:
            assistant_message.status = MessageStatus.STREAMING
            await self.chat_store.store_message(assistant_message)

            # Get conversation context
            context = await self._build_context(user_message, conversation_id)

            # Prepare messages for LLM
            llm_messages = await self._prepare_llm_messages(
                user_message, conversation_id, context
            )

            # Generate streaming response
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
                    "sources": context.get("sources", []),
                    "token_count": await self.model_router.count_tokens(full_response),
                },
            )

            # Update and store final message
            assistant_message.content = full_response
            assistant_message.status = MessageStatus.COMPLETED
            assistant_message.processing_time = processing_time
            assistant_message.response_type = ResponseType.TEXT
            assistant_message.sources = context.get("sources", [])
            assistant_message.token_count = await self.model_router.count_tokens(
                full_response
            )
            assistant_message.updated_at = utc_now()

            await self.chat_store.store_message(assistant_message)

            # Store in memory
            await self._store_conversation_memory(user_message, full_response, context)

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

    async def _build_context(
        self, user_message: str, conversation_id: str
    ) -> Dict[str, Any]:
        """Build context for LLM including memory and documents."""
        context = {
            "sources": [],
            "memory_context": [],
            "document_context": [],
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

            # Search relevant memories
            if self.memory_manager:
                relevant_memories = await self.memory_manager.search_memories(
                    query=user_message, limit=5
                )
                context["memory_context"] = [
                    {
                        "type": "memory",
                        "content": mem.content,
                        "importance": mem.importance.value,
                        "tags": mem.tags,
                    }
                    for mem in relevant_memories
                ]
                context["sources"].extend(
                    [
                        {
                            "type": "memory",
                            "id": mem.id,
                            "content": mem.content[:100] + "...",
                            "importance": mem.importance.value,
                        }
                        for mem in relevant_memories
                    ]
                )

            # Search relevant documents
            if self.document_manager:
                relevant_docs = await self.document_manager.search_documents(
                    query=user_message, limit=3
                )

                relevant_chunks = await self.document_manager.search_content(
                    query=user_message, limit=5
                )

                context["document_context"] = [
                    {
                        "type": "document",
                        "title": doc.title,
                        "content": " ".join(
                            [chunk.content for chunk in doc.chunks[:2]]
                        ),
                        "category": doc.category.value,
                    }
                    for doc in relevant_docs
                ]

                context["document_context"].extend(
                    [
                        {
                            "type": "chunk",
                            "content": chunk.content,
                            "source": chunk.document_id,
                        }
                        for chunk in relevant_chunks
                    ]
                )

                context["sources"].extend(
                    [
                        {
                            "type": "document",
                            "id": doc.id,
                            "title": doc.title,
                            "category": doc.category.value,
                        }
                        for doc in relevant_docs
                    ]
                )

        except Exception as e:
            self.logger.warning(f"Failed to build complete context", error=str(e))

        return context

    async def _prepare_llm_messages(
        self, user_message: str, conversation_id: str, context: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Prepare messages for LLM with context."""

        # System prompt
        system_prompt = self._build_system_prompt(context)

        messages = [{"role": "system", "content": system_prompt}]

        # Add recent conversation history
        conversation_context = context.get("conversation_context", [])
        for msg in conversation_context[-6:]:  # Last 6 messages for context
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        return messages

    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        """Build system prompt with context."""
        prompt_parts = [
            "You are FRIDAY, a highly intelligent personal AI assistant.",
            "You have access to the user's personal documents, memories, and conversation history.",
            "Provide helpful, accurate, and personalized responses based on the available context.",
            "Always cite your sources when referencing specific documents or memories.",
            "",
        ]

        # Add memory context
        memory_context = context.get("memory_context", [])
        if memory_context:
            prompt_parts.append("RELEVANT MEMORIES:")
            for mem in memory_context:
                prompt_parts.append(
                    f"- {mem['content']} (Tags: {', '.join(mem['tags'])})"
                )
            prompt_parts.append("")

        # Add document context
        document_context = context.get("document_context", [])
        if document_context:
            prompt_parts.append("RELEVANT DOCUMENTS:")
            for doc in document_context:
                if doc["type"] == "document":
                    prompt_parts.append(f"Document: {doc['title']} ({doc['category']})")
                    prompt_parts.append(f"Content: {doc['content'][:500]}...")
                else:
                    prompt_parts.append(f"Content chunk: {doc['content']}")
            prompt_parts.append("")

        prompt_parts.extend(
            [
                "Use this context to provide informed and personalized responses.",
                "If you reference specific information, mention which document or memory it came from.",
                "Be conversational but professional.",
            ]
        )

        return "\n".join(prompt_parts)

    async def _store_conversation_memory(
        self, user_message: str, assistant_response: str, context: Dict[str, Any]
    ):
        """Store important parts of conversation in memory."""
        if not self.memory_manager:
            return

        try:
            # Store user intent/query if it seems important
            if len(user_message) > 20 and any(
                word in user_message.lower()
                for word in [
                    "remember",
                    "important",
                    "prefer",
                    "like",
                    "don't like",
                    "always",
                    "never",
                ]
            ):
                await self.memory_manager.create_memory(
                    content=f"User said: {user_message}",
                    memory_type=MemoryType.EPISODIC,
                    importance=MemoryImportance.MEDIUM,
                    tags=["conversation", "user_statement"],
                )

            # Store assistant insights if they seem valuable
            if len(assistant_response) > 50 and not assistant_response.startswith(
                "I don't"
            ):
                await self.memory_manager.create_memory(
                    content=f"Provided information: {assistant_response[:200]}...",
                    memory_type=MemoryType.EPISODIC,
                    importance=MemoryImportance.LOW,
                    tags=["conversation", "assistant_response"],
                    expires_in_hours=168,  # 1 week
                )

        except Exception as e:
            self.logger.warning(f"Failed to store conversation memory", error=str(e))

    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID."""
        return await self.chat_store.get_conversation(conversation_id)

    async def get_conversation_messages(
        self, conversation_id: str, limit: Optional[int] = None
    ) -> List[ChatMessage]:
        """Get messages for a conversation."""
        return await self.chat_store.get_conversation_messages(conversation_id, limit)

    async def get_user_conversations(
        self, limit: Optional[int] = None
    ) -> List[Conversation]:
        """Get user's conversations."""
        return await self.chat_store.get_user_conversations(limit)

    async def search_conversations(
        self, query: str, limit: int = 10
    ) -> List[ChatMessage]:
        """Search across all conversations."""
        return await self.chat_store.search_messages(query, limit=limit)

    async def shutdown(self) -> None:
        """Shutdown chat manager."""
        self.logger.info("Shutting down chat manager")
        await self.chat_store.shutdown()

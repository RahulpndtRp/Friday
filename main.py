import asyncio
import uvicorn
from pathlib import Path
import sys
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from core.config.settings import Settings
from core.brain.orchestrator import BrainOrchestrator
from core.brain.memory_brain import MemoryBrain
from core.brain.document_brain import DocumentBrain
from core.memory.memory_manager import MemoryManager
from core.documents.document_manager import DocumentManager
from core.chat.chat_manager import ChatManager
from core.telemetry.logger import StructuredLogger
from interfaces.main_api import app, register_chat_manager
from core.brain.orchestrator import EnhancedBrainOrchestrator
from core.brain.hybrid_memory_brain import HybridMemoryBrain


class EnhancedFridayAssistant:
    """
    Enhanced FRIDAY Assistant with hybrid memory powered by memrp
    """

    def __init__(self):
        self.settings = Settings()
        self.logger = StructuredLogger("friday.main")

        # Core components - using enhanced orchestrator
        self.orchestrator = EnhancedBrainOrchestrator(self.settings)
        self.document_managers = {}
        self.chat_managers = {}

        # Default user for single-user setup
        self.default_user_id = "friday_user_001"

        self._is_running = False

    async def initialize(self):
        """Initialize the enhanced FRIDAY assistant."""
        self.logger.info(
            "🚀 Initializing Enhanced FRIDAY Personal Assistant",
            environment=self.settings.environment,
            debug=self.settings.debug,
            memory_system="memrp_hybrid",
        )

        try:
            # Step 1: Initialize core brains
            await self._initialize_core_brains()

            # Step 2: Initialize user-specific managers
            await self._initialize_user_managers(self.default_user_id)

            # Step 3: Register with API
            register_chat_manager(
                self.default_user_id, self.chat_managers[self.default_user_id]
            )

            self._is_running = True
            self.logger.info("✅ Enhanced FRIDAY Assistant initialized successfully")

        except Exception as e:
            self.logger.error(
                f"❌ Failed to initialize Enhanced FRIDAY Assistant", error=str(e)
            )
            raise

    async def _initialize_core_brains(self):
        """Initialize and register all brain components."""
        self.logger.info("Initializing core brain components")

        # Step 1: Create Hybrid Memory Brain (replaces old memory system)
        hybrid_memory_brain = HybridMemoryBrain(self.settings)
        self.orchestrator.register_brain(hybrid_memory_brain)
        self.logger.info("✅ Hybrid Memory Brain registered (memrp-powered)")

        # Step 2: Create Document Brain (enhanced with vector search)
        document_brain = DocumentBrain(self.settings)
        self.orchestrator.register_brain(document_brain)
        self.logger.info("✅ Document Brain registered")

        # Step 3: Initialize all brains
        await self.orchestrator.initialize_all_brains()
        self.logger.info("✅ All brains initialized successfully")

    async def _initialize_user_managers(self, user_id: str):
        """Initialize managers for a specific user."""
        self.logger.info(f"Initializing managers for user", user_id=user_id)

        # Document manager (keep existing for file processing)
        document_manager = DocumentManager(user_id, self.settings)
        await document_manager.initialize()
        self.document_managers[user_id] = document_manager

        # Chat manager with enhanced brain integration
        chat_manager = EnhancedChatManager(user_id, self.settings)
        await chat_manager.initialize()

        # Set dependencies - now using orchestrator instead of individual managers
        chat_manager.set_dependencies(
            brain_orchestrator=self.orchestrator,
            document_manager=document_manager,
        )

        self.chat_managers[user_id] = chat_manager

        self.logger.info(f"✅ Enhanced user managers initialized", user_id=user_id)

    async def start_api_server(self):
        """Start the FastAPI server."""
        if not self._is_running:
            await self.initialize()

        self.logger.info(
            f"🌐 Starting Enhanced API server",
            host=self.settings.api_host,
            port=self.settings.api_port,
        )

        config = uvicorn.Config(
            app,
            host=self.settings.api_host,
            port=self.settings.api_port,
            reload=self.settings.api_reload,
            log_level="info",
        )

        server = uvicorn.Server(config)
        await server.serve()

    async def start(self):
        """Start the enhanced FRIDAY assistant with API server."""
        try:
            await self.start_api_server()
        except KeyboardInterrupt:
            await self.shutdown()

    async def shutdown(self):
        """Shutdown the FRIDAY assistant gracefully."""
        self.logger.info("🔄 Shutting down Enhanced FRIDAY Assistant")

        self._is_running = False

        # Shutdown all chat managers
        for chat_manager in self.chat_managers.values():
            await chat_manager.shutdown()

        # Shutdown all document managers
        for doc_manager in self.document_managers.values():
            await doc_manager.shutdown()

        # Shutdown orchestrator (which handles all brains)
        await self.orchestrator.shutdown_all()

        self.logger.info("✅ Enhanced FRIDAY Assistant shutdown complete")


class EnhancedChatManager(ChatManager):
    """
    Enhanced Chat Manager that leverages the new brain orchestrator
    instead of individual memory/document managers
    """

    def __init__(self, user_id: str, settings: Settings):
        super().__init__(user_id, settings)
        self.brain_orchestrator = None

    def set_dependencies(self, brain_orchestrator=None, document_manager=None):
        """Set dependencies - now primarily using brain orchestrator."""
        self.brain_orchestrator = brain_orchestrator
        self.document_manager = document_manager  # Keep for file processing

    async def _build_context(
        self, user_message: str, conversation_id: str
    ) -> Dict[str, Any]:
        """
        Enhanced context building using brain orchestrator
        """
        context = {
            "sources": [],
            "memory_context": [],
            "document_context": [],
            "conversation_context": [],
        }

        try:
            # Get conversation history (keep existing logic)
            recent_messages = await self.chat_store.get_conversation_messages(
                conversation_id, limit=10
            )
            context["conversation_context"] = [
                {"role": msg.message_type.value, "content": msg.content}
                for msg in recent_messages
            ]

            # Use brain orchestrator for enhanced context
            if self.brain_orchestrator:
                brain_request = BrainRequest(
                    request_id=f"context_{conversation_id}",
                    user_id=self.user_id,
                    message=user_message,
                    context={
                        "operation": "get_context",
                        "context_size": 10,
                        "conversation_history": context["conversation_context"],
                    },
                    metadata={"source": "chat_manager"},
                )

                # Process through orchestrator
                brain_responses = await self.brain_orchestrator.process_request(
                    brain_request
                )

                # Extract memory context from responses
                if "memory" in brain_responses:
                    memory_response = brain_responses["memory"]
                    if memory_response.success and memory_response.response:
                        memories = memory_response.response.get("context_memories", [])
                        context["memory_context"] = [
                            {
                                "type": "memory",
                                "content": mem.get("memory", ""),
                                "score": mem.get("score", 0.0),
                                "id": mem.get("id", ""),
                            }
                            for mem in memories
                        ]

                        # Add to sources
                        context["sources"].extend(
                            [
                                {
                                    "type": "memory",
                                    "id": mem.get("id", ""),
                                    "content": mem.get("memory", "")[:100] + "...",
                                    "score": mem.get("score", 0.0),
                                }
                                for mem in memories[:5]
                            ]
                        )

                # Extract document context if available
                if "knowledge" in brain_responses:
                    doc_response = brain_responses["knowledge"]
                    if doc_response.success and doc_response.response:
                        documents = doc_response.response.get("documents", [])
                        context["document_context"] = [
                            {
                                "type": "document",
                                "title": doc.get("title", ""),
                                "content": doc.get("content", ""),
                                "category": doc.get("category", ""),
                            }
                            for doc in documents
                        ]

            # Fallback to document manager if needed
            if not context["document_context"] and self.document_manager:
                relevant_docs = await self.document_manager.search_documents(
                    query=user_message, limit=3
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

        except Exception as e:
            self.logger.warning(f"Failed to build enhanced context", error=str(e))

        return context

    async def _store_conversation_memory(
        self, user_message: str, assistant_response: str, context: Dict[str, Any]
    ):
        """
        Enhanced memory storage using brain orchestrator
        """
        if not self.brain_orchestrator:
            return

        try:
            # Store user message
            user_request = BrainRequest(
                request_id=f"store_user_{int(time.time())}",
                user_id=self.user_id,
                message=user_message,
                context={"operation": "store", "infer": True, "source": "user_message"},
                metadata={"conversation_context": context},
            )

            await self.brain_orchestrator.process_request(user_request)

            # Store assistant response if it contains valuable information
            if len(assistant_response) > 50 and not assistant_response.startswith(
                "I don't"
            ):
                assistant_request = BrainRequest(
                    request_id=f"store_assistant_{int(time.time())}",
                    user_id=self.user_id,
                    message=f"Assistant provided: {assistant_response[:200]}...",
                    context={
                        "operation": "store",
                        "infer": True,
                        "source": "assistant_response",
                    },
                    metadata={"conversation_context": context},
                )

                await self.brain_orchestrator.process_request(assistant_request)

        except Exception as e:
            self.logger.warning(f"Failed to store conversation memory", error=str(e))


# Import required components
from src.core.brain.base_brain import BrainRequest
import time


# Entry point
async def main():
    """Main entry point with enhanced FRIDAY."""
    friday = EnhancedFridayAssistant()
    await friday.start()


if __name__ == "__main__":
    print("🤖 Starting Enhanced FRIDAY Personal Assistant...")
    print("📡 API will be available at: http://localhost:8000")
    print("🧠 Memory System: memrp hybrid (semantic + episodic)")
    print("🧪 Test interface at: http://localhost:8000/test")
    print("📚 API docs at: http://localhost:8000/docs")
    print("🔌 WebSocket at: ws://localhost:8000/ws/chat/friday_user_001")
    print("")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Enhanced FRIDAY Assistant stopped by user")
    except Exception as e:
        print(f"\n❌ Enhanced FRIDAY Assistant failed to start: {e}")

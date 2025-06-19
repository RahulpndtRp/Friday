import asyncio
import uvicorn
from pathlib import Path
import sys

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


class FridayAssistant:
    """Main FRIDAY Personal Assistant application."""

    def __init__(self):
        self.settings = Settings()
        self.logger = StructuredLogger("friday.main")

        # Core components
        self.orchestrator = BrainOrchestrator(self.settings)
        self.memory_managers = {}
        self.document_managers = {}
        self.chat_managers = {}

        # Default user for single-user setup
        self.default_user_id = "friday_user_001"

        self._is_running = False
        self._api_server = None

    async def initialize(self):
        """Initialize the FRIDAY assistant."""
        self.logger.info(
            "🚀 Initializing FRIDAY Personal Assistant",
            environment=self.settings.environment,
            debug=self.settings.debug,
        )

        try:
            # Initialize brain orchestrator
            await self.orchestrator.initialize_all_brains()

            # Register core brains
            memory_brain = MemoryBrain(self.settings)
            document_brain = DocumentBrain(self.settings)

            self.orchestrator.register_brain(memory_brain)
            self.orchestrator.register_brain(document_brain)

            # Initialize user-specific managers
            await self._initialize_user_managers(self.default_user_id)

            # Register with API
            register_chat_manager(
                self.default_user_id, self.chat_managers[self.default_user_id]
            )

            self._is_running = True
            self.logger.info("✅ FRIDAY Assistant initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize FRIDAY Assistant", error=str(e))
            raise

    async def _initialize_user_managers(self, user_id: str):
        """Initialize managers for a specific user."""
        self.logger.info(f"Initializing managers for user", user_id=user_id)

        # Memory manager
        memory_manager = MemoryManager(user_id, self.settings)
        await memory_manager.initialize()
        self.memory_managers[user_id] = memory_manager

        # Document manager
        document_manager = DocumentManager(user_id, self.settings)
        await document_manager.initialize()
        self.document_managers[user_id] = document_manager

        # Chat manager
        chat_manager = ChatManager(user_id, self.settings)
        await chat_manager.initialize()

        # Set dependencies
        chat_manager.set_dependencies(
            brain_orchestrator=self.orchestrator,
            memory_manager=memory_manager,
            document_manager=document_manager,
        )

        self.chat_managers[user_id] = chat_manager

        self.logger.info(f"✅ User managers initialized", user_id=user_id)

    async def start_api_server(self):
        """Start the FastAPI server."""
        if not self._is_running:
            await self.initialize()

        self.logger.info(
            f"🌐 Starting API server",
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
        """Start the FRIDAY assistant with API server."""
        try:
            await self.start_api_server()
        except KeyboardInterrupt:
            await self.shutdown()

    async def shutdown(self):
        """Shutdown the FRIDAY assistant gracefully."""
        self.logger.info("🔄 Shutting down FRIDAY Assistant")

        self._is_running = False

        # Shutdown all chat managers
        for chat_manager in self.chat_managers.values():
            await chat_manager.shutdown()

        # Shutdown all document managers
        for doc_manager in self.document_managers.values():
            await doc_manager.shutdown()

        # Shutdown all memory managers
        for mem_manager in self.memory_managers.values():
            await mem_manager.shutdown()

        # Shutdown orchestrator
        await self.orchestrator.shutdown_all()

        self.logger.info("✅ FRIDAY Assistant shutdown complete")


# Entry point
async def main():
    """Main entry point."""
    friday = FridayAssistant()
    await friday.start()


if __name__ == "__main__":
    print("🤖 Starting FRIDAY Personal Assistant...")
    print("📡 API will be available at: http://localhost:8000")
    print("🧪 Test interface at: http://localhost:8000/test")
    print("📚 API docs at: http://localhost:8000/docs")
    print("🔌 WebSocket at: ws://localhost:8000/ws/chat/friday_user_001")
    print("")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 FRIDAY Assistant stopped by user")
    except Exception as e:
        print(f"\n❌ FRIDAY Assistant failed to start: {e}")

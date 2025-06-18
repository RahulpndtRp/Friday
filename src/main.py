import asyncio
from src.core.config.settings import Settings
from src.core.brain.orchestrator import BrainOrchestrator
from src.core.telemetry.logger import StructuredLogger

class FridayAssistant:
    """Main application class for FRIDAY Personal Assistant."""
    
    def __init__(self):
        self.settings = Settings()
        self.logger = StructuredLogger("friday.main")
        self.orchestrator = BrainOrchestrator(self.settings)
        self._is_running = False
    
    async def initialize(self):
        """Initialize the FRIDAY assistant."""
        self.logger.info(
            "Initializing FRIDAY Personal Assistant",
            environment=self.settings.environment,
            debug=self.settings.debug
        )
        
        # Initialize orchestrator and brains
        await self.orchestrator.initialize_all_brains()
        
        self._is_running = True
        self.logger.info("FRIDAY Assistant initialized successfully")
    
    async def start(self):
        """Start the FRIDAY assistant."""
        if not self._is_running:
            await self.initialize()
        
        self.logger.info("Starting FRIDAY Assistant")
        # Here we'll add API server startup in next phase
        
        try:
            # Keep the application running
            while self._is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the FRIDAY assistant gracefully."""
        self.logger.info("Shutting down FRIDAY Assistant")
        self._is_running = False
        await self.orchestrator.shutdown_all()


if __name__ == "__main__":
    app = FridayAssistant()
    asyncio.run(app.start())
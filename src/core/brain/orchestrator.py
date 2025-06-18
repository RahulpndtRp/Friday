from typing import List, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from src.core.telemetry.logger import StructuredLogger, log_execution_time
from src.core.config.settings import Settings
from src.core.brain.base_brain import BrainType, BaseBrain, BrainRequest, BrainResponse


class BrainOrchestrator:
    """Orchestrates communication between different brain components."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = StructuredLogger("brain.orchestrator")
        self.brains: Dict[BrainType, BaseBrain] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._request_queue = asyncio.Queue()
        self._is_running = False

    def register_brain(self, brain: BaseBrain) -> None:
        """Register a brain component with the orchestrator."""
        self.brains[brain.brain_type] = brain
        self.logger.info(
            f"Brain registered",
            brain_type=brain.brain_type.value,
            total_brains=len(self.brains),
        )

    async def initialize_all_brains(self) -> None:
        """Initialize all registered brains."""
        self.logger.info("Initializing all brains")

        initialization_tasks = []
        for brain_type, brain in self.brains.items():
            task = asyncio.create_task(self._safe_initialize_brain(brain_type, brain))
            initialization_tasks.append(task)

        results = await asyncio.gather(*initialization_tasks, return_exceptions=True)

        successful_inits = sum(1 for result in results if result is True)
        self.logger.info(
            f"Brain initialization completed",
            successful=successful_inits,
            total=len(self.brains),
            failed=len(self.brains) - successful_inits,
        )

    async def _safe_initialize_brain(
        self, brain_type: BrainType, brain: BaseBrain
    ) -> bool:
        """Safely initialize a single brain with error handling."""
        try:
            await brain.initialize()
            self.logger.info(
                f"Brain initialized successfully", brain_type=brain_type.value
            )
            return True
        except Exception as e:
            self.logger.error(
                f"Brain initialization failed",
                brain_type=brain_type.value,
                error=str(e),
            )
            return False

    @log_execution_time
    async def process_request(self, request: BrainRequest) -> Dict[str, BrainResponse]:
        """Process a request through relevant brains."""
        self.logger.info(
            f"Processing request",
            request_id=request.request_id,
            user_id=request.user_id,
            message_length=len(request.message),
        )

        # Determine which brains should process this request
        relevant_brains = self._determine_relevant_brains(request)

        # Process request through relevant brains concurrently
        tasks = []
        for brain_type in relevant_brains:
            if brain_type in self.brains:
                brain = self.brains[brain_type]
                task = asyncio.create_task(self._safe_process_brain(brain, request))
                tasks.append((brain_type, task))

        # Collect responses
        responses = {}
        for brain_type, task in tasks:
            try:
                response = await task
                responses[brain_type.value] = response
            except Exception as e:
                self.logger.error(
                    f"Brain processing failed",
                    brain_type=brain_type.value,
                    error=str(e),
                )

        return responses

    async def _safe_process_brain(
        self, brain: BaseBrain, request: BrainRequest
    ) -> BrainResponse:
        """Safely process request through a brain with error handling."""
        start_time = time.time()

        try:
            response = await brain.process(request)
            return response
        except Exception as e:
            processing_time = time.time() - start_time
            return BrainResponse(
                request_id=request.request_id,
                brain_type=brain.brain_type,
                response={},
                confidence=0.0,
                processing_time=processing_time,
                metadata={"error_handled": True},
                success=False,
                error=str(e),
            )

    def _determine_relevant_brains(self, request: BrainRequest) -> List[BrainType]:
        """Determine which brains should process the request."""
        # For now, let's route to all available brains
        # In future, we can add intelligent routing logic
        return list(self.brains.keys())

    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all brains."""
        health_status = {}

        for brain_type, brain in self.brains.items():
            try:
                status = await brain.health_check()
                health_status[brain_type.value] = {
                    "status": "healthy",
                    "details": status,
                }
            except Exception as e:
                health_status[brain_type.value] = {
                    "status": "unhealthy",
                    "error": str(e),
                }

        return health_status

    async def shutdown_all(self) -> None:
        """Shutdown all brains gracefully."""
        self.logger.info("Shutting down all brains")

        shutdown_tasks = []
        for brain in self.brains.values():
            task = asyncio.create_task(brain.shutdown())
            shutdown_tasks.append(task)

        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        self.executor.shutdown(wait=True)

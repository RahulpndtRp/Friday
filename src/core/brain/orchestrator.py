"""
Enhanced Brain Orchestrator with intelligent routing
Integrates with the new Hybrid Memory Brain
"""

from typing import List, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import json

from src.core.telemetry.logger import StructuredLogger, log_execution_time
from src.core.config.settings import Settings
from src.core.brain.base_brain import BrainType, BaseBrain, BrainRequest, BrainResponse
from src.core.llm.model_router import ModelRouter


class EnhancedBrainOrchestrator:
    """
    Enhanced orchestrator that intelligently routes requests to appropriate brains
    with special integration for the new Hybrid Memory Brain
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = StructuredLogger("brain.orchestrator")
        self.brains: Dict[BrainType, BaseBrain] = {}

        # Add LLM for intelligent routing decisions
        self.model_router = ModelRouter(settings)

        self._request_queue = asyncio.Queue()
        self._is_running = False

        # Intent classification cache
        self._intent_cache: Dict[str, Dict] = {}

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

        # Initialize model router first
        await self.model_router.initialize()

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
        """Process a request through intelligently selected brains."""
        self.logger.info(
            f"Processing request",
            request_id=request.request_id,
            user_id=request.user_id,
            message_length=len(request.message),
        )

        # Step 1: Classify intent and determine relevant brains
        relevant_brains = await self._determine_relevant_brains(request)

        # Step 2: Check if memory operation is needed
        memory_first = await self._needs_memory_first(request)

        # Step 3: Process through relevant brains
        if memory_first:
            responses = await self._process_with_memory_priority(
                request, relevant_brains
            )
        else:
            responses = await self._process_concurrent(request, relevant_brains)

        return responses

    async def _determine_relevant_brains(
        self, request: BrainRequest
    ) -> List[BrainType]:
        """Use LLM to intelligently determine which brains should process the request."""

        # Check cache first
        cache_key = self._get_intent_cache_key(request.message)
        if cache_key in self._intent_cache:
            cached_intent = self._intent_cache[cache_key]
            return self._brains_for_intent(cached_intent)

        # Use LLM for intent classification
        intent_prompt = self._build_intent_classification_prompt(request.message)

        try:
            llm_response = await self.model_router.generate_response(
                messages=[{"role": "user", "content": intent_prompt}],
                context={"routing_preference": "fast"},
                stream=False,
            )

            # Parse LLM response
            intent_data = self._parse_intent_response(llm_response)

            # Cache the result
            self._intent_cache[cache_key] = intent_data

            return self._brains_for_intent(intent_data)

        except Exception as e:
            self.logger.warning(
                f"Intent classification failed, using fallback", error=str(e)
            )
            return self._fallback_brain_selection(request)

    def _build_intent_classification_prompt(self, message: str) -> str:
        """Build prompt for LLM-based intent classification."""
        return f"""
Analyze this user message and classify the intent. Return JSON with the following fields:

1. "primary_intent": main goal (remember, recall, search, analyze, action, chat)
2. "requires_memory": true if needs access to past conversations/facts
3. "requires_knowledge": true if needs document/external knowledge
4. "requires_reasoning": true if needs complex analysis
5. "requires_action": true if needs to perform actions (API calls, etc.)
6. "temporal_reference": true if references past events ("last week", "yesterday")
7. "confidence": 0.0-1.0 confidence in classification

User message: "{message}"

Respond with only valid JSON.
"""

    def _parse_intent_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM intent classification response."""
        try:
            # Clean the response and parse JSON
            clean_response = llm_response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response.split("```json")[1].split("```")[0]
            elif clean_response.startswith("```"):
                clean_response = clean_response.split("```")[1].split("```")[0]

            return json.loads(clean_response)
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse intent JSON: {llm_response}")
            return self._default_intent()

    def _default_intent(self) -> Dict[str, Any]:
        """Default intent when classification fails."""
        return {
            "primary_intent": "chat",
            "requires_memory": True,
            "requires_knowledge": False,
            "requires_reasoning": False,
            "requires_action": False,
            "temporal_reference": False,
            "confidence": 0.5,
        }

    def _brains_for_intent(self, intent_data: Dict[str, Any]) -> List[BrainType]:
        """Map intent classification to relevant brain types."""
        relevant_brains = []

        # Memory brain is almost always relevant for context
        if intent_data.get("requires_memory", False) or intent_data.get(
            "temporal_reference", False
        ):
            relevant_brains.append(BrainType.MEMORY)

        # Knowledge brain for document/fact retrieval
        if intent_data.get("requires_knowledge", False):
            relevant_brains.append(BrainType.KNOWLEDGE)

        # Reasoning brain for analysis
        if intent_data.get("requires_reasoning", False):
            relevant_brains.append(BrainType.REASONING)

        # Action brain for API calls, etc.
        if intent_data.get("requires_action", False):
            relevant_brains.append(
                BrainType.ORCHESTRATOR
            )  # Using orchestrator as action brain

        # Ensure we always have at least memory for conversation context
        if not relevant_brains:
            relevant_brains.append(BrainType.MEMORY)

        # Filter to only available brains
        available_brains = [bt for bt in relevant_brains if bt in self.brains]

        return available_brains if available_brains else [BrainType.MEMORY]

    def _fallback_brain_selection(self, request: BrainRequest) -> List[BrainType]:
        """Fallback brain selection when LLM classification fails."""
        # Simple keyword-based fallback
        message_lower = request.message.lower()

        brains = []

        # Memory indicators
        if any(
            word in message_lower
            for word in [
                "remember",
                "recall",
                "last",
                "previous",
                "before",
                "yesterday",
                "ago",
            ]
        ):
            brains.append(BrainType.MEMORY)

        # Knowledge indicators
        if any(
            word in message_lower
            for word in ["document", "file", "search", "find", "what is", "explain"]
        ):
            brains.append(BrainType.KNOWLEDGE)

        # Default to memory for conversation context
        if not brains:
            brains.append(BrainType.MEMORY)

        return [bt for bt in brains if bt in self.brains]

    async def _needs_memory_first(self, request: BrainRequest) -> bool:
        """Determine if memory should be processed first to provide context."""
        message_lower = request.message.lower()

        # Memory-first indicators
        memory_first_keywords = [
            "remember",
            "recall",
            "last time",
            "previously",
            "before",
            "what did",
            "when did",
            "who said",
            "mentioned",
        ]

        return any(keyword in message_lower for keyword in memory_first_keywords)

    async def _process_with_memory_priority(
        self, request: BrainRequest, brain_types: List[BrainType]
    ) -> Dict[str, BrainResponse]:
        """Process memory first, then use results to enhance other brain requests."""
        responses = {}

        # Step 1: Process memory first if available
        if BrainType.MEMORY in brain_types and BrainType.MEMORY in self.brains:
            memory_brain = self.brains[BrainType.MEMORY]
            memory_response = await self._safe_process_brain(memory_brain, request)
            responses[BrainType.MEMORY.value] = memory_response

            # Step 2: Enhance request with memory context for other brains
            if memory_response.success and memory_response.response:
                enhanced_request = self._enhance_request_with_memory(
                    request, memory_response
                )

                # Process other brains with enhanced context
                other_brains = [bt for bt in brain_types if bt != BrainType.MEMORY]
                if other_brains:
                    other_responses = await self._process_concurrent(
                        enhanced_request, other_brains
                    )
                    responses.update(other_responses)
        else:
            # No memory brain, process normally
            responses = await self._process_concurrent(request, brain_types)

        return responses

    def _enhance_request_with_memory(
        self, request: BrainRequest, memory_response: BrainResponse
    ) -> BrainRequest:
        """Enhance request with memory context for other brains."""
        enhanced_context = request.context.copy()

        # Add memory context
        if memory_response.response:
            enhanced_context["memory_context"] = memory_response.response

        # Create enhanced request
        enhanced_request = BrainRequest(
            request_id=request.request_id,
            user_id=request.user_id,
            message=request.message,
            context=enhanced_context,
            metadata=request.metadata,
            priority=request.priority,
        )

        return enhanced_request

    async def _process_concurrent(
        self, request: BrainRequest, brain_types: List[BrainType]
    ) -> Dict[str, BrainResponse]:
        """Process request through multiple brains concurrently."""
        tasks = []
        for brain_type in brain_types:
            if brain_type in self.brains:
                brain = self.brains[brain_type]
                task = asyncio.create_task(self._safe_process_brain(brain, request))
                tasks.append((brain_type, task))

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

    def _get_intent_cache_key(self, message: str) -> str:
        """Generate cache key for intent classification."""
        # Simple hash of message for caching
        import hashlib

        return hashlib.md5(message.lower().encode()).hexdigest()[:8]

    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all brains with enhanced monitoring."""
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

        # Add orchestrator health
        health_status["orchestrator"] = {
            "status": "healthy",
            "registered_brains": len(self.brains),
            "model_router": "operational" if self.model_router else "unavailable",
            "intent_cache_size": len(self._intent_cache),
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

        # Clear caches
        self._intent_cache.clear()

        self.logger.info("Enhanced orchestrator shutdown complete")


class BrainOrchestrator:
    """Orchestrates communication between different brain components."""

    def __init__(self, settings: "Settings"):
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

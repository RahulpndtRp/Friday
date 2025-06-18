from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import asyncio

from src.core.telemetry.logger import StructuredLogger
from src.core.config.settings import Settings


class BrainType(Enum):
    REASONING = "reasoning"
    KNOWLEDGE = "knowledge"
    MEMORY = "memory"
    EMOTIONAL = "emotional"
    ORCHESTRATOR = "orchestrator"


@dataclass
class BrainRequest:
    """Standardized request format for all brains."""

    request_id: str
    user_id: str
    message: str
    context: Dict[str, Any]
    metadata: Dict[str, Any]
    priority: int = 5  # 1-10, 10 being highest


@dataclass
class BrainResponse:
    """Standardized response format for all brains."""

    request_id: str
    brain_type: BrainType
    response: Dict[str, Any]
    confidence: float  # 0.0 to 1.0
    processing_time: float
    metadata: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None


class BaseBrain(ABC):
    """Abstract base class for all AI brain components."""

    def __init__(self, brain_type: BrainType, settings: Settings):
        self.brain_type = brain_type
        self.settings = settings
        self.logger = StructuredLogger(f"brain.{brain_type.value}")
        self.is_initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the brain component."""
        pass

    @abstractmethod
    async def process(self, request: BrainRequest) -> BrainResponse:
        """Process a request and return a response."""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Return health status of the brain."""
        pass

    async def shutdown(self) -> None:
        """Cleanup and shutdown the brain."""
        self.logger.info(f"Shutting down {self.brain_type.value} brain")
        self.is_initialized = False

    def _validate_request(self, request: BrainRequest) -> bool:
        """Validate incoming request format."""
        required_fields = ["request_id", "user_id", "message"]
        return all(
            hasattr(request, field) and getattr(request, field)
            for field in required_fields
        )

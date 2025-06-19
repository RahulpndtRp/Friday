from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
import asyncio

from src.core.config.settings import Settings
from src.core.telemetry.logger import StructuredLogger, log_execution_time


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, settings: "Settings"):
        self.settings = settings
        self.logger = StructuredLogger(f"llm.{self.__class__.__name__.lower()}")
        self.is_initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the LLM provider."""
        pass

    @abstractmethod
    async def generate_response(
        self, messages: List[Dict[str, str]], stream: bool = False, **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check provider health."""
        return {
            "status": "healthy" if self.is_initialized else "not_initialized",
            "model_info": self.get_model_info(),
        }

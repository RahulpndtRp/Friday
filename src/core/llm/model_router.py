from typing import Dict, Any, List, Optional, Union, AsyncGenerator
import asyncio

from src.core.config.settings import Settings
from src.core.telemetry.logger import StructuredLogger, log_execution_time
from src.core.llm.base_llm import BaseLLMProvider
from src.core.llm.openai_provider import OpenAIProvider


class ModelRouter:
    """Routes requests to appropriate LLM providers."""

    def __init__(self, settings: "Settings"):
        self.settings = settings
        self.logger = StructuredLogger("llm.router")
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.default_provider = "openai"

        # Routing rules
        self.routing_rules = {
            "sensitive": "local",  # Use local models for sensitive data
            "complex": "openai",  # Use OpenAI for complex reasoning
            "fast": "openai",  # Use OpenAI for fast responses
            "creative": "openai",  # Use OpenAI for creative tasks
        }

    async def initialize(self) -> None:
        """Initialize all available providers."""
        self.logger.info("Initializing model router")

        # Initialize OpenAI provider
        if self.settings.openai_api_key:
            try:
                openai_provider = OpenAIProvider(self.settings)
                await openai_provider.initialize()
                self.providers["openai"] = openai_provider
                self.logger.info("OpenAI provider registered")
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize OpenAI provider", error=str(e)
                )

        if not self.providers:
            raise RuntimeError("No LLM providers available")

        self.logger.info(
            f"Model router initialized with {len(self.providers)} providers"
        )

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Route request to appropriate provider and generate response."""

        # Determine which provider to use
        provider_name = self._select_provider(messages, context)
        provider = self.providers.get(provider_name)

        if not provider:
            provider_name = self.default_provider
            provider = self.providers.get(provider_name)

        if not provider:
            raise RuntimeError("No available LLM provider")

        self.logger.info(
            f"Routing request to provider",
            provider=provider_name,
            message_count=len(messages),
            stream=stream,
        )

        return await provider.generate_response(messages, stream=stream, **kwargs)

    def _select_provider(
        self, messages: List[Dict[str, str]], context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Select appropriate provider based on request characteristics."""

        if context:
            # Check for routing hints in context
            routing_hint = context.get("routing_preference")
            if routing_hint and routing_hint in self.routing_rules:
                preferred_provider = self.routing_rules[routing_hint]
                if preferred_provider in self.providers:
                    return preferred_provider

            # Check for sensitive data indicators
            if context.get("has_sensitive_data", False):
                if "local" in self.providers:
                    return "local"

        # Default routing logic
        return self.default_provider

    async def count_tokens(self, text: str, provider: Optional[str] = None) -> int:
        """Count tokens using specified or default provider."""
        provider_name = provider or self.default_provider
        llm_provider = self.providers.get(provider_name)

        if not llm_provider:
            # Fallback estimation
            return len(text) // 4

        return await llm_provider.count_tokens(text)

    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return list(self.providers.keys())

    def get_provider_info(self, provider: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific provider."""
        llm_provider = self.providers.get(provider)
        return llm_provider.get_model_info() if llm_provider else None

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all providers."""
        health_status = {}

        for name, provider in self.providers.items():
            try:
                status = await provider.health_check()
                health_status[name] = status
            except Exception as e:
                health_status[name] = {"status": "unhealthy", "error": str(e)}

        return {
            "overall_status": (
                "healthy"
                if any(
                    status.get("status") == "healthy"
                    for status in health_status.values()
                )
                else "unhealthy"
            ),
            "providers": health_status,
            "default_provider": self.default_provider,
        }

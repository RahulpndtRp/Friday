import openai
from typing import Dict, Any, List, Optional, AsyncGenerator, Union

from src.core.llm.base_llm import BaseLLMProvider
from src.core.config.settings import Settings


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, settings: "Settings"):
        super().__init__(settings)
        self.client = None
        self.model = "gpt-4o-mini"
        self.max_tokens = 4000

    async def initialize(self) -> None:
        """Initialize OpenAI client."""
        try:
            if not self.settings.openai_api_key:
                raise ValueError("OpenAI API key not configured")

            self.client = openai.AsyncOpenAI(api_key=self.settings.openai_api_key)

            # Test the connection
            await self.client.models.list()

            self.is_initialized = True
            self.logger.info("OpenAI provider initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI provider", error=str(e))
            raise

    async def generate_response(
        self, messages: List[Dict[str, str]], stream: bool = False, **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate response using OpenAI."""
        if not self.is_initialized:
            raise RuntimeError("Provider not initialized")

        try:
            model = kwargs.get("model", self.model)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            temperature = kwargs.get("temperature", 0.7)

            if stream:
                return self._stream_response(messages, model, max_tokens, temperature)
            else:
                return await self._complete_response(
                    messages, model, max_tokens, temperature
                )

        except Exception as e:
            self.logger.error(f"Failed to generate response", error=str(e))
            raise

    async def _complete_response(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate complete response."""
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return response.choices[0].message.content

    async def _stream_response(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response."""
        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    async def count_tokens(self, text: str) -> int:
        """Estimate token count (approximate)."""
        # Simple approximation: ~4 characters per token
        return len(text) // 4

    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information."""
        return {
            "provider": "openai",
            "model": self.model,
            "max_tokens": self.max_tokens,
            "supports_streaming": True,
            "supports_function_calling": True,
        }

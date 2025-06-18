from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from enum import Enum


class Environment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"


class Settings(BaseSettings):
    """Application settings with validation and environment variable support."""

    # Application
    app_name: str = Field(
        default="FRIDAY Personal Assistant", description="Application name"
    )
    environment: Environment = Field(
        default=Environment.DEVELOPMENT, description="Runtime environment"
    )
    debug: bool = Field(default=True, description="Debug mode")

    # API Configuration
    api_host: str = Field(default="localhost", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_reload: bool = Field(default=True, description="API auto-reload")

    # LLM Configuration
    default_llm_provider: LLMProvider = Field(
        default=LLMProvider.OPENAI, description="Default LLM provider"
    )
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(
        default=None, description="Anthropic API key"
    )
    google_api_key: Optional[str] = Field(default=None, description="Google API key")

    # Memory Configuration
    memory_db_path: str = Field(
        default="data/memory.db", description="Memory database path"
    )
    vector_db_path: str = Field(
        default="data/vector_db", description="Vector database path"
    )
    max_memory_size: int = Field(default=10000, description="Maximum memory entries")

    # Security
    encryption_key: Optional[str] = Field(
        default=None, description="Encryption key for sensitive data"
    )
    enable_local_processing: bool = Field(
        default=True, description="Enable local processing for sensitive data"
    )

    # Telemetry
    enable_telemetry: bool = Field(
        default=True, description="Enable telemetry and logging"
    )
    log_level: str = Field(default="INFO", description="Logging level")
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

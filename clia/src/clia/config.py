import os
from enum import Enum
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class ExecutionMode(str, Enum):
    """Execution mode for tool calls."""
    CONFIRM = "confirm"
    AUTO = "auto"

class ProviderType(str, Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"

class Settings(BaseSettings):
    """Global configuration settings for CLIA."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    default_provider: ProviderType = Field(
        default=ProviderType.ANTHROPIC,
        description="Default provider to use ('anthropic', 'openai', or 'ollama')"
    )

    anthropic_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Anthropic model name"
    )

    openai_model: str = Field(
        default="gpt-4o",
        description="OpenAI model name"
    )

    ollama_model: str = Field(
        default="qwen2.5-coder:latest",
        description="Default Ollama model (e.g., qwen2.5-coder, llama3.1)"
    )

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server base endpoint URL"
    )

    anthropic_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"),
        description="Anthropic API Key"
    )

    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY"),
        description="OpenAI API Key"
    )

    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.CONFIRM,
        description="Tool execution mode ('confirm' or 'auto')"
    )

    max_loop_interations: int = Field(
        default=25,
        description="Safety limit for max agentic loop turns per task"
    )

    mcp_config_path: str = Field(
        default="mcp_config.json",
        description="Path to MCP servers configuarion file"
    )

settings = Settings()
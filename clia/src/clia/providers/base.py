from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional
from pydantic import BaseModel, Field

class ToolCall(BaseModel):
    """Normalized tool call specification requested by the LLM."""
    id: str = Field(description="Unique tool invocation call ID")
    name: str = Field(description="Name of the tool to execute")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the tool")

class LLMResponse(BaseModel):
    """Normalized response payload from a provider generation call."""
    content: Optional[str] = Field(default=None, description="Text response content from model")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="Tool execution calls requested by model")
    finish_reason: Optional[str] = Field(default=None, descripton="Reason for generation stop (e.g., 'stop', 'tool_calls')")
    raw_response: Optional[Any] = Field(default=None, description="Original raw response object from provider SDK")

class StreamChunk(BaseModel):
    """Normalized streaming chunk emitted during token streaming."""
    delta_text: Optional[str] = Field(default=None, description="Text token fragment")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="Tool call deltas if assembled")
    finish_reason: Optional[str] = Field(default=None, description="Completion status if end of stream")

class BaseProvider(ABC):
    """Abstract base class for all LLM providers in CLIA."""
    def __init__(self, model_name: str, **kwargs: Any):
        self.model_name = model_name

    @abstractmethod
    async def generate_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """
        Generate a non-streaming response from the provider.
        Args:
            messages: List of message dictionaries in standard schema.
            tools: Optional list of formatted tool schemas (JSON/MSP format).
        
        Returns:
            LLMResponse object containing text content and/or tool calls.
        """    
        pass

    @abstractmethod
    async def stream_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream generation tokens and tool call deltas asynchronously.
        
        Args:
            messages: List of message dicitonaries.
            tools: Optional list of tool schemas.

        Returns:
            StreamChunk insstanes carrying incremental updates.
        """
        pass
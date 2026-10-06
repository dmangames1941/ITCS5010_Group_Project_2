import logging
from typing import Any, Callable, Dict, List, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

class LocalLLMClient:
    """Wrapper for managing local model execution and tool calling."""

    def __init__(
            self,
            model_name: str = "llama3.1",
            base_url: str = "http://localhost:11434",
            temperature: float = 0.0,
            reasoning: Optional[bool] = None
    ):
        """
        Initialize the local ChatOllama client.

        Args:
            model_name: Name of the Ollama model(e.g., 'llama3.1', 'qwen2.5').
            base_url: Endpoint for the local Ollama instance.
            temperature: Sampling temperature
            reasoning: Enable/disable explicit reasoning tags if supported.
        """
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature

        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
            reasoning=reasoning
        )
        self._bound_llm = None

    def bind_tools(self, tools:List[Callable | BaseTool | Dict[str, Any]]) -> None:
        """Bind Python functions or LangChain tools to the local model."""
        logger.info(f"Binding {len(tools)} tools to local model {self.model_name}")
        self._bound_llm = self.llm.bind_tools(tools)

    def generate(
            self,
            messages: List[BaseMessage],
            system_prompt: Optional[str] = None
    ) -> BaseMessage:
        """
        Execute a standard generation loop with optional system context.
        
        Args:
            messages: Conversation history.
            system_prompt: Optional system prompt override.
        
        Returns:
            BaseMessage: Generated response message containing text or tool calls.
        """
        formatted_messages = []
        if system_prompt:
            formatted_messages.appent(SystemMessage(content=system_prompt))
        formatted_messages.extend(messages)
        runner = self._bound_llm if self._bound_llm else self.llm
        return runner.invoke(formatted_messages)

    def stream(
            self,
            messages: List[BaseMessage],
            system_prompt: Optional[str] = None
    ):
        """Stream chunks from the local model."""
        formatted_messages = []
        if system_prompt:
            formatted_messages.append(SystemMessage(content=system_prompt))
        formatted_messages.extend(messages)
        runner = self._bound_llm if self._bound_llm else self.llm
        yield from runner.stream(formatted_messages)
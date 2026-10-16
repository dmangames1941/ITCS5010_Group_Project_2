
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


DEFAULT_SYSTEM_PROMPT = """You are CLIA (Command Line Interface Assistant), an autonomous AI coding assistant running in a terminal REPL environment.

Core Responsibilities & Capabilities:
1. Assist developers by understanding tasks, analyzing local codebases, reading and writing files, executing shell commands, searching documentation, and querying external resources using MCP tools.
2. Operate in an agentic loop: Reason about the task -> Call appropriate tools -> Observe execution results -> Refine the plan -> Repeat until the task is completed.
3. Keep responses concise and action-oriented. Prioritize accurate tool execution over explaining intended actions.
4. When modifying code, follow existing repository patterns and make precise changes.
"""


class AgentState(BaseModel):
    """
    Tracks conversation history and state during the agentic loop.
    """

    system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT)
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    max_history_messages: int = Field(
        default=40,
        ge=1,
        description="Maximum number of messages kept before pruning."
    )

    def get_full_messages(self) -> List[Dict[str, Any]]:
        """
        Returns the system prompt followed by conversation history.
        """
        system_msg = {
            "role": "system",
            "content": self.system_prompt
        }
        return [system_msg] + self.messages

    def add_user_message(self, content: str) -> None:
        """
        Adds a user message to conversation history.
        """
        self.messages.append({
            "role": "user",
            "content": content
        })
        self._prune_context_if_needed()

    def add_assistant_message(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Adds an assistant response and optional tool calls.
        """
        msg: Dict[str, Any] = {"role": "assistant"}

        if content:
            msg["content"] = content

        if tool_calls:
            msg["tool_calls"] = tool_calls

        self.messages.append(msg)
        self._prune_context_if_needed()

    def add_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        content: str
    ) -> None:
        """
        Adds an MCP tool execution result to conversation history.
        """
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": content
        })
        self._prune_context_if_needed()

    def clear(self) -> None:
        """
        Clears conversation history for a new session.
        """
        self.messages.clear()

    def _prune_context_if_needed(self) -> None:
        """
        Removes older messages when the history exceeds its limit.
        Preserves the initial user message when possible and avoids
        starting the retained history with an orphaned tool result.
        """
        if len(self.messages) <= self.max_history_messages:
            return

        initial_user_msg = None

        if self.messages and self.messages[0]["role"] == "user":
            initial_user_msg = self.messages[0]

        recent_messages = self.messages[-self.max_history_messages:]

        while recent_messages and recent_messages[0].get("role") == "tool":
            recent_messages.pop(0)

        if initial_user_msg and initial_user_msg not in recent_messages:
            self.messages = [initial_user_msg] + recent_messages
        else:
            self.messages = recent_messages

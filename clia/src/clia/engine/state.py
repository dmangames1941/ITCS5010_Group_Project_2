from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

DEFAULT_SYSTEM_PROMPT = """You are CLIA (Command Line Interface Assistant), an autonomous AI coding assistant running in a terminal REPL environment.

Core Responsitilities & Capabilities:
1. You assist developers by understanding tasks, analyzing local codebases, reading/writing files, executing shell commands, searching documentation, and querying external resoures using MCP (Model Context Protocol) tools.
2. You operate in an agentic loop: Reason about the task -> Call appropriate tools -> Observe execution results -> Refine your plan -> Interate until the task is fully completed.
3. Keep your reasoning concise and action-oriented. Prioritize taking accurate actions via available tools over explaining what you intend to do.
4. When writing code or editing files, ensure high code quality, follow existing repository patterns, and make precise modifications.
"""

class AgentState(BaseModel):
    """
    Tracks conversation state and history across iterations of the agentic loop.
    """
    system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT)
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    max_history_messages: int = Field(
        default=40,
        description="Meximum number of historical turns kept before context pruning triggers."
    )

    def get_full_messages(self) -> List[Dict[str, Any]]:
        """
        Returns the system prompt prepended to the active conversation hisotry.
        """
        system_msg = {"role": "system", "content": self.system_prompt}
        return [system_msg] + self.messages

    def add_user_message(self, content: str) -> None:
        """
        Appends a user task oor message to state.
        """
        self.messages.append({"role": "user", "content": content})
        self._prune_context_if_needed()

    def add_assistant_message(
            self,
            content: Optional[str] = None,
            tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Appends an assistant turn (text output and/or requested tool calls) to state.
        """
        msg: Dict[str, Any] = {"role": "assistant"}
        if content:
            msg["content"] = content
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)
        self._prune_context_if_needed()

    def add_tool_result(self, tool_call_id: str, tool_name: str, content: str) -> None:
        """
        Appends a tool execution observation back into state.
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
        Clears message history for a fresh session.
        """
        self.messages.clear()

    def _prune_context_if_needed(self) -> None:
        """
        Prunes older messages if history exceeds max_history_messsages while preserving recent context and avoiding ophaned tool responses at the boundary.
        """
        if len(self.messages <= self.max_history_messages):
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
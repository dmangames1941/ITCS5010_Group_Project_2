import json
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field
from clia.cli.display import display_tool_confirmation
from clia.cli.prompts import ask_confirmation

class ExecutionDecision(BaseModel):
    """
    Represents the permission decision for a tool call.
    """
    tool_call_id: str
    tool_name: str
    arguments: Dict[str, Any]
    approved: bool
    user_feedback: Optional[str] = None

class ExecutionManager:
    """
    Evaluates and enforces tool execution policy based on the agent's current mode.
    """
    def __init__(self, mode: str = "confirm"):
        """
        Args:
            mode; Execution mode, either 'confirm' (prompt user per tool) or 'auto' (execute without prompt).
        """
        self.mode = mode.lower()

    def set_mode(self, mode:str) -> None:
        """
        Updates the execution mode at runtime.
        """
        if mode.lower() not in ("confirm", "auto"):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'confirm' or 'auto'.")
        self.mode = mode.lower()

    def evaluate_tool_call(
            self,
            tool_call_id: str,
            tool_name: str,
            arguments: Dict[str, Any]
    ) -> ExecutionDecision:
        """
        Determines whether a tool call is authorized to execute.

        In 'auto' mode, tool calls pass automatically.
        In 'confirm' mode, details are rendered to stdout and the user is prompted [y/N].
        """
        if self.mode == "auto":
            return ExecutionDecision(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=arguments,
                approved=True
            )

        display_tool_confirmation(tool_name, arguments)
        user_choice = ask_confirmation(f"Execute tool '{tool_name}'?", default=True)

        if user_choice:
            return ExecutionDecision(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=arguments,
                approved=True
            )
        else:
            return ExecutionDecision(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=arguments,
                approved=False,
                user_feedback="User rejected execution of this tool call."
            )

    async def execute_or_skip(
            self,
            tool_call_id: str,
            tool_name: str,
            arguments: Dict[str, Any],
            executor_func: Callable[[str, Dict[str, Any]], Any]
    ) -> Dict[str, Any]:
        """
        Evaluates permissions and executes the tool via 'executor_func' if approved.
        Returns:
            Standardized dictionary containing the tool observation or denial feedback.
        """
        decision = self.evaluate_tool_call(tool_call_id, tool_name, arguments)
        if not decision.approved:
            return {
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "content": f"[Tool Execution Cancelled]: {decision.user_feedback}",
                "status": "denied"
            }

        try:
            result = await executor_func(tool_name, arguments)

            if isinstance(result, (dict, list)):
                content_str = json.dumps(result, indent=2)
            else:
                content_str = str(result)
            return {
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "content": content_str,
                "status": "success"
            }
        except Exception as e:
            return {
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "content": f"[Tool Execution Error]: {str(e)}",
                "status": "error"
            }
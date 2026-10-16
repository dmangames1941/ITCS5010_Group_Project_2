
import json
from typing import Any, Callable, Dict, List, Optional

from clia.config import Settings
from clia.engine.execution import ExecutionManager
from clia.engine.state import AgentState
from clia.providers.base import BaseProvider


class AgentLoop:
    """
    Manages the agentic loop:
    1. Receive a user task.
    2. Send messages and available tools to the LLM.
    3. Execute requested MCP tools.
    4. Return tool results to the LLM.
    5. Repeat until the task is complete.
    """

    def __init__(
        self,
        provider: BaseProvider,
        state: Optional[AgentState] = None,
        execution_manager: Optional[ExecutionManager] = None,
        mcp_client: Optional[Any] = None,
        settings: Optional[Settings] = None,
        max_iterations: int = 15
    ):
        self.provider = provider
        self.settings = settings or Settings()
        self.state = state or AgentState()

        self.execution_manager = (
            execution_manager
            or ExecutionManager(mode=self.settings.execution_mode)
        )

        self.mcp_client = mcp_client
        self.max_iterations = max_iterations

    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """Retrieve available tools from connected MCP servers."""

        if self.mcp_client and hasattr(self.mcp_client, "get_tools"):
            try:
                tools = await self.mcp_client.get_tools()
                return tools if isinstance(tools, list) else []
            except Exception as e:
                print(f"[Warning] Failed to fetch MCP tools: {e}")
                return []

        return []

    async def execute_mcp_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """Execute a tool through the MCP client."""

        if not self.mcp_client:
            raise RuntimeError("MCP client is not connected.")

        if hasattr(self.mcp_client, "call_tool"):
            return await self.mcp_client.call_tool(
                tool_name, arguments
            )

        elif hasattr(self.mcp_client, "execute_tool"):
            return await self.mcp_client.execute_tool(
                tool_name, arguments
            )

        raise AttributeError(
            "MCP client missing 'call_tool' or 'execute_tool' method."
        )

    async def run_turn(
        self,
        user_input: str,
        token_callback: Optional[Callable[[str], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Run an autonomous agent turn until completion."""

        self.state.add_user_message(user_input)

        iteration = 0
        final_text_response = ""

        while iteration < self.max_iterations:
            iteration += 1

            if status_callback:
                status_callback(
                    f"CLIA is thinking (iteration {iteration})..."
                )

            available_tools = await self.get_available_tools()
            full_messages = self.state.get_full_messages()

            response = await self.provider.generate_response(
                messages=full_messages,
                tools=available_tools if available_tools else None,
                token_callback=token_callback
            )

            # Support both dictionary and Pydantic responses.
            if isinstance(response, dict):
                assistant_content = response.get("content") or ""
                raw_tool_calls = response.get("tool_calls") or []
            else:
                assistant_content = response.content or ""
                raw_tool_calls = response.tool_calls or []

            tool_calls = [
                call.model_dump() if hasattr(call, "model_dump") else call
                for call in raw_tool_calls
            ]

            self.state.add_assistant_message(
                content=assistant_content,
                tool_calls=tool_calls if tool_calls else None
            )

            if assistant_content:
                final_text_response = assistant_content

            # Stop when the model has no more tools to call.
            if not tool_calls:
                break

            for index, tool_call in enumerate(tool_calls):
                call_id = tool_call.get(
                    "id", f"call_{iteration}_{index}"
                )

                function = tool_call.get("function") or {}

                tool_name = (
                    tool_call.get("name")
                    or function.get("name")
                )

                arguments = tool_call.get("arguments")

                if arguments is None:
                    arguments = function.get("arguments", {})

                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw_input": arguments}

                if not isinstance(arguments, dict):
                    arguments = {"raw_input": str(arguments)}

                if not tool_name:
                    self.state.add_tool_result(
                        tool_call_id=call_id,
                        tool_name="unknown",
                        content="Tool name was not provided."
                    )
                    continue

                if status_callback:
                    status_callback(
                        f"Evaluating tool call '{tool_name}'..."
                    )

                execution_result = (
                    await self.execution_manager.execute_or_skip(
                        tool_call_id=call_id,
                        tool_name=tool_name,
                        arguments=arguments,
                        executor_func=self.execute_mcp_tool
                    )
                )

                self.state.add_tool_result(
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    content=str(
                        execution_result.get("content", "")
                    )
                )

            # The next iteration sends tool results back to the LLM.

        else:
            if status_callback:
                status_callback(
                    "[Warning] Reached maximum tool loop iterations."
                )

        return final_text_response

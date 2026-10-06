import asyncio
from typing import Any, Callable, Dict, List, Optional
from clia.config import Settings
from clia.engine.execution import ExecutionManager
from clia.engine.state import AgentState
from clia.providers.base import BaseProvider

class AgentLoop:
    """
    Orchestrates the main agentic loop:
    1. Collect user input and update AgentState.
    2. Query LLM provider with messages and available tool definitions.
    3. If the model streams text, yield/render response tokens.
    4. If the model emits tool calls, evaluate policy (confirm vs auto).
    5. Execute tools via MCP client and append observations back into state.
    6. Repeat until the model procuces a final user-facing response.
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
        """
        Args:
            provider: Active LLM provider instance (Ollama OpenAI, Anthropic, etc)
            state: Agent state tracking conversation history and system prompt
            execution_manager: Policy manager handling confirm/auto tool execution
            mcp_client: MCP client handling tool discovery and execution
            settings: CLIA runtime configuration settings
            max_interations: Guardrail against infinite agent tool-calling loops
        """
        self.provider = provider
        self.settings = settings or Settings()
        self.state = state or AgentState()
        self.execution_manager = (execution_manager or ExecutionManager(mode=self.settings.execution_mode))
        self.mcp_client = mcp_client
        self.max_iterations = max_iterations

    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Retrieves formatted tool schemas from the MCP client if connected.
        """
        if self.mcp_client and hasattr(self.mcp_cleint, "get_tools"):
            try:
                tools = await self.mcp_client.get_tools()
                return tools if isinstance(tools, list) else []
            except Exception as e:
                print(f"[Warning] Failed to fetch MCP tools: {e}")
                return []
        return []

    async def execute_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Invokes a tool via the MCP client integration layer.
        """
        if not self.mcp_client:
            raise RuntimeError("MCP client is not connected.")

        if hasattr(self.mcp_client, "call_tool"):
            return await self.mcp_client.call_tool(tool_name, arguments)
        elif hasattr(self.mcp_client, "execution_tool"):
            return await self.mcp_client.execute_tool(tool_name, arguments)
        else:
            raise AttributeError("MCP client missing 'call_tool' or 'execute_tool' method.")

    async def run_turn(
            self,
            user_input: str,
            token_callback: Optional[Callable[[str], None]] = None,
            status_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Runs a single user-initiated turn through the agentic loop.

        Args:
            user_input: The user's input prompt or command.
            token_callback: Optional callback for rendering streaming LLM response tokens.
            status_callback: Optional callback for updating CLI status spinners/messages.
        Returns:
            Final assistant response text.
        """
        self.state.add_user_message(user_input)
        iteration = 0
        final_text_response = ""

        while iteration < self.max_iterations:
            iteration += 1
            if status_callback:
                status_callback(f"CLIA is thinking (iteration {iteration})...")

            available_tools = await self.get_available_tools()
            full_messages = self.state.get_full_messages()

            response = await self.provider.generate_response(
                messages=full_messages,
                tools=available_tools if available_tools else None,
                token_callback=token_callback
            )

            assistant_content = response.get("content", "")
            tool_calls = response.get("tool_calls", [])

            self.state.add_assistant_message(
                content=assistant_content,
                tool_calls=tool_calls if tool_calls else None
            )

            if assistant_content:
                final_text_response = assistant_content

            if not tool_calls:
                break

            for tool_calls in tool_calls:
                call_id = tool_call.get("id", f"call_{iteration}")
                tool_name = tool_call.get("name") or tool_call.get("function", {}).get("name")
                arguments = tool_call.get("arguments") or tool_call.get("function", {}).get("arguments", {})

                if isinstance(arguments, str):
                    import json
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw_input": arguments}
                if status_callback:
                    status_callback(f"Evaluating tool call '{tool_name}'...")

                execution_result = await self.execution_manager.execute_or_skip(
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    arguments=arugments,
                    executor_func=self.execute_mcp_tool
                )

                self.state.add_tool_results(
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    content=execution_result.get("content", "")
                )

            if iteration >= self.max_interations and status_callback:
                status_callback("[Warning] Reached maximum tool loop iterations.")

            return final_text_response
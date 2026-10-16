
import asyncio

from clia.engine.loop import AgentLoop
from clia.engine.execution import ExecutionManager
from clia.providers.base import LLMResponse, ToolCall


class FakeMCPClient:
    def __init__(self):
        self.tool_called = False

    async def get_tools(self):
        return [{
            "type": "function",
            "function": {
                "name": "filesystem__list_directory",
                "description": "List files in a directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    },
                    "required": ["path"]
                }
            }
        }]

    async def call_tool(self, name, arguments):
        assert name == "filesystem__list_directory"
        assert arguments["path"] == "./demo_files"

        self.tool_called = True

        return {
            "content": [
                {"type": "text", "text": "example.py"}
            ]
        }


class FakeProvider:
    def __init__(self):
        self.calls = 0

    async def generate_response(
        self, messages, tools=None, token_callback=None
    ):
        self.calls += 1

        if self.calls == 1:
            assert tools is not None

            return LLMResponse(
                content="Checking project files.",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="filesystem__list_directory",
                        arguments={"path": "./demo_files"}
                    )
                ]
            )

        assert messages[-1]["role"] == "tool"

        return LLMResponse(
            content="I found example.py in the project.",
            tool_calls=[]
        )


async def main():
    provider = FakeProvider()
    client = FakeMCPClient()

    agent = AgentLoop(
        provider=provider,
        mcp_client=client,
        execution_manager=ExecutionManager(mode="auto")
    )

    result = await agent.run_turn(
        "List the files in the project."
    )

    assert client.tool_called
    assert provider.calls == 2
    assert result == "I found example.py in the project."

    print("Agentic loop test passed!")
    print("Tool discovery: passed")
    print("Tool execution: passed")
    print("Observation returned to LLM: passed")
    print("Multiple iterations: passed")


if __name__ == "__main__":
    asyncio.run(main())

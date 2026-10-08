import json
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(self, config_path="mcp_config.json"):
        self.config_path = config_path
        self.stack = AsyncExitStack()
        self.sessions = {}
        self.tool_map = {}

    async def __aenter__(self):
        await self.stack.__aenter__()
        await self.connect_servers()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stack.__aexit__(exc_type, exc, tb)

    async def connect_servers(self):
        config = json.loads(Path(self.config_path).read_text())

        for server_name, server_config in config["mcpServers"].items():
            params = StdioServerParameters(
                command=server_config["command"],
                args=server_config.get("args", []),
                env={**os.environ, **server_config.get("env", {})}
            )

            read, write = await self.stack.enter_async_context(
                stdio_client(params)
            )

            session = await self.stack.enter_async_context(
                ClientSession(read, write)
            )

            await session.initialize()
            self.sessions[server_name] = session
            print(f"Connected to {server_name}")

    async def get_tools(self):
        all_tools = []
        self.tool_map.clear()

        for server_name, session in self.sessions.items():
            cursor = None

            while True:
                response = await session.list_tools(cursor=cursor)

                for tool in response.tools:
                    name = f"{server_name}__{tool.name}"
                    self.tool_map[name] = (server_name, tool.name)

                    all_tools.append({
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema
                        }
                    })

                cursor = response.nextCursor
                if cursor is None:
                    break

        return all_tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]):
        if tool_name not in self.tool_map:
            await self.get_tools()

        if tool_name not in self.tool_map:
            raise ValueError(f"Unknown tool: {tool_name}")

        server_name, original_name = self.tool_map[tool_name]
        session = self.sessions[server_name]

        result = await session.call_tool(original_name, arguments)

        if result.isError:
            raise RuntimeError(str(result.content))

        return {
            "content": [
                block.model_dump(mode="json")
                for block in result.content
            ],
            "structured_content": result.structuredContent
        }

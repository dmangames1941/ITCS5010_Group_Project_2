import asyncio
import json
from pathlib import Path

from clia.mcp.client import MCPClient


async def main():
    async with MCPClient() as client:
        tools = await client.get_tools()

        print("\nAvailable MCP tools:")
        for tool in tools:
            print("-", tool["function"]["name"])

        folder = str(Path("demo_files").resolve())

        print("\nTesting filesystem server:")
        result = await client.call_tool(
            "filesystem__list_directory",
            {"path": folder}
        )
        print(json.dumps(result, indent=2))

        print("\nTesting external resources server:")
        result = await client.call_tool(
            "fetch__fetch",
            {"url": "https://example.com"}
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

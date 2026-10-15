import asyncio
from pathlib import Path
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    """Launch the server, discover its tools, and test retrieval over MCP."""
    server_path = Path(__file__).resolve().parent / "server.py"

    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
    )

    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            tool_names = [tool.name for tool in tools.tools]
            print("Available tools:", tool_names)

            if "search_library_docs" not in tool_names:
                raise RuntimeError("Documentation search tool was not found.")

            result = await session.call_tool(
                "search_library_docs",
                arguments={
                    "query": "How do I connect to the database?",
                    "top_k": 3,
                },
            )

            if result.isError:
                raise RuntimeError(f"MCP tool failed: {result.content}")

            print("\nTool response:")
            for content in result.content:
                if content.type == "text":
                    print(content.text)


if __name__ == "__main__":
    asyncio.run(main())

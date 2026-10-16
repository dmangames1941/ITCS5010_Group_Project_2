
import asyncio
import json
from pathlib import Path

from clia.mcp.client import MCPClient


async def main():
    folder = Path("demo_files").resolve()
    folder.mkdir(parents=True, exist_ok=True)

    async with MCPClient() as client:
        tools = await client.get_tools()
        tool_names = [tool["function"]["name"] for tool in tools]

        print("\nAvailable MCP tools:")
        for name in tool_names:
            print("-", name)

        required = [
            "filesystem__list_directory",
            "fetch__fetch",
            "custom_rag__search_library_docs"
        ]

        for name in required:
            if name not in tool_names:
                raise RuntimeError(f"Missing MCP tool: {name}")

        print("\nTesting filesystem server:")
        result = await client.call_tool(
            "filesystem__list_directory",
            {"path": str(folder)}
        )
        print(json.dumps(result, indent=2))

        print("\nTesting external resources server:")
        result = await client.call_tool(
            "fetch__fetch",
            {"url": "https://example.com"}
        )
        print(json.dumps(result, indent=2))

        print("\nTesting custom RAG server:")
        result = await client.call_tool(
            "custom_rag__search_library_docs",
            {
                "query": "How do I create a LangChain agent?",
                "top_k": 3
            }
        )
        print(json.dumps(result, indent=2))

        data = result.get("structured_content")
        if data is None:
            for block in result["content"]:
                if block.get("type") == "text":
                    try:
                        data = json.loads(block["text"])
                        break
                    except json.JSONDecodeError:
                        continue

        if not isinstance(data, dict) or data.get("result_count", 0) < 1:
            raise RuntimeError("RAG returned no documentation passages")

        print("\nAll three MCP server tests passed!")


if __name__ == "__main__":
    asyncio.run(main())

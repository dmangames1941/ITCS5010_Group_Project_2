"""CLIA interactive entry point.

Run from the project's clia directory: python -m clia.main
"""

import argparse
import asyncio
import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from clia.cli.display import display_tool_results, display_welome_banner, print_status
from clia.config import Settings
from clia.engine.execution import ExecutionManager
from clia.engine.loop import AgentLoop
from clia.mcp.client import MCPClient
from clia.providers.base import BaseProvider, LLMResponse, StreamChunk, ToolCall


SYSTEM_PROMPT = (
    "You are CLIA, a coding assistant. Use available MCP tools to inspect and "
    "modify files when needed. Only modify files the user requests. "
    "Report what you actually verified, and never claim a tool ran if it did not."
)


class LangChainProvider(BaseProvider):
    """Adapt LangChain chat models to CLIA's BaseProvider interface."""

    def __init__(self, model, model_name):
        super().__init__(model_name=model_name)
        self.model = model

    @staticmethod
    def _messages(messages):
        converted = [SystemMessage(content=SYSTEM_PROMPT)]
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content") or ""

            if role == "system":
                converted.append(SystemMessage(content=content))
            elif role == "user":
                converted.append(HumanMessage(content=content))
            elif role == "assistant":
                calls = []
                for i, call in enumerate(message.get("tool_calls") or []):
                    fn = call.get("function") or {}
                    name = call.get("name") or fn.get("name")
                    arguments = call.get("arguments", fn.get("arguments", {}))
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {"raw_input": arguments}
                    if name:
                        calls.append({
                            "name": name,
                            "args": arguments if isinstance(arguments, dict) else {},
                            "id": call.get("id") or f"previous_call_{i}",
                            "type": "tool_call",
                        })
                converted.append(AIMessage(content=content, tool_calls=calls))
            elif role == "tool":
                converted.append(ToolMessage(
                    content=str(content),
                    tool_call_id=message.get("tool_call_id") or message.get("id") or "unknown",
                ))
        return converted

    async def generate_response(self, messages, tools=None, **kwargs):
        model = self.model.bind_tools(tools) if tools else self.model
        response = await model.ainvoke(self._messages(messages))
        calls = [
            ToolCall(
                id=call.get("id") or f"call_{i}",
                name=call["name"],
                arguments=call.get("args") or {},
            )
            for i, call in enumerate(response.tool_calls or [])
        ]
        content = response.content
        if not isinstance(content, str):
            content = "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        return LLMResponse(content=content, tool_calls=calls)

    async def stream_response(self, messages, tools=None, **kwargs):
        """Basic streaming API; AgentLoop currently calls generate_response."""
        model = self.model.bind_tools(tools) if tools else self.model
        async for chunk in model.astream(self._messages(messages)):
            if isinstance(chunk.content, str) and chunk.content:
                yield StreamChunk(delta_text=chunk.content)


class VisibleMCPClient:
    """Print observations without changing the team's MCP client."""

    def __init__(self, client):
        self.client = client

    async def get_tools(self):
        return await self.client.get_tools()

    async def call_tool(self, name, arguments):
        print_status(f"Calling {name}")
        result = await self.client.call_tool(name, arguments)
        display_tool_results(name, json.dumps(result, indent=2, default=str))
        return result


def build_model(provider, name, settings):
    if provider == "ollama":
        from clia.providers.local import LocalLLMClient
        return LocalLLMClient(
            model_name=name, base_url=settings.ollama_base_url
        ).llm
    if provider == "openai":
        from clia.providers.cloud import get_openai_chat
        return get_openai_chat(model_name=name, temperature=0)
    if provider == "anthropic":
        from clia.providers.cloud import get_anthropic_chat
        return get_anthropic_chat(model_name=name, temperature=0)
    raise ValueError(f"Unsupported provider: {provider}")


async def main():
    parser = argparse.ArgumentParser(description="CLIA coding assistant")
    parser.add_argument("--provider", choices=["ollama", "openai", "anthropic"], default="ollama")
    parser.add_argument("--model", help="Override the provider's default model")
    parser.add_argument("--mode", choices=["confirm", "auto"], default="confirm")
    args = parser.parse_args()

    settings = Settings()
    default_models = {
        "ollama": settings.ollama_model,
        "openai": settings.openai_model,
        "anthropic": settings.anthropic_model,
    }
    model_name = args.model or default_models[args.provider]
    model = build_model(args.provider, model_name, settings)
    provider = LangChainProvider(model, model_name)

    config_path = Path(settings.mcp_config_path)
    if not config_path.is_file():
        raise FileNotFoundError("Start CLIA from the clia directory (mcp_config.json not found).")
    Path("demo_files").mkdir(exist_ok=True)

    async with MCPClient(str(config_path)) as client:
        visible_client = VisibleMCPClient(client)
        tools = await visible_client.get_tools()
        print(f"Discovered {len(tools)} MCP tools across {len(client.sessions)} servers.")
        agent = AgentLoop(
            provider=provider,
            mcp_client=visible_client,
            execution_manager=ExecutionManager(mode=args.mode),
            settings=settings,
            max_iterations=settings.max_loop_interations,
        )
        display_welome_banner(mode=args.mode, provider=f"{args.provider}: {model_name}")

        while True:
            try:
                task = input("\nclia> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if task.lower() in {"exit", "quit"}:
                break
            if not task:
                continue
            try:
                answer = await agent.run_turn(task, status_callback=print_status)
                print(f"\nCLIA: {answer}")
            except Exception as exc:
                print(f"Task failed: {exc}")


if __name__ == "__main__":
    asyncio.run(main())

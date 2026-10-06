import os
from typing import Any, List, Optional, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

def get_openai_chat(
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        **kwargs: Any
) -> BaseChatModel:
    """
    Initializes and returns a ChatOpenAI model instance
    
    Args:
        model_name: The OpenAI model identifier.
        temperature: Sampling temperature.
        api_key: Optional explicit API key (defaults to OPENAI_API_KEY env var).
        **kwargs: Additional parameters passed to ChatOpenAI.
    """
    from langchain_openai import ChatOpenAI

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key missing. Set OPENAI_API_KEY env variable or pass api_key.")
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=key,
        **kwargs
    )

def get_anthropic_chat(
        model_name: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        **kwargs: Any
) -> BaseChatModel:
    """
    Initializes and returns a ChatAnthropic model instance.

    Args:
        model_name: The Anthropic model identifier.
        temperature: Sampling temperature.
        api_key: Optional explicit API key (defaults to ANTHROPIC_API_KEY env var).
        **kwargs: Additional parameters passed to ChatAnthropic.
    """
    from langchain_anthropic import ChatAnthropic

    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("Anthropic API key missing. Set ANTHROPIC_API_KEY env variable or pass api_key.")
    return ChatAnthropic(
        model=model_name,
        temperature=temperature,
        api_key=key,
        **kwargs
    )

def get_google_chat(
        model_name: str = "gemini-1.5-pro",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        **kwargs: Any
) -> BaseChatModel:
    """
    Initializes and returns a ChatGoogleGenerativeAI model instance.

    Args:
    model_name: The Gemini/Google model identifier.
    temperature: Sampling temperature.
    api_key: Optional explicit API key (defaults to GOOGLE_API_KEY env var).
    **kwargs: Additional parameters passed to ChatGoogleGenerativeAI.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Google API key missing. Set GOOGLE_API_KEY env variable or pass api_key")
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=key,
        **kwargs
    )

def bind_cloud_tools(
        model: BaseChatModel,
        tools: List[Union[BaseTool, Any]],
        strict: Optional[bool] = None
) -> Any:
    """
    Binds a sequence of tools to a cloud language model.

    Args:
        model: The LangChain chat model instance.
        tools: List of tool instances, functions, or schemas.
        strict: Optional flag or structured/strict tool call enforcement.

    Returns:
        A Runnable object bound with tools.
    """
    if hasattr(model, "bind_tools"):
        kwargs = {}
        if strict is not None:
            kwargs["strict"] = strict
        return model.bind_tools(tools, **kwargs)
    raise AttributeError(f"Model {type(model).__name__} does not support bind_tools.")
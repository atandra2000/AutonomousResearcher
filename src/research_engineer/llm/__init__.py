"""Provider-agnostic LLM layer.

Public surface::

    from research_engineer.llm import (
        LLMProvider,
        LLMRequest,
        LLMResponse,
        LLMMessage,
        LLMRole,
        LLMUsage,
        ToolDefinition,
        ToolCall,
        ToolResult,
        ProviderError,
        OllamaCloudProvider,
        ModelRouter,
        ProviderFactory,
        get_factory,
        get_router,
        reset_factory,
        reset_router,
        load_config,
        register_provider_type,
        run_react_loop,
        ReActLoopConfig,
        ReActStep,
        ReActResult,
        ReActTermination,
    )
"""

from research_engineer.llm.base import (
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMRole,
    LLMUsage,
    ProviderError,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from research_engineer.llm.factory import (
    AgentModelSpec,
    ProviderFactory,
    get_factory,
    load_config,
    register_provider_type,
    reset_factory,
)
from research_engineer.llm.ollama_provider import OllamaCloudProvider
from research_engineer.llm.react_loop import (
    ReActLoopConfig,
    ReActResult,
    ReActStep,
    ReActTermination,
    run_react_loop,
)
from research_engineer.llm.router import ModelRouter, get_router, reset_router

__all__ = [
    # Base
    "LLMProvider",
    "LLMMessage",
    "LLMRequest",
    "LLMResponse",
    "LLMRole",
    "LLMUsage",
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
    "ProviderError",
    # Providers
    "OllamaCloudProvider",
    # Factory
    "ProviderFactory",
    "AgentModelSpec",
    "get_factory",
    "load_config",
    "register_provider_type",
    "reset_factory",
    # Router
    "ModelRouter",
    "get_router",
    "reset_router",
    # ReAct loop
    "run_react_loop",
    "ReActLoopConfig",
    "ReActStep",
    "ReActResult",
    "ReActTermination",
]

# Files

- [Agent System](agents.md) - Responsibilities and safety boundaries for the specialized agents that research, plan, generate patches, evaluate experiments, retain memory, and coordinate autonomous change-making.
- [LLM Layer](llm-layer.md) - How the system loads provider configuration, binds a provider and model to each agent, and applies resilient, observable model calls. Use this page to safely change vendor, model, pricing, or agent-level LLM behavior.
- [Memory and Retrieval](memory.md) - Repository-scoped code memory combines a durable SQLite catalog, a derived symbol graph, and vector-backed hybrid retrieval to ground planning and code search. This page explains lifecycle, persistence boundaries, retrieval behavior, and safety constraints for index changes.
- [Tools and Safe Execution](tools.md) - Typed tools provide agents with focused research, persistence, repository, and terminal capabilities. This page documents the state-changing and trust-boundary operations, their safeguards, and their present limitations.

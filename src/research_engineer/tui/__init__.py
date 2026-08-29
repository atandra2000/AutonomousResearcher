"""Interactive terminal UI for research-engineer.

A codex-style chat session: research goals, paper analysis, and free-form
LLM questions are handled as conversational turns in the terminal. Slash
commands dispatch to the platform's agents; output is rendered with rich.
"""

from research_engineer.tui.session import ChatSession

__all__ = ["ChatSession"]

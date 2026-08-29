"""Tests for optional LangGraph persistence wiring."""

from __future__ import annotations

import pytest

from research_engineer.graphs.checkpoints import checkpoint_from_environment


@pytest.mark.asyncio
async def test_checkpoint_is_disabled_without_a_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RE_LANGGRAPH_CHECKPOINT_DSN", raising=False)

    async with checkpoint_from_environment() as checkpointer:
        assert checkpointer is None

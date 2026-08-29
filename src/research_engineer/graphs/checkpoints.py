"""Opt-in LangGraph checkpoint resources."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any


@asynccontextmanager
async def checkpoint_from_environment() -> AsyncIterator[Any | None]:
    """Yield a Postgres checkpointer when explicitly configured."""
    dsn = os.getenv("RE_LANGGRAPH_CHECKPOINT_DSN")
    if not dsn:
        yield None
        return

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as exc:
        raise RuntimeError(
            "Install the service extra to use RE_LANGGRAPH_CHECKPOINT_DSN"
        ) from exc

    async with AsyncPostgresSaver.from_conn_string(dsn) as checkpointer:
        await checkpointer.setup()
        yield checkpointer

"""Async rate limiter with multiple strategies."""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""

    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiter."""

    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    rate: float = 1.0  # requests per second
    capacity: int = 10
    window_seconds: float = 60.0
    burst_size: int = 5

    @classmethod
    def default(cls) -> "RateLimitConfig":
        return cls()

    @classmethod
    def strict(cls) -> "RateLimitConfig":
        return cls(rate=0.5, capacity=5)


class RateLimiter:
    """Rate limiter with multiple strategies."""

    def __init__(self, config: RateLimitConfig):
        self._config = config
        self._strategy: Any = None
        self._init_strategy()
        self._total_requests = 0
        self._blocked_requests = 0

    def _init_strategy(self) -> None:
        """Initialize rate limiting strategy."""
        if self._config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            from .base_cache import TokenBucketRateLimiter

            self._strategy = TokenBucketRateLimiter(
                config=type(
                    "TokenBucketConfig",
                    (),
                    {
                        "rate": self._config.rate,
                        "capacity": self._config.capacity,
                        "refill_interval": 0.1,
                    },
                )()
            )
        elif self._config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            from .base_cache import SlidingWindowRateLimiter

            self._strategy = SlidingWindowRateLimiter(
                requests_per_window=int(
                    self._config.rate * self._config.window_seconds
                ),
                window_seconds=self._config.window_seconds,
            )

    async def is_allowed(self) -> bool:
        """Check if request is allowed."""
        allowed = await self._check_rate_limit()
        if allowed:
            self._total_requests += 1
        else:
            self._blocked_requests += 1
        return allowed

    async def wait(self) -> None:
        """Wait until request is allowed."""
        while not await self.is_allowed():
            await asyncio.sleep(0.01)

    async def _check_rate_limit(self) -> bool:
        """Check if under rate limit."""
        if self._config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._strategy.acquire()  # type: ignore
        elif self._config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._strategy.is_allowed()  # type: ignore
        return True

    async def stats(self) -> dict:
        """Get rate limiter stats."""
        strategy_stats = await self._strategy.stats() if self._strategy else {}
        return {
            **strategy_stats,
            "total_requests": self._total_requests,
            "blocked_requests": self._blocked_requests,
            "strategy": self._config.strategy.value,
        }

    async def reset(self) -> None:
        """Reset rate limiter stats."""
        self._total_requests = 0
        self._blocked_requests = 0


class AsyncRateLimiter:
    """Context manager for rate limiting."""

    def __init__(
        self,
        rate_limiter: RateLimiter,
        delay_on_block: float = 0.1,
    ):
        self._rate_limiter = rate_limiter
        self._delay = delay_on_block
        self._blocked_count = 0

    async def __aenter__(self) -> None:
        """Enter rate-limited context."""
        await self._rate_limiter.wait()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit rate-limited context."""
        pass

    @property
    def blocked_count(self) -> int:
        """Get number of blocked requests."""
        return self._blocked_count

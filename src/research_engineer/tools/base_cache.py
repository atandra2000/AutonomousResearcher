"""Base classes for caching and rate limiting."""

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Generic, TypeVar

T = TypeVar("T")


class CacheEntry:
    """A single cache entry with metadata."""

    def __init__(self, value: T, created_at: float | None = None):
        self.value = value
        self.created_at = created_at or time.time()

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CacheEntry":
        return cls(value=data["value"], created_at=data["created_at"])


class CacheBase(ABC, Generic[T]):
    """Abstract base class for cache implementations."""

    @abstractmethod
    async def get(self, key: str) -> T | None:
        """Get a value from cache."""

    @abstractmethod
    async def set(self, key: str, value: T, ttl: float | None = None) -> None:
        """Set a value in cache."""

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a value from cache."""

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""

    @abstractmethod
    async def stats(self) -> dict:
        """Get cache statistics."""


@dataclass
class TokenBucketConfig:
    """Configuration for token bucket rate limiter."""

    rate: float = 1.0  # tokens per second
    capacity: int = 10  # max tokens in bucket
    refill_interval: float = 0.1  # seconds between refills

    @classmethod
    def from_dict(cls, data: dict) -> "TokenBucketConfig":
        return cls(
            rate=data.get("rate", 1.0),
            capacity=data.get("capacity", 10),
            refill_interval=data.get("refill_interval", 0.1),
        )


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    requests_per_second: float = 1.0
    burst_size: int = 10
    window_seconds: float = 60.0

    @classmethod
    def from_dict(cls, data: dict) -> "RateLimitConfig":
        return cls(
            requests_per_second=data.get("requests_per_second", 1.0),
            burst_size=data.get("burst_size", 10),
            window_seconds=data.get("window_seconds", 60.0),
        )


class SimpleCache(CacheBase[T]):
    """In-memory cache implementation using asyncio locks."""

    def __init__(self, ttl: float | None = None):
        self._cache: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._ttl = ttl

    async def get(self, key: str) -> T | None:
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if self._ttl and entry.age_seconds > self._ttl:
                    del self._cache[key]
                    return None
                return entry.value
            return None

    async def set(self, key: str, value: T, ttl: float | None = None) -> None:
        async with self._lock:
            self._cache[key] = CacheEntry(value)

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()

    async def stats(self) -> dict:
        async with self._lock:
            return {
                "size": len(self._cache),
                "type": "SimpleCache",
                "ttl": self._ttl,
            }


class FileCache(CacheBase[T]):
    """File-based cache using JSON serialization."""

    def __init__(self, path: str, ttl: float | None = None):
        self._path = path
        self._cache: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._ttl = ttl
        self._loaded = False

    async def _load(self) -> None:
        """Load cache from file."""
        import pathlib
        if not self._loaded:
            path = pathlib.Path(self._path)
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                    self._cache = {
                        k: CacheEntry.from_dict(v) for k, v in data.items()
                    }
                except (json.JSONDecodeError, KeyError):
                    self._cache = {}
            self._loaded = True

    async def _save(self) -> None:
        """Save cache to file."""
        import pathlib
        async with self._lock:
            data = {
                k: v.to_dict()
                for k, v in self._cache.items()
            }
            path = pathlib.Path(self._path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))

    async def get(self, key: str) -> T | None:
        await self._load()
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if self._ttl and entry.age_seconds > self._ttl:
                    del self._cache[key]
                    await self._save()
                    return None
                return entry.value
            return None

    async def set(self, key: str, value: T, ttl: float | None = None) -> None:
        await self._load()
        async with self._lock:
            self._cache[key] = CacheEntry(value)
            await self._save()

    async def delete(self, key: str) -> bool:
        await self._load()
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                await self._save()
                return True
            return False

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()
            await self._save()

    async def stats(self) -> dict:
        await self._load()
        async with self._lock:
            return {
                "size": len(self._cache),
                "type": "FileCache",
                "path": self._path,
                "ttl": self._ttl,
            }


class TokenBucketRateLimiter:
    """Token bucket rate limiter implementation."""

    def __init__(self, config: TokenBucketConfig):
        self._config = config
        self._tokens = config.capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()
        self._total_requests = 0
        self._blocked_requests = 0
        self._last_request = 0.0

    async def acquire(self) -> bool:
        """Try to acquire a token. Returns True if allowed."""
        async with self._lock:
            self._refill()
            if self._tokens > 0:
                self._tokens -= 1
                self._total_requests += 1
                self._last_request = time.monotonic()
                return True
            self._blocked_requests += 1
            return False

    async def wait(self) -> None:
        """Wait until a token is available."""
        while True:
            async with self._lock:
                self._refill()
                if self._tokens > 0:
                    self._tokens -= 1
                    self._total_requests += 1
                    self._last_request = time.monotonic()
                    return
            await asyncio.sleep(0.01)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        tokens_to_add = elapsed * self._config.rate
        self._tokens = min(
            self._config.capacity,
            self._tokens + tokens_to_add,
        )
        self._last_refill = now

    async def stats(self) -> dict:
        async with self._lock:
            return {
                "tokens": self._tokens,
                "capacity": self._config.capacity,
                "rate": self._config.rate,
                "total_requests": self._total_requests,
                "blocked_requests": self._blocked_requests,
                "last_request": self._last_request,
            }


class SlidingWindowRateLimiter:
    """Sliding window rate limiter."""

    def __init__(self, requests_per_window: int = 60, window_seconds: float = 60.0):
        self._requests_per_window = requests_per_window
        self._window_seconds = window_seconds
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()
        self._total_requests = 0
        self._blocked_requests = 0

    async def is_allowed(self) -> bool:
        """Check if request is allowed."""
        async with self._lock:
            self._cleanup()
            if len(self._timestamps) < self._requests_per_window:
                self._timestamps.append(time.monotonic())
                self._total_requests += 1
                return True
            self._blocked_requests += 1
            return False

    async def wait(self) -> None:
        """Wait until request is allowed."""
        while not await self.is_allowed():
            await asyncio.sleep(0.01)

    def _cleanup(self) -> None:
        """Remove expired timestamps."""
        now = time.monotonic()
        cutoff = now - self._window_seconds
        self._timestamps = [t for t in self._timestamps if t > cutoff]

    async def stats(self) -> dict:
        async with self._lock:
            self._cleanup()
            return {
                "requests_in_window": len(self._timestamps),
                "limit": self._requests_per_window,
                "window_seconds": self._window_seconds,
                "total_requests": self._total_requests,
                "blocked_requests": self._blocked_requests,
            }


def generate_key(tool_name: str, input_data: dict) -> str:
    """Generate a cache key from tool name and input data."""
    key_data = {
        "tool": tool_name,
        "input": input_data,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()

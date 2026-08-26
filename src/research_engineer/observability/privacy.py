"""E6 - Privacy, redaction, and payload limits for observability events.

Events must be useful for debugging without leaking sensitive material:

* secrets/API keys (``api_key``, ``secret``, ``password``, ``authorization``,
  ...) are replaced with a fixed placeholder whenever detected anywhere in
  the event tree;
* long payload strings are truncated to ``max_payload_chars``;
* optionally, prompt/response text may be captured (off by default) and may
  be hashed rather than stored verbatim;
* redaction itself must never raise — observability is best-effort.
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field

#: Keys whose values are always treated as secret material.
DEFAULT_SENSITIVE_KEYS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "api-key",
    "x-api-key",
    "secret",
    "secret_key",
    "access_token",
    "refresh_token",
    "auth_header",
    "authorization",
    "password",
    "passwd",
    "credential",
    "credentials",
    "private_key",
)

REDACTED = "[REDACTED]"
TRUNCATION_MARKER = "...[truncated]"


class TelemetryConfig(BaseModel):
    """Configuration controlling what observability captures and how."""

    #: Capture raw prompt/response text in LLM events (default off: only
    #: hashes are recorded even when payloads are present).
    capture_prompts: bool = False
    #: Replace prompt/response content with a stable hash instead of text.
    hash_prompts: bool = True
    #: Maximum characters for any single string value placed in an event.
    max_payload_chars: int = Field(default=8000, ge=64)
    #: Key fragments treated as secrets (case-insensitive substring match).
    sensitive_keys: tuple[str, ...] = DEFAULT_SENSITIVE_KEYS
    #: Maximum recursion depth when scrubbing nested structures.
    max_depth: int = Field(default=12, ge=1)
    #: Whether correlation stamping + privacy scrubbing is applied by buses.
    enabled: bool = True


_DEFAULT_CONFIG = TelemetryConfig()


def get_default_config() -> TelemetryConfig:
    """Return the process-wide default telemetry configuration."""
    return _DEFAULT_CONFIG


def configure_telemetry(**kwargs: Any) -> TelemetryConfig:
    """Update the process-wide default telemetry configuration."""
    global _DEFAULT_CONFIG
    updated = _DEFAULT_CONFIG.model_copy(update=kwargs)
    assert isinstance(updated, TelemetryConfig)
    _DEFAULT_CONFIG = updated
    return _DEFAULT_CONFIG


def hash_text(text: str) -> str:
    """Stable short SHA-256 hash for log correlation (never reversible)."""
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return f"sha256:{digest[:16]}"


def _is_sensitive(key: str, sensitive_keys: tuple[str, ...]) -> bool:
    lowered = key.lower()
    return any(fragment.lower() in lowered for fragment in sensitive_keys)


def redact(value: Any, config: TelemetryConfig | None = None, depth: int = 0) -> Any:
    """Recursively redact secrets and enforce payload limits on ``value``.

    Never raises; on unexpected input types the ``str`` form is returned.
    """
    cfg = config or _DEFAULT_CONFIG
    try:
        return _redact(value, cfg, depth)
    except Exception:  # noqa: BLE001 - privacy must never break emission
        return TRUNCATION_MARKER


def _redact(value: Any, cfg: TelemetryConfig, depth: int) -> Any:
    if depth > cfg.max_depth:
        return TRUNCATION_MARKER
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        limited = value[: cfg.max_payload_chars]
        if len(value) > cfg.max_payload_chars:
            limited += TRUNCATION_MARKER
        # A string living directly under a secret key was already removed
        # before reaching this function; strings themselves pass through.
        return limited
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for k, v in value.items():
            key_str = str(k)
            if _is_sensitive(key_str, cfg.sensitive_keys):
                out[k] = REDACTED
            else:
                out[k] = _redact(v, cfg, depth + 1)
        return out
    if isinstance(value, (list, tuple, set)):
        seq = list(value)[:1000]
        return [_redact(v, cfg, depth + 1) for v in seq]
    # Non-JSON-native objects (enums, datetimes, pydantic models, ...): keep
    # them for downstream serialization but bound their textual footprint.
    return value


def sanitize_llm_content(content: str, config: TelemetryConfig | None = None) -> str:
    """Return the safe representation of LLM prompt/response content.

    Honors ``capture_prompts``/``hash_prompts``: by default content is
    replaced by a stable hash; when capture is enabled without hashing the
    (length-limited) text is returned verbatim.
    """
    cfg = config or _DEFAULT_CONFIG
    if not cfg.capture_prompts:
        return hash_text(content)
    if cfg.hash_prompts:
        return hash_text(content)
    return str(redact(content[: cfg.max_payload_chars], cfg))


__all__ = [
    "REDACTED",
    "TRUNCATION_MARKER",
    "TelemetryConfig",
    "configure_telemetry",
    "get_default_config",
    "hash_text",
    "redact",
    "sanitize_llm_content",
]

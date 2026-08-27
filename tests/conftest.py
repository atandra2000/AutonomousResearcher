"""Shared pytest fixtures/markers for the research engineer test suite.

P4 CI stabilization:

* ``network`` marker: tests that genuinely require live internet (e.g.
  arXiv API) must be marked ``@pytest.mark.network``; they are skipped
  automatically when connectivity is unavailable so the suite stays
  deterministic in offline CI environments.
* Per-test timeout via ``pytest-timeout`` (configured in pyproject) so a
  single hung test can never stall the whole CI run indefinitely.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator

import pytest

#: Hosts a "network available" probe connects to (short timeout).
_PROBE_HOSTS: tuple[tuple[str, int], ...] = (
    ("arxiv.org", 443),
    ("export.arxiv.org", 443),
)


def _probe_network(seconds: float = 3.0) -> bool:
    """True when at least one probe host accepts a TCP connection."""
    for host, port in _PROBE_HOSTS:
        try:
            with socket.create_connection((host, port), timeout=seconds):
                return True
        except OSError:
            continue
    return False


@pytest.fixture(scope="session")
def network_available() -> bool:
    """Session-level connectivity probe (cached)."""
    return _probe_network()


@pytest.fixture(autouse=True)
def _skip_network_tests(
    request: pytest.FixtureRequest, network_available: bool
) -> Iterator[None]:
    """Skip live-network tests unless connectivity is actually available."""
    if request.node.get_closest_marker("network") and not network_available:
        pytest.skip("live network unavailable (arXiv unreachable)")
    yield

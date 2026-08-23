"""Regression tests for production-readiness fixes.

Covers:
- LLM-wired code generation (proposed content, target resolution)
- Real unified diffs from proposed content (patch generation)
- IDF-weighted paper search relevance
- LLM degradation warnings from resolve_llm
- LLM resilience (retry, backoff, empty-content guard)
- Validation gate: syntax checks at generation and application time
"""

import logging

import pytest

from research_engineer.agents._llm_support import resolve_llm
from research_engineer.llm.base import (
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMRole,
    ProviderError,
)
from research_engineer.llm.resilience import complete_with_retry
from research_engineer.models.coding import (
    ChangeType,
    CodeChange,
    GeneratedPatch,
)
from research_engineer.models.literature import SearchResult, SearchSource
from research_engineer.tools.code_generation import CodeGenerationTool
from research_engineer.tools.paper_search import PaperSearchTool
from research_engineer.tools.patch_application import (
    PatchApplicationInput,
    PatchApplicationTool,
)
from research_engineer.tools.patch_generation import (
    PatchGenerationInput,
    PatchGenerationTool,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _ScriptedProvider(LLMProvider):
    """Returns queued responses in order."""

    name = "scripted"

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        content = self.responses.pop(0) if self.responses else ""
        return LLMResponse(content=content, model="fake", provider=self.name)


def _make_change(repo_path: str) -> CodeChange:
    return CodeChange(
        file_path="src/implementation.py",  # intentionally nonexistent
        change_type=ChangeType.MODIFICATION,
        description="Add top_k_tokens",
        reason="test",
    )


# ---------------------------------------------------------------------------
# Code generation: LLM refinement + target resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refinement_sets_proposed_content(tmp_path):
    provider = _ScriptedProvider(["def top_k_tokens(x):\n    return x\n"])
    tool = CodeGenerationTool(llm=provider)
    (tmp_path / "src").mkdir()
    # The rule-based generator targets src/implementation.py by default;
    # create it so refinement reads real context instead of resolving.
    (tmp_path / "src" / "implementation.py").write_text("# placeholder\n")
    out = await tool.execute(
        _input_for(tmp_path, [], "Update the implementation module")
    )
    assert out.changes, "rule-based generation should produce a change"
    assert out.changes[0].proposed_content == (
        "def top_k_tokens(x):\n    return x\n"
    )


@pytest.mark.asyncio
async def test_target_resolution_repoints_missing_file(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "generate.py").write_text("x = 1\n")
    # First call: target resolution -> src/generate.py; second: file content.
    provider = _ScriptedProvider(["src/generate.py", "y = 2\n"])
    tool = CodeGenerationTool(llm=provider)
    change = _make_change(str(tmp_path))
    out = await tool.execute(_input_for(tmp_path, [change], "Add a function"))
    assert out.changes[0].file_path == "src/generate.py"
    assert out.changes[0].proposed_content == "y = 2\n"


@pytest.mark.asyncio
async def test_unresolvable_target_is_skipped(tmp_path, caplog):
    (tmp_path / "src").mkdir()
    provider = _ScriptedProvider(["does/not/exist.py"])  # not in candidates
    tool = CodeGenerationTool(llm=provider)
    change = _make_change(str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="research_engineer.tools.code_generation"):
        out = await tool.execute(_input_for(tmp_path, [change], "task"))
    assert out.changes[0].file_path == "src/implementation.py"


def test_strip_code_fences():
    from research_engineer.tools.code_generation import _strip_code_fences

    assert _strip_code_fences("```python\na=1\n```") == "a=1\n"
    assert _strip_code_fences("a=1") == "a=1"
    assert _strip_code_fences("```\na=1") == "a=1\n"


def _input_for(tmp_path, changes, task: str):  # noqa: ARG001
    from research_engineer.tools.code_generation import CodeGenerationInput

    return CodeGenerationInput(
        repo_path=str(tmp_path),
        task_description=task,
        constraints=[],
    )


# ---------------------------------------------------------------------------
# Patch generation: real diffs from proposed content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_modification_diff_from_proposed_content(tmp_path):
    target = tmp_path / "src" / "generate.py"
    target.parent.mkdir(parents=True)
    target.write_text("a = 1\n")
    change = CodeChange(
        file_path="src/generate.py",
        change_type=ChangeType.MODIFICATION,
        description="bump",
        reason="test",
        proposed_content="a = 2\n",
    )
    out = await PatchGenerationTool().execute(
        PatchGenerationInput(changes=[change], repo_path=str(tmp_path))
    )
    diff = out.patches[0].diff
    assert "-a = 1" in diff
    assert "+a = 2" in diff
    assert "[Added:" not in diff  # no placeholder metadata


@pytest.mark.asyncio
async def test_new_file_diff_from_proposed_content(tmp_path):
    change = CodeChange(
        file_path="new_module.py",
        change_type=ChangeType.NEW_FILE,
        description="new",
        reason="test",
        proposed_content="def f():\n    return 42\n",
    )
    out = await PatchGenerationTool().execute(
        PatchGenerationInput(changes=[change], repo_path=str(tmp_path))
    )
    diff = out.patches[0].diff
    assert "--- /dev/null" in diff
    assert "+def f():" in diff


# ---------------------------------------------------------------------------
# Paper search relevance
# ---------------------------------------------------------------------------


def _result(pid: str, title: str, abstract: str = "") -> SearchResult:
    return SearchResult(
        paper_id=pid,
        title=title,
        abstract=abstract,
        source=SearchSource.ARXIV,
    )


def test_relevance_prefers_topical_coverage_over_single_word():
    tool = PaperSearchTool()
    query = "sliding window attention"
    results = [
        _result("fk", "Sliding Friction in the Frenkel-Kontorova Model"),
        _result(
            "swa",
            "Sliding Window Attention for Transformers",
            abstract="We study efficient attention with sliding windows.",
        ),
    ]
    ranked = tool._rank_results(results, query, sort="relevance")
    assert ranked[0].paper_id == "swa"


def test_relevance_exact_phrase_bonus():
    tool = PaperSearchTool()
    results = [
        _result("a", "Attention mechanisms: a survey"),
        _result("b", "On Attention"),
    ]
    ranked = tool._rank_results(results, "attention mechanisms", sort="relevance")
    assert ranked[0].relevance_score >= ranked[1].relevance_score


# ---------------------------------------------------------------------------
# resolve_llm degradation warnings
# ---------------------------------------------------------------------------


class _ExplodingRouter:
    def for_agent(self, name: str) -> None:
        raise RuntimeError("boom")


def test_resolve_llm_warns_on_router_failure(caplog):
    with caplog.at_level(logging.WARNING, logger="research_engineer.agents._llm_support"):
        provider = resolve_llm("SomeAgent", None, router=_ExplodingRouter())
    assert provider is None
    assert any("rule-based mode" in r.message for r in caplog.records)


class _OkRouter:
    def for_agent(self, name: str):
        return object()


def test_resolve_llm_no_warning_on_success(caplog):
    with caplog.at_level(logging.WARNING, logger="research_engineer.agents._llm_support"):
        provider = resolve_llm("SomeAgent", None, router=_OkRouter())
    assert provider is not None
    assert not caplog.records


# ---------------------------------------------------------------------------
# LLM resilience: retry, backoff, empty-content guard
# ---------------------------------------------------------------------------


class _FlakyProvider(LLMProvider):
    name = "flaky"

    def __init__(self, failures: int, content: str = "ok") -> None:
        self.failures = failures
        self.content = content

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if self.failures > 0:
            self.failures -= 1
            raise ProviderError("Ollama Cloud request failed: timeout")
        return LLMResponse(content=self.content, model="m", provider=self.name)


@pytest.mark.asyncio
async def test_retry_succeeds_after_transient_failures():
    provider = _FlakyProvider(failures=2)
    resp = await complete_with_retry(
        provider,
        LLMRequest(messages=[LLMMessage(role=LLMRole.USER, content="hi")]),
        base_delay=0.0,
    )
    assert resp.content == "ok"


@pytest.mark.asyncio
async def test_retry_exhaustion_raises_provider_error():
    provider = _FlakyProvider(failures=99)
    with pytest.raises(ProviderError):
        await complete_with_retry(
            provider,
            LLMRequest(messages=[LLMMessage(role=LLMRole.USER, content="hi")]),
            max_attempts=2,
            base_delay=0.0,
        )


class _PermanentProvider(LLMProvider):
    name = "permanent"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise ProviderError("Ollama Cloud returned HTTP 410: model retired")


@pytest.mark.asyncio
async def test_permanent_error_fails_fast():
    with pytest.raises(ProviderError, match="410"):
        await complete_with_retry(
            _PermanentProvider(),
            LLMRequest(messages=[LLMMessage(role=LLMRole.USER, content="x")]),
            max_attempts=3,
            base_delay=0.0,
        )


class _TruncatingProvider(LLMProvider):
    """Emits empty content at finish_reason=length until budget doubled."""

    name = "truncating"

    def __init__(self) -> None:
        self.seen_max_tokens: list[int | None] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.seen_max_tokens.append(request.max_tokens)
        if request.max_tokens and request.max_tokens >= 128:
            return LLMResponse(
                content="recovered", model="m", provider=self.name
            )
        return LLMResponse(
            content="", model="m", provider=self.name, finish_reason="length"
        )


@pytest.mark.asyncio
async def test_empty_content_escalates_max_tokens_once():
    provider = _TruncatingProvider()
    resp = await complete_with_retry(
        provider,
        LLMRequest(
            messages=[LLMMessage(role=LLMRole.USER, content="hi")],
            max_tokens=64,
        ),
        base_delay=0.0,
    )
    assert resp.content == "recovered"
    assert provider.seen_max_tokens == [64, 128]


# ---------------------------------------------------------------------------
# Validation gate: syntax checks at generation and application time
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generation_rejects_non_parsing_proposal(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "implementation.py").write_text("# ok\n")
    provider = _ScriptedProvider(["def broken( :\n    pass"])  # syntax error
    tool = CodeGenerationTool(llm=provider)
    out = await tool.execute(
        _input_for(tmp_path, [], "Update the implementation module")
    )
    assert out.changes[0].proposed_content is None


@pytest.mark.asyncio
async def test_apply_rejects_invalid_python_new_file(tmp_path):
    patch = GeneratedPatch(
        patch_id="p1",
        file_path="bad_module.py",
        change_type=ChangeType.NEW_FILE,
        diff="--- /dev/null\n+++ b/bad_module.py\n@@ -0,0 +1,1 @@\n+def broken(:\n",
        explanation="",
        reason="test",
        approval_required=False,
    )
    out = await PatchApplicationTool().execute(
        PatchApplicationInput(
            patches=[patch], repo_path=str(tmp_path),
            dry_run=False, require_approval=False, approved=True,
        )
    )
    assert patch.patch_id in out.failed_patches
    assert not (tmp_path / "bad_module.py").exists()


# ---------------------------------------------------------------------------
# Hunk-based unified diff application
# ---------------------------------------------------------------------------


def _apply(content: str, diff: str) -> str:
    from research_engineer.tools.patch_application import PatchApplicationTool

    return PatchApplicationTool()._apply_diff_to_content(content, diff)


def test_applies_mid_file_insertion_at_correct_position():
    content = "line1\nline2\nline5\n"
    diff = (
        "--- a/f.py\n+++ b/f.py\n"
        "@@ -1,3 +1,4 @@\n"
        " line1\n"
        " line2\n"
        "+line3\n"
        "+line4\n"
        " line5\n"
    )
    assert _apply(content, diff).splitlines() == [
        "line1", "line2", "line3", "line4", "line5",
    ]


def test_applies_replacement_hunk():
    content = "a = 1\nb = 2\nc = 3\n"
    diff = (
        "@@ -1,3 +1,3 @@\n"
        " a = 1\n"
        "-b = 2\n"
        "+b = 20\n"
        " c = 3\n"
    )
    assert "b = 20" in _apply(content, diff)
    assert "b = 2\n" not in _apply(content, diff)


def test_refuses_diff_with_mismatched_context():
    content = "actual\ncontent\n"
    diff = "@@ -1,2 +1,2 @@\n wrong\n-context\n+new\n"
    assert _apply(content, diff) == content

"""Tests for the TerminalTool (Phase 11)."""

from __future__ import annotations

import asyncio

import pytest

from research_engineer.tools.terminal import (
    ALLOWED_COMMAND_PREFIXES,
    TerminalInput,
    TerminalOutput,
    TerminalTool,
)


@pytest.fixture
def repo(tmp_path):
    """Create a small fake repo with a file and git init."""
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    f = tmp_path / "src" / "pkg" / "foo.py"
    f.write_text("def foo():\n    return 1\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# repo\n", encoding="utf-8")
    # git init so git_status / git_diff work
    import subprocess

    subprocess.run(
        ["git", "init"], cwd=str(tmp_path), capture_output=True, check=False
    )
    subprocess.run(
        ["git", "config", "user.email", "t@t.com"],
        cwd=str(tmp_path),
        capture_output=True,
        check=False,
    )
    subprocess.run(
        ["git", "config", "user.name", "test"],
        cwd=str(tmp_path),
        capture_output=True,
        check=False,
    )
    subprocess.run(
        ["git", "add", "."], cwd=str(tmp_path), capture_output=True, check=False
    )
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=str(tmp_path),
        capture_output=True,
        check=False,
    )
    return tmp_path


class TestTerminalToolValidation:
    @pytest.mark.asyncio
    async def test_validate_rejects_missing_repo(self):
        tool = TerminalTool()
        inp = TerminalInput(operation="git_status", repo_path="/nonexistent")
        assert not await tool.validate(inp)

    @pytest.mark.asyncio
    async def test_validate_rejects_empty_operation(self, tmp_path):
        tool = TerminalTool()
        inp = TerminalInput(operation="", repo_path=str(tmp_path))
        assert not await tool.validate(inp)


class TestRunCommand:
    @pytest.mark.asyncio
    async def test_run_command_echo(self, repo):
        tool = TerminalTool()
        out = await tool.execute(
            TerminalInput(
                operation="run_command",
                repo_path=str(repo),
                command=["echo", "hello"],
            )
        )
        assert out.success
        assert "hello" in out.stdout

    @pytest.mark.asyncio
    async def test_run_command_dry_run(self, repo):
        tool = TerminalTool()
        out = await tool.execute(
            TerminalInput(
                operation="run_command",
                repo_path=str(repo),
                command=["echo", "x"],
                dry_run=True,
            )
        )
        assert out.success
        assert "echo x" in out.stdout

    @pytest.mark.asyncio
    async def test_run_command_disallowed(self, repo):
        tool = TerminalTool()
        with pytest.raises(Exception):
            await tool.execute(
                TerminalInput(
                    operation="run_command",
                    repo_path=str(repo),
                    command=["rm", "-rf", "/"],
                )
            )

    @pytest.mark.asyncio
    async def test_run_command_failure_exit_code(self, repo):
        tool = TerminalTool()
        out = await tool.execute(
            TerminalInput(
                operation="run_command",
                repo_path=str(repo),
                command=["python3", "-c", "import sys; sys.exit(2)"],
            )
        )
        assert not out.success
        assert out.exit_code == 2


class TestReadFile:
    @pytest.mark.asyncio
    async def test_read_file_success(self, repo):
        tool = TerminalTool()
        out = await tool.execute(
            TerminalInput(
                operation="read_file",
                repo_path=str(repo),
                file_path="src/pkg/foo.py",
            )
        )
        assert out.success
        assert "def foo" in (out.content or "")

    @pytest.mark.asyncio
    async def test_read_file_missing(self, repo):
        tool = TerminalTool()
        out = await tool.execute(
            TerminalInput(
                operation="read_file",
                repo_path=str(repo),
                file_path="nope.py",
            )
        )
        assert not out.success
        assert "not found" in (out.error or "").lower()

    @pytest.mark.asyncio
    async def test_read_file_truncation(self, repo):
        tool = TerminalTool()
        out = await tool.execute(
            TerminalInput(
                operation="read_file",
                repo_path=str(repo),
                file_path="src/pkg/foo.py",
                max_bytes=5,
            )
        )
        assert out.success
        assert "truncated" in (out.content or "")


class TestWriteFile:
    @pytest.mark.asyncio
    async def test_write_file_creates_parents(self, repo):
        tool = TerminalTool()
        out = await tool.execute(
            TerminalInput(
                operation="write_file",
                repo_path=str(repo),
                file_path="src/new/dir/bar.py",
                content="x = 1\n",
            )
        )
        assert out.success
        assert (repo / "src" / "new" / "dir" / "bar.py").exists()

    @pytest.mark.asyncio
    async def test_write_file_overwrites(self, repo):
        tool = TerminalTool()
        out = await tool.execute(
            TerminalInput(
                operation="write_file",
                repo_path=str(repo),
                file_path="src/pkg/foo.py",
                content="def bar():\n    return 2\n",
            )
        )
        assert out.success
        assert "def bar" in (repo / "src" / "pkg" / "foo.py").read_text()


class TestSearchCode:
    @pytest.mark.asyncio
    async def test_search_code_finds_match(self, repo):
        tool = TerminalTool()
        out = await tool.execute(
            TerminalInput(
                operation="search_code",
                repo_path=str(repo),
                pattern="def foo",
            )
        )
        assert out.success
        assert len(out.matches) >= 1
        assert any("foo" in m["text"] for m in out.matches)

    @pytest.mark.asyncio
    async def test_search_code_no_matches(self, repo):
        tool = TerminalTool()
        out = await tool.execute(
            TerminalInput(
                operation="search_code",
                repo_path=str(repo),
                pattern="zzznotfoundzzz",
            )
        )
        assert out.success
        assert out.matches == []

    @pytest.mark.asyncio
    async def test_search_code_invalid_regex(self, repo):
        tool = TerminalTool()
        out = await tool.execute(
            TerminalInput(
                operation="search_code",
                repo_path=str(repo),
                pattern="[unclosed",
            )
        )
        assert not out.success
        assert "regex" in (out.error or "").lower()


class TestGitStatus:
    @pytest.mark.asyncio
    async def test_git_status_clean(self, repo):
        tool = TerminalTool()
        out = await tool.execute(
            TerminalInput(operation="git_status", repo_path=str(repo))
        )
        assert out.success
        # Clean repo -> empty porcelain output
        assert out.stdout.strip() == ""


class TestGitDiff:
    @pytest.mark.asyncio
    async def test_git_diff_shows_change(self, repo):
        # Modify a file after commit
        (repo / "src" / "pkg" / "foo.py").write_text("def foo():\n    return 2\n")
        tool = TerminalTool()
        out = await tool.execute(
            TerminalInput(operation="git_diff", repo_path=str(repo))
        )
        assert out.success
        assert "return 2" in (out.content or out.stdout)


class TestApplyPatch:
    @pytest.mark.asyncio
    async def test_apply_patch_modifies_file(self, repo):
        tool = TerminalTool()
        diff = (
            "diff --git a/src/pkg/foo.py b/src/pkg/foo.py\n"
            "--- a/src/pkg/foo.py\n"
            "+++ b/src/pkg/foo.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def foo():\n"
            "-    return 1\n"
            "+    return 42\n"
        )
        out = await tool.execute(
            TerminalInput(
                operation="apply_patch",
                repo_path=str(repo),
                patch_diff=diff,
            )
        )
        assert out.success
        assert "return 42" in (repo / "src" / "pkg" / "foo.py").read_text()

    @pytest.mark.asyncio
    async def test_apply_patch_bad_diff_fails(self, repo):
        tool = TerminalTool()
        diff = (
            "diff --git a/missing.py b/missing.py\n"
            "--- a/missing.py\n"
            "+++ b/missing.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-old\n"
            "+new\n"
        )
        out = await tool.execute(
            TerminalInput(
                operation="apply_patch",
                repo_path=str(repo),
                patch_diff=diff,
            )
        )
        assert not out.success


class TestUnknownOperation:
    @pytest.mark.asyncio
    async def test_unknown_operation_raises(self, repo):
        tool = TerminalTool()
        with pytest.raises(Exception):
            await tool.execute(
                TerminalInput(
                    operation="bogus", repo_path=str(repo)
                )
            )


class TestAllowlist:
    def test_allowed_prefixes_contains_python(self):
        assert "python" in ALLOWED_COMMAND_PREFIXES
        assert "pytest" in ALLOWED_COMMAND_PREFIXES

    def test_allowed_prefixes_excludes_rm(self):
        assert "rm" not in ALLOWED_COMMAND_PREFIXES
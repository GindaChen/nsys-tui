"""Tests for root_cause_store: parse, validate, submit, list."""

import pytest


def test_parse_entry_valid():
    """Parse a well-formed root cause markdown entry."""
    from nsys_ai.root_cause_store import parse_entry

    md = """---
name: Test Pattern
severity: warning
tags: [gpu, memory]
detection_skill: gpu_idle_gaps
---

## Symptom
GPU is idle for 10ms between kernels.

## Why It Happens
CPU is too slow.

## How to Fix
Use CUDA graphs.
"""
    entry = parse_entry(md, source="test")
    assert entry.name == "Test Pattern"
    assert entry.severity == "warning"
    assert entry.tags == ["gpu", "memory"]
    assert entry.detection_skill == "gpu_idle_gaps"
    assert "GPU is idle" in entry.symptom
    assert "CUDA graphs" in entry.fix
    assert "CPU is too slow" in entry.mechanism


def test_parse_entry_missing_frontmatter():
    """Parse entry without YAML frontmatter."""
    from nsys_ai.root_cause_store import parse_entry

    md = """## Symptom
Something is slow.

## How to Fix
Make it faster.
"""
    entry = parse_entry(md)
    assert entry.name == "Untitled"
    assert "Something is slow" in entry.symptom


def test_validate_entry_errors():
    """Validate catches missing required fields."""
    from nsys_ai.root_cause_store import RootCauseEntry, validate_entry

    entry = RootCauseEntry(name="", severity="invalid")
    errors = validate_entry(entry)
    assert len(errors) >= 3  # missing name, invalid severity, missing symptom, missing fix


def test_validate_entry_passes():
    """Valid entry passes validation."""
    from nsys_ai.root_cause_store import RootCauseEntry, validate_entry

    entry = RootCauseEntry(
        name="Good Pattern",
        severity="warning",
        symptom="GPU is idle",
        fix="Use CUDA graphs",
    )
    errors = validate_entry(entry)
    assert errors == []


def test_submit_and_load(tmp_path):
    """Submit a root cause and load it back from the directory."""
    from nsys_ai.root_cause_store import submit_entry

    src = tmp_path / "input.md"
    src.write_text(
        """---
name: GIL Contention
severity: warning
tags: [python, cpu]
detection_skill: thread_utilization
---

## Symptom
Single thread at >90% CPU.

## How to Fix
Use num_workers > 0.
"""
    )

    dest_dir = tmp_path / "submitted"
    entry, errors = submit_entry(str(src), dest_dir=str(dest_dir))

    assert errors == [], f"Unexpected errors: {errors}"
    assert entry.name == "GIL Contention"
    assert (dest_dir / "gil_contention.md").exists()

    # Load it back
    from nsys_ai.root_cause_store import _load_dir_entries

    loaded = _load_dir_entries(str(dest_dir), source="user")
    assert len(loaded) == 1
    assert loaded[0].name == "GIL Contention"


def test_submit_validation_failure(tmp_path):
    """Submit rejects an entry missing required sections."""
    from nsys_ai.root_cause_store import submit_entry

    src = tmp_path / "bad.md"
    src.write_text("# No frontmatter, no sections\n\nJust text.\n")

    entry, errors = submit_entry(str(src), dest_dir=str(tmp_path / "dest"))
    assert len(errors) > 0
    assert any("name" in e.lower() or "symptom" in e.lower() for e in errors)


def test_parse_book_md():
    """Parse the built-in book.md and verify entries are extracted."""
    from nsys_ai.root_cause_store import _find_book_md, _parse_book_md

    book = _find_book_md()
    if book is None:
        pytest.skip("book.md not found in repo")

    entries = _parse_book_md(book)
    assert len(entries) >= 10  # book.md has 10 root causes
    names = [e.name for e in entries]
    assert "GPU Bubbles (Pipeline Stalls)" in names
    assert "Excessive Synchronization" in names


def test_list_entries_includes_builtin():
    """list_entries should include builtin entries from book.md."""
    from nsys_ai.root_cause_store import _find_book_md, list_entries

    if _find_book_md() is None:
        pytest.skip("book.md not found")

    entries = list_entries(root_causes_dir="/nonexistent")
    assert len(entries) >= 10
    assert all(e.source == "builtin" for e in entries)


def test_to_summary_row():
    """to_summary_row returns the expected dict keys."""
    from nsys_ai.root_cause_store import RootCauseEntry

    entry = RootCauseEntry(
        name="Test",
        severity="info",
        tags=["a", "b"],
        detection_skill="top_kernels",
    )
    row = entry.to_summary_row()
    assert row["name"] == "Test"
    assert row["tags"] == "a, b"
    assert row["detection_skill"] == "top_kernels"

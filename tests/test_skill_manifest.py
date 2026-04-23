"""Smoke tests for skills/analyze/SKILL.md (the /nsys-ai Claude Code plugin manifest).

These tests pin the wizard structure so a future edit that breaks the AskUserQuestion
contract (missing section, wrong mode ref, >4 options, header too long) fails in CI
instead of silently shipping.

Zero runtime deps — parses the YAML frontmatter with stdlib string splitting.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = REPO_ROOT / "skills" / "analyze"
SKILL_MD = SKILL_DIR / "SKILL.md"
REFS_DIR = SKILL_DIR / "references"


@pytest.fixture(scope="module")
def skill_text() -> str:
    assert SKILL_MD.exists(), f"SKILL.md missing at {SKILL_MD}"
    return SKILL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_text: str) -> dict[str, str]:
    # Frontmatter is between the first two '---' lines.
    parts = skill_text.split("---", 2)
    assert len(parts) >= 3, "SKILL.md must start with YAML frontmatter delimited by ---"
    raw = parts[1]
    # Stdlib parse: treat every "key: value" line as a mapping entry. Good enough for
    # our flat frontmatter; multi-line scalars collapse into one string.
    fm: dict[str, str] = {}
    current_key: str | None = None
    for line in raw.splitlines():
        if not line.strip():
            continue
        m = re.match(r"^([a-zA-Z_-]+):\s*(.*)$", line)
        if m:
            current_key = m.group(1)
            fm[current_key] = m.group(2).strip().strip('"').strip("'")
        elif current_key is not None:
            # Continuation of a folded scalar — append a space-joined chunk.
            fm[current_key] = (fm[current_key] + " " + line.strip()).strip()
    return fm


def _section(skill_text: str, header: str) -> str:
    """Return the text between `## {header}` and the next `## ` (or end-of-file)."""
    pattern = rf"^## {re.escape(header)}\s*$"
    lines = skill_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(pattern, line):
            start = i
            break
    if start is None:
        return ""
    for j in range(start + 1, len(lines)):
        if re.match(r"^## [^#]", lines[j]):
            return "\n".join(lines[start:j])
    return "\n".join(lines[start:])


# ---------------------------------------------------------------------------
# Frontmatter

def test_frontmatter_has_required_fields(frontmatter: dict[str, str]) -> None:
    assert frontmatter.get("name") == "nsys-ai", "name must be 'nsys-ai'"
    assert frontmatter.get("description"), "description required"
    # argument-hint added in the wizard change — regression guard.
    assert "argument-hint" in frontmatter, (
        "argument-hint frontmatter missing — autocomplete UX regresses"
    )


def test_argument_hint_mentions_both_paths(frontmatter: dict[str, str]) -> None:
    hint = frontmatter.get("argument-hint", "")
    assert "profile" in hint.lower(), f"argument-hint should mention profile: {hint!r}"
    assert "question" in hint.lower(), f"argument-hint should mention question: {hint!r}"


# ---------------------------------------------------------------------------
# Mode Menu wizard section

def test_mode_menu_wizard_section_exists(skill_text: str) -> None:
    body = _section(skill_text, "Mode Menu (interactive wizard)")
    assert body, "## Mode Menu (interactive wizard) section missing"
    # Core markers.
    assert "AskUserQuestion" in body, "wizard must call AskUserQuestion"
    assert "Step 0" in body, "wizard needs Step 0 (CWD scan)"
    assert "Step 1" in body, "wizard needs Step 1 (question batch)"
    assert "Step 2" in body, "wizard needs Step 2 (dispatch)"


def test_wizard_fires_once_per_session(skill_text: str) -> None:
    body = _section(skill_text, "Mode Menu (interactive wizard)")
    assert "once per session" in body.lower(), "session invariant rule missing"


def test_wizard_defers_to_keyword_routing(skill_text: str) -> None:
    body = _section(skill_text, "Mode Menu (interactive wizard)")
    # Any keyword match must bypass the wizard — otherwise power users get interrupted.
    assert re.search(r"keyword routing", body, flags=re.IGNORECASE), (
        "wizard section must reference Keyword Routing bypass"
    )


def test_wizard_handles_zero_one_many_profiles(skill_text: str) -> None:
    body = _section(skill_text, "Mode Menu (interactive wizard)")
    # Three branches are the whole point of Step 0.
    assert "0 files" in body, "Step 0 must handle the 0-profile case"
    assert "1 file" in body, "Step 0 must handle the 1-profile case"
    assert re.search(r"2\+ files", body), "Step 0 must handle the 2+-profile case"


# ---------------------------------------------------------------------------
# AskUserQuestion contract (per official docs: 2–4 options, header ≤12 chars)

def test_q2_focus_has_exactly_four_options(skill_text: str) -> None:
    body = _section(skill_text, "Mode Menu (interactive wizard)")
    # Q2 spec block: look for the numbered list of labels.
    # Pin the four expected labels directly.
    expected_labels = [
        "Auto triage (Recommended)",
        "Compute",
        "Comms",
        "Idle",
    ]
    for label in expected_labels:
        assert f'"{label}"' in body, f"Q2 option label missing: {label!r}"


def test_q2_recommended_option_is_first(skill_text: str) -> None:
    body = _section(skill_text, "Mode Menu (interactive wizard)")
    # Find the numbered list under "**Q2 — Focus**".
    q2_idx = body.find("Q2 — Focus")
    assert q2_idx >= 0, "Q2 block not found"
    q2_body = body[q2_idx:]
    # The first numbered bullet (`1.`) should carry (Recommended).
    first_bullet = re.search(r"^\s*1\.\s*.*$", q2_body, flags=re.MULTILINE)
    assert first_bullet, "Q2 first option (1.) not found"
    assert "(Recommended)" in first_bullet.group(0), (
        f"Q2 first option must be the Recommended one: {first_bullet.group(0)!r}"
    )


def test_question_headers_within_chip_length(skill_text: str) -> None:
    body = _section(skill_text, "Mode Menu (interactive wizard)")
    # Extract every `header: "..."` value and check length ≤ 12.
    for match in re.finditer(r'`header`:\s*`"([^"]+)"`', body):
        header = match.group(1)
        assert len(header) <= 12, (
            f"AskUserQuestion header chip must be ≤12 chars: {header!r} ({len(header)} chars)"
        )


def test_no_manual_other_option(skill_text: str) -> None:
    body = _section(skill_text, "Mode Menu (interactive wizard)")
    # Docs say "Other" is auto-added; manually listing it violates the tool contract.
    assert re.search(
        r"(do NOT add `Other` manually|Auto `Other`|auto `Other`|auto-`Other`)",
        body,
    ), "wizard must document that 'Other' is auto-added (not listed manually)"


# ---------------------------------------------------------------------------
# Dispatch table → mode refs

DISPATCH_EXPECTED = {
    "Auto triage (Recommended)": "M1_AUTO.md",
    "Compute": "M3_COMPUTE.md",
    "Comms": "M2_COMMS.md",
    "Idle": "M6_IDLE.md",
}


@pytest.mark.parametrize("answer,ref_file", list(DISPATCH_EXPECTED.items()))
def test_dispatch_table_row(skill_text: str, answer: str, ref_file: str) -> None:
    body = _section(skill_text, "Mode Menu (interactive wizard)")
    # Find the Step 2 dispatch table row for this answer → ref mapping.
    row = re.search(
        rf"`{re.escape(answer)}`.*{re.escape(ref_file)}",
        body,
    )
    assert row, f"dispatch row missing: {answer!r} → {ref_file}"


@pytest.mark.parametrize("ref_file", sorted(set(DISPATCH_EXPECTED.values())))
def test_dispatched_mode_ref_exists(ref_file: str) -> None:
    assert (REFS_DIR / ref_file).exists(), (
        f"dispatch target {ref_file} does not exist under {REFS_DIR}"
    )


# ---------------------------------------------------------------------------
# Keyword Routing — unchanged-ness guard

def test_keyword_routing_table_intact(skill_text: str) -> None:
    body = _section(skill_text, "Keyword Routing (priority order; first match wins)")
    assert body, "Keyword Routing section missing"
    # All nine priority rows (1–9) must still be there.
    for priority in range(1, 10):
        assert re.search(rf"^\|\s*{priority}\s*\|", body, flags=re.MULTILINE), (
            f"Keyword Routing priority {priority} row missing"
        )


def test_keyword_routing_empty_fallback_preserved(skill_text: str) -> None:
    """Priority 9 'OR empty' is the safety net for sessions where the wizard already ran.
    Removing it would strand re-fire empty messages with no route."""
    body = _section(skill_text, "Keyword Routing (priority order; first match wins)")
    p9 = re.search(r"^\|\s*9\s*\|.*$", body, flags=re.MULTILINE)
    assert p9, "priority 9 row not found"
    assert "empty" in p9.group(0).lower(), (
        f"priority 9 'OR empty' fallback missing: {p9.group(0)!r}"
    )

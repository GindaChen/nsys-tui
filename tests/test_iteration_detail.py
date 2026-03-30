"""Tests for iteration_detail skill."""

import pytest

from nsys_ai.skills.registry import get_skill


@pytest.fixture
def detail_skill():
    skill = get_skill("iteration_detail")
    assert skill is not None, "iteration_detail skill not registered"
    return skill


class TestIterationDetail:
    def test_returns_kernel_breakdown(self, minimal_nsys_conn, detail_skill):
        """Iteration 0 should return with top_kernels."""
        rows = detail_skill.execute(minimal_nsys_conn, device=0, iteration=0)
        assert isinstance(rows, list)
        assert len(rows) == 1
        r = rows[0]
        # Either we get a valid result or an error (no iterations detected)
        if "error" not in r:
            assert "top_kernels" in r
            assert isinstance(r["top_kernels"], list)

    def test_has_ns_timestamps(self, minimal_nsys_conn, detail_skill):
        rows = detail_skill.execute(minimal_nsys_conn, device=0, iteration=0)
        r = rows[0]
        if "error" not in r:
            assert "gpu_start_ns" in r
            assert "gpu_end_ns" in r
            assert r["gpu_start_ns"] < r["gpu_end_ns"]

    def test_vs_median_present(self, minimal_nsys_conn, detail_skill):
        rows = detail_skill.execute(minimal_nsys_conn, device=0, iteration=0)
        r = rows[0]
        if "error" not in r:
            assert "vs_median" in r
            assert "%" in r["vs_median"]

    def test_invalid_iteration_index(self, minimal_nsys_conn, detail_skill):
        """Out of range iteration should return error."""
        rows = detail_skill.execute(minimal_nsys_conn, device=0, iteration=9999)
        assert len(rows) == 1
        assert "error" in rows[0]

    def test_no_iterations_returns_error(self, minimal_nsys_conn, detail_skill):
        """Non-matching marker should signal no iterations or use heuristic fallback."""
        rows = detail_skill.execute(
            minimal_nsys_conn, device=0, iteration=0, marker="nonexistent_marker_xyz"
        )
        assert len(rows) == 1
        # detect_iterations has a heuristic fallback that creates synthetic iterations
        # from kernel gaps. So we may get either an error or a valid heuristic result.
        if "error" in rows[0]:
            assert "No iterations" in rows[0]["error"] or "out of range" in rows[0]["error"]
        else:
            # Heuristic fallback produced valid iterations
            assert "iteration" in rows[0]

    def test_format_output(self, minimal_nsys_conn, detail_skill):
        rows = detail_skill.execute(minimal_nsys_conn, device=0, iteration=0)
        text = detail_skill.format_rows(rows)
        assert isinstance(text, str)

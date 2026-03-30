"""Tests for kernel_instances skill."""

import pytest

from nsys_ai.skills.registry import get_skill


@pytest.fixture
def ki_skill():
    skill = get_skill("kernel_instances")
    assert skill is not None, "kernel_instances skill not registered"
    return skill


class TestKernelInstances:
    def test_returns_instances_with_ns(self, minimal_nsys_conn, ki_skill):
        rows = ki_skill.execute(minimal_nsys_conn, device=0, limit=5)
        assert isinstance(rows, list)
        assert len(rows) > 0
        r = rows[0]
        assert "start_ns" in r
        assert "end_ns" in r
        assert "duration_ms" in r
        assert r["start_ns"] < r["end_ns"]

    def test_has_both_names(self, minimal_nsys_conn, ki_skill):
        rows = ki_skill.execute(minimal_nsys_conn, device=0, limit=1)
        assert len(rows) > 0
        r = rows[0]
        assert "kernel_name" in r  # demangled
        assert "short_name" in r

    def test_name_filter(self, minimal_nsys_conn, ki_skill):
        """Filtering by name should return only matching kernels."""
        all_rows = ki_skill.execute(minimal_nsys_conn, device=0, limit=100)
        if not all_rows:
            pytest.skip("No kernels in test data")

        # Use a substring from the first kernel's name
        target = all_rows[0]["short_name"][:5]
        filtered = ki_skill.execute(minimal_nsys_conn, device=0, name=target, limit=100)
        assert len(filtered) >= 1
        assert len(filtered) <= len(all_rows)

    def test_limit_respected(self, minimal_nsys_conn, ki_skill):
        rows = ki_skill.execute(minimal_nsys_conn, device=0, limit=2)
        assert len(rows) <= 2

    def test_empty_name_returns_all(self, minimal_nsys_conn, ki_skill):
        rows = ki_skill.execute(minimal_nsys_conn, device=0, name="", limit=100)
        assert isinstance(rows, list)

    def test_sql_injection_safe(self, minimal_nsys_conn, ki_skill):
        """Name containing SQL injection chars should not crash."""
        rows = ki_skill.execute(
            minimal_nsys_conn, device=0, name="'; DROP TABLE StringIds; --", limit=5
        )
        assert isinstance(rows, list)

    def test_format_output(self, minimal_nsys_conn, ki_skill):
        rows = ki_skill.execute(minimal_nsys_conn, device=0, limit=3)
        text = ki_skill.format_rows(rows)
        assert "Kernel Instances" in text

    def test_duckdb_path(self, duckdb_conn, ki_skill):
        rows = ki_skill.execute(duckdb_conn, device=0, limit=3)
        assert isinstance(rows, list)
        if rows:
            assert "start_ns" in rows[0]

import sqlite3

from nsys_ai.skills.base import Skill, _compute_interval_union


def test_compute_interval_union():
    # Empty list
    assert _compute_interval_union([]) == 0

    # Single interval
    assert _compute_interval_union([(100, 200)]) == 100

    # Non-overlapping
    assert _compute_interval_union([(100, 200), (300, 400)]) == 200

    # Partially overlapping
    assert _compute_interval_union([(100, 250), (200, 300)]) == 200

    # Completely contained
    assert _compute_interval_union([(100, 400), (200, 300)]) == 300

    # Multiple overlaps and out of order
    intervals = [(0, 50), (10, 60), (100, 200), (150, 250), (50, 100)]
    # All overlap into [0, 250] contiguous
    assert _compute_interval_union(intervals) == 250


def test_profiler_overhead_probing_and_trimming():
    """Verify that Skill.execute automatically computes overhead_ns handling trim bounds."""
    conn = sqlite3.connect(":memory:")

    # 1. Without the table, overhead_ns should be 0
    dummy_skill = Skill(
        name="dummy",
        title="Dummy",
        description="Dummy description",
        category="utility",
        execute_fn=lambda conn, **kwargs: [kwargs]
    )
    res_no_table = dummy_skill.execute(conn)
    assert res_no_table[0]["overhead_ns"] == 0

    # 2. Create the PROFILER_OVERHEAD table with some intervals
    conn.execute("CREATE TABLE PROFILER_OVERHEAD (start INTEGER, [end] INTEGER)")
    # Interval 1: 100 -> 200
    # Interval 2: 150 -> 300
    # Interval 3: 500 -> 600
    conn.execute("INSERT INTO PROFILER_OVERHEAD VALUES (100, 200), (150, 300), (500, 600)")

    # Executing the skill should now calculate overhead
    # Union of [100, 300] and [500, 600] = 200 + 100 = 300 ns
    res_with_table = dummy_skill.execute(conn)
    assert res_with_table[0]["overhead_ns"] == 300

    # 3. Test trimming window (trim_start_ns=200, trim_end_ns=550)
    # The intervals should be clamped explicitly or ignored:
    # 100->200 (ignored as it ends at 200, wait clamped to start=200, end=200 -> ignored)
    # 150->300 becomes 200->300 (duration 100)
    # 500->600 becomes 500->550 (duration 50)
    # Expected union = 150 ns
    res_trimmed = dummy_skill.execute(conn, trim_start_ns=200, trim_end_ns=550)
    assert res_trimmed[0]["overhead_ns"] == 150

"""Quantify exact true pipeline bubble / idle time percentage on GPUs.

Uses Python-level interval merging (O(n log n) sort + O(n) sweep) instead
of SQL window functions, which are prohibitively slow on large profiles.
"""

from ..base import Skill, _resolve_activity_tables


def _format(rows):
    if not rows:
        return "(No kernel or memory operations detected for bubble analysis)"
    lines = [
        "── Pipeline Bubble (Idle Time) Metrics ──",
        f"{'GPU':<4s}  {'Total Span(ms)':>15s}  {'Active(ms)':>12s}  {'Bubble(ms)':>12s}  {'Bubble %':>10s}",
        "-" * 61,
    ]
    for r in rows:
        gpu_str = str(r["deviceId"])
        lines.append(
            f"{gpu_str:<4s}  {r['total_span_ms']:>15.2f}  {r['active_ms']:>12.2f}  "
            f"{r['bubble_ms']:>12.2f}  {r['bubble_pct']:>9.1f}%"
        )
    return "\n".join(lines)


def _merge_intervals(intervals: list[tuple[int, int]]) -> tuple[int, list[tuple[int, int]]]:
    """Merge overlapping intervals. Returns (total_active_ns, merged_list)."""
    if not intervals:
        return 0, []
    intervals.sort()
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    total = sum(e - s for s, e in merged)
    return total, merged


def _execute(conn, **kwargs):
    tables = _resolve_activity_tables(conn)
    kernel_table = tables.get("kernel", "CUPTI_ACTIVITY_KIND_KERNEL")
    memcpy_table = tables.get("memcpy")

    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")

    # Verify kernel table exists
    try:
        conn.execute(f"SELECT 1 FROM {kernel_table} LIMIT 1")
    except Exception:
        return []

    # --- Fetch kernel intervals per device ---
    params_k = []
    trim_clause_k = ""
    if trim_start is not None and trim_end is not None:
        trim_clause_k = 'WHERE start >= ? AND "end" <= ?'
        params_k = [trim_start, trim_end]

    kernel_rows = conn.execute(
        f'SELECT deviceId, start, "end" FROM {kernel_table} {trim_clause_k}',
        params_k,
    ).fetchall()

    # --- Fetch memcpy intervals per device (if available) ---
    memcpy_rows = []
    if memcpy_table:
        try:
            params_m = []
            trim_clause_m = ""
            if trim_start is not None and trim_end is not None:
                trim_clause_m = 'WHERE start >= ? AND "end" <= ?'
                params_m = [trim_start, trim_end]
            memcpy_rows = conn.execute(
                f'SELECT deviceId, start, "end" FROM {memcpy_table} {trim_clause_m}',
                params_m,
            ).fetchall()
        except Exception:
            pass

    # --- Group intervals by deviceId ---
    from collections import defaultdict

    by_device: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for dev, s, e in kernel_rows:
        by_device[dev].append((s, e))
    for dev, s, e in memcpy_rows:
        by_device[dev].append((s, e))

    if not by_device:
        return []

    # --- Merge intervals and compute bubble metrics per device ---
    results = []
    for dev in sorted(by_device):
        intervals = by_device[dev]
        active_ns, merged = _merge_intervals(intervals)
        if not merged:
            continue
        global_start = merged[0][0]
        global_end = merged[-1][1]
        total_span = global_end - global_start
        bubble_ns = total_span - active_ns
        bubble_pct = 100.0 * bubble_ns / total_span if total_span > 0 else 0.0

        results.append({
            "deviceId": dev,
            "total_span_ms": round(total_span / 1e6, 2),
            "active_ms": round(active_ns / 1e6, 2),
            "bubble_ms": round(bubble_ns / 1e6, 2),
            "bubble_pct": round(bubble_pct, 2),
        })

    return results


SKILL = Skill(
    name="pipeline_bubble_metrics",
    title="Pipeline Bubble Metrics (True Idle Percentage)",
    description=(
        "Quantifies exact true pipeline bubble (idle time percentage) on GPUs. "
        "It merges all overlapping compute kernels and memory transfers to find the "
        "actual sum of time the GPU was doing work, and reports the remaining time "
        "as a pure bubble metric (Bubble %)."
    ),
    category="utilization",
    execute_fn=_execute,
    format_fn=_format,
    tags=["bubble", "idle", "mfu", "utilization", "gap", "pipeline"],
)

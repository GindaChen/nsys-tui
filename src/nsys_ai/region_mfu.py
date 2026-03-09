"""
region_mfu.py — NVTX-region MFU computation for a single Nsight profile.

This module implements a higher-level "region MFU" tool that:

- Finds NVTX ranges by name (supports both NVTX_EVENTS.text and textId → StringIds).
- Attributes CUDA kernels to the chosen NVTX range via CUPTI_ACTIVITY_KIND_RUNTIME
  correlationId (same semantics as nvtx_tree.py, but specialized for a single
  region instead of building a full tree).
- Aggregates time in three ways:
    * wall_time_ns: NVTX GPU span (end - start)
    * kernel_sum_ns: sum of child kernel durations
    * kernel_union_ns: union-of-intervals over child kernels (no double-counting)
- Computes MFU for the region given theoretical FLOPs and peak TFLOPS.

Exports a single public entry point for callers and tools:

    compute_region_mfu(profile_path, nvtx_name, theoretical_flops, ...)

Error handling is explicit and structured. All public functions return dicts
with either data fields or an "error" block:

    {"error": {"code": "...", "message": "..."}}.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from .hardware import get_peak_tflops
from .profile import NsightSchema, get_first_gpu_name, resolve_profile_path


ErrorDict = Dict[str, Any]
RowDict = Dict[str, Any]


def _error(code: str, message: str) -> ErrorDict:
    return {"error": {"code": code, "message": message}}


def _detect_nvtx_text_id(conn: sqlite3.Connection) -> bool:
    """Return True if NVTX_EVENTS uses textId -> StringIds."""
    try:
        cur = conn.execute("PRAGMA table_info(NVTX_EVENTS)")
        cols = [r[1] for r in cur.fetchall()]
        return "textId" in cols
    except Exception:
        return False


def find_nvtx_ranges(
    conn: sqlite3.Connection,
    nvtx_name: str,
    *,
    match_mode: str = "contains",
) -> List[RowDict]:
    """
    Find NVTX ranges whose resolved text matches ``nvtx_name``.

    Returns a list of dicts with:
        text, start_ns, end_ns, global_tid, duration_ns
    ordered by start_ns ascending.
    """
    if not nvtx_name:
        return []

    has_text_id = _detect_nvtx_text_id(conn)
    if match_mode not in ("contains", "exact"):
        match_mode = "contains"

    if has_text_id:
        base_sql = (
            "SELECT COALESCE(n.text, s.value) AS text, "
            "n.start AS start_ns, n.[end] AS end_ns, n.globalTid AS global_tid "
            "FROM NVTX_EVENTS n "
            "LEFT JOIN StringIds s ON n.textId = s.id "
            "WHERE (n.text IS NOT NULL OR s.value IS NOT NULL) "
            "AND n.[end] > n.start "
        )
    else:
        base_sql = (
            "SELECT n.text AS text, n.start AS start_ns, n.[end] AS end_ns, "
            "n.globalTid AS global_tid "
            "FROM NVTX_EVENTS n "
            "WHERE n.text IS NOT NULL AND n.[end] > n.start "
        )

    params: List[Any] = []
    if match_mode == "exact":
        base_sql += "AND text = ? "
        params.append(nvtx_name)
    else:
        base_sql += "AND text LIKE ? "
        params.append(f"%{nvtx_name}%")

    base_sql += "ORDER BY start_ns"

    cur = conn.execute(base_sql, params)
    rows: List[RowDict] = []
    for text, start_ns, end_ns, global_tid in cur.fetchall():
        if text is None:
            continue
        d = int(end_ns) - int(start_ns)
        if d <= 0:
            continue
        rows.append(
            {
                "text": str(text),
                "start_ns": int(start_ns),
                "end_ns": int(end_ns),
                "global_tid": int(global_tid) if global_tid is not None else None,
                "duration_ns": d,
            }
        )
    return rows


def select_nvtx_occurrence(matches: List[RowDict], occurrence_index: int) -> RowDict | ErrorDict:
    """
    Pick the N-th NVTX match (1-based). Returns the row dict or an error.
    """
    if not matches:
        return _error("NVTX_NOT_FOUND", "No NVTX range matched the requested name.")
    if occurrence_index <= 0:
        return _error("INVALID_ARGUMENT", "occurrence_index must be >= 1.")
    if occurrence_index > len(matches):
        return _error(
            "NVTX_OCCURRENCE_OUT_OF_RANGE",
            f"Requested occurrence_index {occurrence_index}, but only {len(matches)} match(es) were found.",
        )
    row = dict(matches[occurrence_index - 1])
    row["occurrence_index"] = occurrence_index
    return row


def get_region_kernels(
    conn: sqlite3.Connection,
    *,
    nvtx_start_ns: int,
    nvtx_end_ns: int,
    global_tid: Optional[int],
    device_id: Optional[int],
) -> List[RowDict]:
    """
    Attribute kernels to an NVTX region via CUPTI_ACTIVITY_KIND_RUNTIME.correlationId.

    Only runtime calls whose CPU span lies fully inside the NVTX CPU span are
    considered. If ``global_tid`` is provided, restricts to that thread to
    mirror nvtx_tree.py semantics. If ``device_id`` is provided, restricts to
    that GPU.
    """
    schema = NsightSchema(conn)
    if not schema.kernel_table:
        return []

    kernel_table = schema.kernel_table

    where_clauses = ["r.start >= ?", "r.[end] <= ?"]
    params: List[Any] = [int(nvtx_start_ns), int(nvtx_end_ns)]

    if global_tid is not None:
        where_clauses.append("r.globalTid = ?")
        params.append(int(global_tid))

    if device_id is not None:
        dev_filter = "AND k.deviceId = ?"
        params.append(int(device_id))
    else:
        dev_filter = ""

    sql = (
        "SELECT r.correlationId AS correlation_id, "
        "k.deviceId AS device_id, k.streamId AS stream_id, "
        "k.start AS start_ns, k.[end] AS end_ns, "
        "(k.[end] - k.start) AS duration_ns, "
        "s.value AS kernel_name "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        f"JOIN {kernel_table} k ON r.correlationId = k.correlationId "
        "LEFT JOIN StringIds s ON k.shortName = s.id "
        "WHERE "
        + " AND ".join(where_clauses)
        + f" {dev_filter} "
        "ORDER BY start_ns"
    )

    cur = conn.execute(sql, params)
    kernels: List[RowDict] = []
    for row in cur.fetchall():
        (
            correlation_id,
            dev,
            stream_id,
            start_ns,
            end_ns,
            duration_ns,
            kernel_name,
        ) = row
        if duration_ns is None:
            duration_ns = int(end_ns) - int(start_ns)
        d = int(duration_ns)
        if d <= 0:
            continue
        kernels.append(
            {
                "correlation_id": int(correlation_id),
                "device_id": int(dev) if dev is not None else None,
                "stream_id": int(stream_id) if stream_id is not None else None,
                "start_ns": int(start_ns),
                "end_ns": int(end_ns),
                "duration_ns": d,
                "name": str(kernel_name) if kernel_name is not None else "",
            }
        )
    return kernels


def _merge_intervals(intervals: List[Tuple[int, int]]) -> int:
    """Return total union length for a list of [start, end] intervals in ns."""
    if not intervals:
        return 0
    intervals_sorted = sorted(intervals, key=lambda x: x[0])
    total = 0
    cur_start, cur_end = intervals_sorted[0]
    for start, end in intervals_sorted[1:]:
        if start <= cur_end:
            if end > cur_end:
                cur_end = end
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return total


def summarize_region_kernel_times(kernels: List[RowDict]) -> RowDict:
    """
    Summarize kernel timings and coverage for a region.

    Returns:
        {
            "kernel_count": int,
            "kernel_sum_ns": int,
            "kernel_union_ns": int,
            "device_ids": [int, ...],
            "stream_ids": [int, ...],
        }
    """
    if not kernels:
        return {
            "kernel_count": 0,
            "kernel_sum_ns": 0,
            "kernel_union_ns": 0,
            "device_ids": [],
            "stream_ids": [],
        }

    kernel_sum_ns = sum(int(k["duration_ns"]) for k in kernels)
    intervals = [(int(k["start_ns"]), int(k["end_ns"])) for k in kernels]
    kernel_union_ns = _merge_intervals(intervals)

    devs = sorted(
        {int(k["device_id"]) for k in kernels if k.get("device_id") is not None}
    )
    streams = sorted(
        {int(k["stream_id"]) for k in kernels if k.get("stream_id") is not None}
    )

    return {
        "kernel_count": len(kernels),
        "kernel_sum_ns": int(kernel_sum_ns),
        "kernel_union_ns": int(kernel_union_ns),
        "device_ids": devs,
        "stream_ids": streams,
    }


def compute_mfu_metrics_for_region(
    *,
    theoretical_flops: float,
    peak_tflops: float,
    wall_time_s: float,
    kernel_sum_s: float,
    kernel_union_s: float,
) -> RowDict | ErrorDict:
    """
    Compute MFU metrics for a region given FLOPs and various time bases.
    """
    if theoretical_flops <= 0:
        return _error(
            "INVALID_ARGUMENT",
            "theoretical_flops must be positive (e.g. model_flops_per_step).",
        )
    if peak_tflops <= 0:
        return _error(
            "INVALID_ARGUMENT",
            "peak_tflops must be positive (GPU peak TFLOPS for the chosen precision).",
        )
    if wall_time_s <= 0 or kernel_union_s <= 0 or kernel_sum_s <= 0:
        # We allow caller to decide whether missing kernels is an error; this
        # helper only validates positive times.
        return _error(
            "INVALID_ARGUMENT",
            "All time bases (wall_time_s, kernel_sum_s, kernel_union_s) must be positive.",
        )

    def _achieved(time_s: float) -> float:
        return (theoretical_flops / time_s) / 1e12

    achieved_wall = _achieved(wall_time_s)
    achieved_sum = _achieved(kernel_sum_s)
    achieved_union = _achieved(kernel_union_s)

    mfu_wall = 100.0 * achieved_wall / peak_tflops
    mfu_sum = 100.0 * achieved_sum / peak_tflops
    mfu_union = 100.0 * achieved_union / peak_tflops

    return {
        "theoretical_flops": float(theoretical_flops),
        "peak_tflops": float(peak_tflops),
        "achieved_tflops_wall": round(achieved_wall, 2),
        "achieved_tflops_kernel_sum": round(achieved_sum, 2),
        "achieved_tflops_kernel_union": round(achieved_union, 2),
        "mfu_pct_wall": round(mfu_wall, 2),
        "mfu_pct_kernel_sum": round(mfu_sum, 2),
        "mfu_pct_kernel_union": round(mfu_union, 2),
    }


def _auto_peak_tflops(conn: sqlite3.Connection, explicit_peak: Optional[float]) -> RowDict | ErrorDict:
    """
    Resolve peak_tflops either from explicit value or from profile GPU name.
    """
    if explicit_peak is not None:
        if explicit_peak <= 0:
            return _error(
                "INVALID_ARGUMENT",
                "peak_tflops must be positive when provided explicitly.",
            )
        return {"peak_tflops": float(explicit_peak), "source": "explicit"}

    gpu_name = get_first_gpu_name(conn)
    if not gpu_name:
        return _error(
            "GPU_PEAK_UNKNOWN",
            "Could not determine GPU name from profile; provide peak_tflops explicitly.",
        )
    info = get_peak_tflops(gpu_name)
    if "error" in info or "peak_tflops" not in info:
        return _error(
            "GPU_PEAK_UNKNOWN",
            f"Unknown GPU for peak_tflops lookup: {gpu_name}. Provide peak_tflops explicitly.",
        )
    return {"peak_tflops": float(info["peak_tflops"]), "source": "profile"}


def compute_region_mfu_from_conn(
    conn: sqlite3.Connection,
    profile_path: Optional[str],
    nvtx_name: str,
    theoretical_flops: float,
    *,
    peak_tflops: Optional[float] = None,
    occurrence_index: int = 1,
    device_id: Optional[int] = None,
    match_mode: str = "contains",
) -> RowDict | ErrorDict:
    """
    Compute MFU for a single NVTX region using an existing SQLite connection.

    This is the main entry point for the chat agent, which already holds an
    open read-only connection for the profile.
    """
    if not nvtx_name:
        return _error("INVALID_ARGUMENT", "nvtx_name must be a non-empty string.")
    if theoretical_flops <= 0:
        return _error(
            "INVALID_ARGUMENT",
            "theoretical_flops must be positive (e.g. model_flops_per_step).",
        )

    # 1) Find NVTX ranges
    matches = find_nvtx_ranges(conn, nvtx_name, match_mode=match_mode)
    chosen = select_nvtx_occurrence(matches, occurrence_index)
    if "error" in chosen:
        return chosen

    nvtx_start_ns = int(chosen["start_ns"])
    nvtx_end_ns = int(chosen["end_ns"])
    global_tid = chosen.get("global_tid")

    wall_time_ns = max(0, nvtx_end_ns - nvtx_start_ns)
    if wall_time_ns <= 0:
        return _error(
            "NO_KERNELS_IN_REGION",
            "Selected NVTX range has non-positive duration.",
        )

    # 2) Attribute kernels
    kernels = get_region_kernels(
        conn,
        nvtx_start_ns=nvtx_start_ns,
        nvtx_end_ns=nvtx_end_ns,
        global_tid=global_tid,
        device_id=device_id,
    )
    summary = summarize_region_kernel_times(kernels)
    if summary["kernel_count"] == 0:
        return _error(
            "NO_KERNELS_IN_REGION",
            "No kernels found inside the selected NVTX region (after thread/device filters).",
        )

    wall_time_s = wall_time_ns / 1e9
    kernel_sum_s = summary["kernel_sum_ns"] / 1e9 if summary["kernel_sum_ns"] > 0 else 0.0
    kernel_union_s = (
        summary["kernel_union_ns"] / 1e9 if summary["kernel_union_ns"] > 0 else 0.0
    )

    # 3) Resolve peak_tflops
    peak_info = _auto_peak_tflops(conn, peak_tflops)
    if "error" in peak_info:
        return peak_info
    peak = float(peak_info["peak_tflops"])

    # 4) Compute MFU metrics
    metrics = compute_mfu_metrics_for_region(
        theoretical_flops=theoretical_flops,
        peak_tflops=peak,
        wall_time_s=wall_time_s,
        kernel_sum_s=kernel_sum_s or wall_time_s,
        kernel_union_s=kernel_union_s or wall_time_s,
    )
    if "error" in metrics:
        return metrics

    return {
        "nvtx_name": nvtx_name,
        "matched_text": chosen.get("text", ""),
        "match_mode": match_mode,
        "occurrence_index": int(chosen.get("occurrence_index", occurrence_index)),
        "device_id": int(device_id) if device_id is not None else None,
        "profile_path": profile_path,
        "wall_time_s": round(wall_time_s, 6),
        "gpu_kernel_sum_s": round(kernel_sum_s, 6),
        "gpu_kernel_union_s": round(kernel_union_s, 6),
        "time_basis_used": "wall",
        "kernel_count": summary["kernel_count"],
        "device_ids": summary["device_ids"],
        "stream_ids": summary["stream_ids"],
        **metrics,
    }


def compute_region_mfu(
    profile_path: str,
    nvtx_name: str,
    theoretical_flops: float,
    *,
    peak_tflops: Optional[float] = None,
    occurrence_index: int = 1,
    device_id: Optional[int] = None,
    match_mode: str = "contains",
) -> RowDict | ErrorDict:
    """
    Convenience wrapper that opens the profile in read-only mode.

    This is suitable for scripts and potential future CLI usage. The chat
    agent should prefer :func:`compute_region_mfu_from_conn` to reuse its
    existing connection.
    """
    try:
        sqlite_path = resolve_profile_path(profile_path)
    except RuntimeError as e:
        return _error("PROFILE_NOT_LOADED", f"Profile error: {e}")

    uri = f"file:{sqlite_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return compute_region_mfu_from_conn(
            conn,
            sqlite_path,
            nvtx_name,
            theoretical_flops,
            peak_tflops=peak_tflops,
            occurrence_index=occurrence_index,
            device_id=device_id,
            match_mode=match_mode,
        )
    finally:
        conn.close()


"""Detect Python Garbage Collection and intensive memory free stalls."""

from ..base import Skill, _resolve_activity_tables


def _format(rows):
    if not rows:
        return "(No significant GC or memory allocation stalls detected)"
    lines = [
        "── GC & Memory Allocation Stalls ──",
        f"{'API / Event Name':<30s}  {'Count':>7s}  {'Total(ms)':>10s}  {'Max(ms)':>10s}  {'Avg(ms)':>10s}",
        "-" * 73,
    ]
    for r in rows:
        name = r["event_name"]
        if len(name) > 28:
            name = name[:25] + "..."
        lines.append(
            f"{name:<30s}  {r['occurrences']:>7d}  {r['total_ms']:>10.2f}  "
            f"{r['max_ms']:>10.2f}  {r['avg_ms']:>10.2f}"
        )
    return "\n".join(lines)


def _execute(conn, **kwargs):
    tables = _resolve_activity_tables(conn)
    runtime_table = tables.get("runtime", "CUPTI_ACTIVITY_KIND_RUNTIME")

    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")

    try:
        conn.execute(f"SELECT 1 FROM {runtime_table} LIMIT 1")
    except Exception:
        return []

    # --- Part 1: CUDA Memory APIs from Runtime table ---
    params_r = []
    trim_clause_r = ""
    if trim_start is not None and trim_end is not None:
        trim_clause_r = 'AND r.start >= ? AND r."end" <= ?'
        params_r.extend([trim_start, trim_end])

    sql_runtime = f"""
        SELECT
            s.value AS event_name,
            COUNT(*) AS occurrences,
            ROUND(SUM(r."end" - r.start) / 1e6, 2) AS total_ms,
            ROUND(MAX(r."end" - r.start) / 1e6, 2) AS max_ms,
            ROUND(AVG(r."end" - r.start) / 1e6, 2) AS avg_ms
        FROM {runtime_table} r
        JOIN StringIds s ON r.nameId = s.id
        WHERE (s.value LIKE '%cudaFree%'
           OR s.value LIKE '%cuMemFree%'
           OR s.value LIKE '%cudaMalloc%'
           OR s.value LIKE '%cuMemAlloc%')
          {trim_clause_r}
        GROUP BY s.value
        ORDER BY total_ms DESC
    """
    rows_runtime = conn.execute(sql_runtime, params_r).fetchall()
    cols = ["event_name", "occurrences", "total_ms", "max_ms", "avg_ms"]
    results = [dict(zip(cols, r)) if isinstance(r, tuple) else dict(r) for r in rows_runtime]

    # --- Part 2: NVTX events mentioning GC ---
    has_nvtx = False
    try:
        conn.execute("SELECT 1 FROM NVTX_EVENTS LIMIT 1")
        has_nvtx = True
    except Exception:
        pass

    if has_nvtx:
        params_n = []
        trim_clause_n = ""
        if trim_start is not None and trim_end is not None:
            trim_clause_n = 'AND start >= ? AND "end" <= ?'
            params_n.extend([trim_start, trim_end])

        # Handle both text and textId schemas
        try:
            import duckdb as _ddb
            if isinstance(conn, _ddb.DuckDBPyConnection):
                nvtx_cols = [r[0] for r in conn.execute("DESCRIBE NVTX_EVENTS").fetchall()]
            else:
                nvtx_cols = [r[1] for r in conn.execute("PRAGMA table_info(NVTX_EVENTS)").fetchall()]
        except Exception:
            nvtx_cols = []

        if "textId" in nvtx_cols:
            text_expr = "COALESCE(n.text, sid.value)"
            text_join = "LEFT JOIN StringIds sid ON n.textId = sid.id"
            text_filter = f"({text_expr} LIKE '%GC%' OR {text_expr} LIKE '%garbage%')"
            sql_nvtx = f"""
                SELECT
                    {text_expr} AS event_name,
                    COUNT(*) AS occurrences,
                    ROUND(SUM(n."end" - n.start) / 1e6, 2) AS total_ms,
                    ROUND(MAX(n."end" - n.start) / 1e6, 2) AS max_ms,
                    ROUND(AVG(n."end" - n.start) / 1e6, 2) AS avg_ms
                FROM NVTX_EVENTS n
                {text_join}
                WHERE {text_filter}
                  AND n.eventType IN (59, 60)
                  {trim_clause_n}
                GROUP BY {text_expr}
            """
        else:
            sql_nvtx = f"""
                SELECT
                    text AS event_name,
                    COUNT(*) AS occurrences,
                    ROUND(SUM("end" - start) / 1e6, 2) AS total_ms,
                    ROUND(MAX("end" - start) / 1e6, 2) AS max_ms,
                    ROUND(AVG("end" - start) / 1e6, 2) AS avg_ms
                FROM NVTX_EVENTS
                WHERE (text LIKE '%GC%' OR text LIKE '%garbage%')
                  AND eventType IN (59, 60)
                  {trim_clause_n}
                GROUP BY text
            """
        try:
            rows_nvtx = conn.execute(sql_nvtx, params_n).fetchall()
            results.extend(
                dict(zip(cols, r)) if isinstance(r, tuple) else dict(r)
                for r in rows_nvtx
            )
        except Exception:
            pass  # NVTX query is best-effort

    # Sort combined results by total_ms descending
    results.sort(key=lambda x: x.get("total_ms", 0), reverse=True)
    return results


SKILL = Skill(
    name="gc_impact",
    title="Garbage Collection & Memory Stalls",
    description=(
        "Quantifies CPU stalls caused by memory allocation (cudaMalloc), host-side memory freeing (cudaFree), "
        "and Python Garbage Collection (if annotated via NVTX). "
        "Frequent or long cudaFree calls usually indicate a bottleneck where the CPU is blocking."
    ),
    category="bottlenecks",
    execute_fn=_execute,
    format_fn=_format,
    tags=["gc", "memory", "cudafree", "cudamalloc", "stall", "bubble"],
)

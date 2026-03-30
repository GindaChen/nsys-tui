"""Kernel Instance Details — individual kernel instances with ns timestamps.

Returns the top N longest kernel execution instances with exact
nanosecond start/end timestamps. Essential for building findings.json
evidence overlays on the timeline viewer.
"""

from ..base import Skill, SkillParam


def _execute(conn, **kwargs):
    from ...profile import Profile

    prof = Profile._from_conn(conn)
    if not prof.schema.kernel_table:
        return [{"error": "No kernel table found in profile"}]

    device = int(kwargs.get("device", 0))
    name_filter = kwargs.get("name", "")
    limit = int(kwargs.get("limit", 10))

    # Build WHERE clause safely with parameterized query
    params = [device]
    where_extra = ""
    if name_filter:
        where_extra = "AND d.value LIKE ?"
        params.append(f"%{name_filter}%")

    # Handle trim
    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")
    if trim_start is not None and trim_end is not None:
        where_extra += " AND k.start >= ? AND k.[end] <= ?"
        params.extend([int(trim_start), int(trim_end)])

    params.append(limit)

    sql = f"""
        SELECT d.value AS kernel_name,
               s.value AS short_name,
               k.start AS start_ns,
               k.[end] AS end_ns,
               ROUND((k.[end] - k.start) / 1e6, 3) AS duration_ms,
               k.streamId AS stream_id,
               k.deviceId AS device_id
        FROM {prof.schema.kernel_table} k
        JOIN StringIds s ON k.shortName = s.id
        JOIN StringIds d ON k.demangledName = d.id
        WHERE k.deviceId = ?
          {where_extra}
        ORDER BY (k.[end] - k.start) DESC
        LIMIT ?
    """
    return prof._duckdb_query(sql, params)


def _format(rows):
    if not rows:
        return "(No kernel instances found)"
    if "error" in rows[0]:
        return rows[0]["error"]

    lines = ["── Kernel Instances ──"]
    for r in rows:
        name = r.get("kernel_name", "?")
        if len(name) > 60:
            name = name[:57] + "..."
        lines.append(
            f"  {name:<62s}  {r['duration_ms']:>8.3f}ms  "
            f"stream={r['stream_id']}  "
            f"[{r['start_ns']}..{r['end_ns']}]"
        )
    return "\n".join(lines)


def _to_findings(rows: list[dict]) -> list:
    from nsys_ai.annotation import Finding

    findings = []
    for r in rows:
        if "error" in r:
            continue

        name = r.get("short_name") or r.get("kernel_name", "?")
        is_nccl = "nccl" in name.lower()
        dur_ms = r.get("duration_ms", 0.0)

        if is_nccl:
            label = f"Long NCCL ({dur_ms:.2f}ms)"
            sev = "critical" if dur_ms > 5.0 else "warning"
        else:
            label = f"Hotspot: {name[:30]}"
            sev = "info"

        findings.append(
            Finding(
                type="highlight",
                label=label,
                start_ns=r["start_ns"],
                end_ns=r["end_ns"],
                gpu_id=r.get("device_id", 0),
                stream=str(r.get("stream_id", "0")),
                severity=sev,
                note=f"{r.get('kernel_name', name)[:60]}: {dur_ms:.2f}ms",
            )
        )
    return findings


SKILL = Skill(
    name="kernel_instances",
    title="Kernel Instance Details",
    description=(
        "Returns individual kernel execution instances with exact nanosecond "
        "timestamps (start_ns, end_ns). Use to get precise time ranges for "
        "findings.json evidence overlay on the timeline viewer. "
        "Includes both demangled (kernel_name) and short (short_name) identifiers."
    ),
    category="kernels",
    execute_fn=_execute,
    format_fn=_format,
    to_findings_fn=_to_findings,
    params=[
        SkillParam("device", "GPU device ID", "int", False, 0),
        SkillParam("name", "Kernel name substring filter (demangled)", "str", False, ""),
        SkillParam("limit", "Max instances to return", "int", False, 10),
    ],
    tags=["kernel", "instance", "timestamp", "evidence", "finding", "nanosecond"],
)

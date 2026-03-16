"""Compute/Communication overlap breakdown.

Uses overlap_analysis() from overlap.py to quantify how much GPU compute
overlaps with NCCL communication.  This is a Python-level skill (execute_fn)
rather than a SQL skill because it needs interval merging and intersection
logic that can't be expressed in a single SQL query.
"""

from ..base import Skill, SkillParam


def _execute(conn, **kwargs):
    from ...overlap import overlap_analysis
    from ...profile import Profile

    prof = Profile._from_conn(conn)
    device = int(kwargs.get("device", 0))
    result = overlap_analysis(prof, device, trim=None)
    if "error" in result:
        return [result]
    return [result]


def _format(rows):
    if not rows:
        return "(No overlap data available)"
    r = rows[0]
    if "error" in r:
        return f"(Overlap analysis: {r['error']})"
    return (
        "── Compute/Communication Overlap ──\n"
        f"  Total span:    {r['total_ms']:.1f}ms\n"
        f"  Compute only:  {r['compute_only_ms']:.1f}ms\n"
        f"  NCCL only:     {r['nccl_only_ms']:.1f}ms\n"
        f"  Overlap:       {r['overlap_ms']:.1f}ms"
        f" ({r['overlap_pct']}% of NCCL overlapped)\n"
        f"  Idle:          {r['idle_ms']:.1f}ms\n"
        f"  Kernels:       {r['compute_kernels']} compute"
        f" + {r['nccl_kernels']} NCCL"
    )


SKILL = Skill(
    name="overlap_breakdown",
    title="Compute/Communication Overlap Breakdown",
    description=(
        "Quantifies how much GPU compute overlaps with NCCL communication. "
        "Shows compute-only, NCCL-only, overlap, and idle time. "
        "overlap_pct > 60% means NCCL is well-hidden behind compute."
    ),
    category="communication",
    execute_fn=_execute,
    format_fn=_format,
    params=[SkillParam("device", "GPU device ID", "int", False, 0)],
    tags=["overlap", "nccl", "compute", "communication", "distributed"],
)

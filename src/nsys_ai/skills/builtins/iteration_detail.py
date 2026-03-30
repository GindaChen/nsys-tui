"""Per-iteration kernel breakdown — drill into a specific training iteration.

Given an iteration index N, returns the top kernels, NCCL stats, and
comparison to the median iteration. Use after `iteration_timing` identifies
a slow iteration.
"""

import logging

from ..base import Skill, SkillParam

_log = logging.getLogger(__name__)


def _execute(conn, **kwargs):
    from ...overlap import detect_iterations
    from ...profile import Profile

    prof = Profile._from_conn(conn)
    device = int(kwargs.get("device", 0))
    iteration = int(kwargs.get("iteration", 0))
    marker = kwargs.get("marker", "sample_0")

    # Support --trim passthrough
    trim = None
    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")
    if trim_start is not None and trim_end is not None:
        trim = (int(trim_start), int(trim_end))

    # 1. Get all iterations
    iters = detect_iterations(prof, device, trim=trim, marker=marker)
    if not iters:
        return [{"error": "No iterations detected (NVTX marker not found)"}]
    if iteration < 0 or iteration >= len(iters):
        return [{"error": f"Iteration {iteration} out of range (0-{len(iters) - 1})"}]

    target = iters[iteration]

    # gpu_start_s / gpu_end_s are in SECONDS → convert to ns
    start_ns = int(target["gpu_start_s"] * 1e9)
    end_ns = int(target["gpu_end_s"] * 1e9)

    # 2. Run aggregate_kernels within this iter's time range
    kernels = prof.aggregate_kernels(device=device, trim=(start_ns, end_ns), limit=10)

    total_kernel_ns = sum(k.get("total_ns", 0) for k in kernels)
    top_kernels = []
    for k in kernels[:5]:
        pct = round(k["total_ns"] / total_kernel_ns * 100, 1) if total_kernel_ns > 0 else 0
        name = k.get("demangled", "?")
        if len(name) > 60:
            name = name[:57] + "..."
        top_kernels.append({
            "name": name,
            "total_ms": round(k["total_ns"] / 1e6, 2),
            "count": k["count"],
            "pct": pct,
        })

    # 3. Compute vs_median
    durs = [it["duration_ms"] for it in iters]
    sorted_durs = sorted(durs)
    median = sorted_durs[len(sorted_durs) // 2]
    vs_median = round((target["duration_ms"] - median) / median * 100, 1) if median > 0 else 0

    return [{
        "iteration": iteration,
        "total_iterations": len(iters),
        "duration_ms": target["duration_ms"],
        "gpu_start_ns": start_ns,
        "gpu_end_ns": end_ns,
        "top_kernels": top_kernels,
        "kernel_count": target.get("kernel_count", 0),
        "nccl_count": target.get("nccl_count", 0),
        "compute_ms": target.get("compute_ms", 0),
        "vs_median": f"{'+' if vs_median >= 0 else ''}{vs_median}%",
        "median_ms": round(median, 2),
    }]


def _format(rows):
    if not rows:
        return "(No iteration detail)"
    r = rows[0]
    if "error" in r:
        return r["error"]

    lines = [f"── Iteration {r['iteration']} Detail (of {r['total_iterations']}) ──"]
    lines.append(f"  Duration:   {r['duration_ms']:.1f}ms  (vs median: {r['vs_median']})")
    lines.append(f"  Median:     {r['median_ms']:.1f}ms")
    lines.append(f"  Kernels:    {r['kernel_count']}  (NCCL: {r['nccl_count']})")
    lines.append(f"  Compute:    {r['compute_ms']:.1f}ms")
    lines.append(f"  Time range: [{r['gpu_start_ns']}..{r['gpu_end_ns']}]")

    lines.append("")
    lines.append("  Top Kernels:")
    for k in r.get("top_kernels", []):
        lines.append(
            f"    {k['name'][:50]:<52s}  {k['total_ms']:>8.1f}ms  "
            f"×{k['count']}  ({k['pct']}%)"
        )

    return "\n".join(lines)


SKILL = Skill(
    name="iteration_detail",
    title="Per-Iteration Kernel Breakdown",
    description=(
        "Drill into a specific training iteration: shows top kernels, NCCL stats, "
        "compute time, and comparison to the median iteration. "
        "Use after `iteration_timing` identifies a slow iteration."
    ),
    category="nvtx",
    execute_fn=_execute,
    format_fn=_format,
    params=[
        SkillParam("iteration", "Iteration index (0-based)", "int", True, 0),
        SkillParam("device", "GPU device ID", "int", False, 0),
        SkillParam("marker", "NVTX marker for iteration boundary detection", "str", False, "sample_0"),
    ],
    tags=["iteration", "detail", "breakdown", "variance", "nvtx", "drill-down"],
)

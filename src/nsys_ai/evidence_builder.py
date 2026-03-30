"""
evidence_builder.py — Convert profile analysis into visual Finding overlays.

Each method queries individual kernel instances (not aggregates)
to produce findings with exact nanosecond timestamps for timeline overlay.
"""

import logging
import statistics

from .annotation import EvidenceReport, Finding
from .profile import Profile

_log = logging.getLogger(__name__)


class EvidenceBuilder:
    """Generates findings from a profile using direct SQL queries.

    Usage::

        with Profile("profile.sqlite") as prof:
            builder = EvidenceBuilder(prof, device=0)
            report = builder.build()
            # report.findings is a list of Finding objects
    """

    def __init__(
        self,
        prof: Profile,
        device: int = 0,
        trim: tuple[int, int] | None = None,
    ):
        self.prof = prof
        self.device = device
        self.trim = trim or tuple(prof.meta.time_range)

    # Map analyzer_name -> (skill_name, params)
    _SKILL_PIPELINE = {
        "slow_iterations": ("iteration_timing", {}),
        "idle_gaps": ("gpu_idle_gaps", {"limit": 5, "min_gap_ns": 1000000}),
        "nccl_stalls": ("kernel_instances", {"name": "nccl", "limit": 3}),
        "kernel_hotspots": ("kernel_instances", {"limit": 3}),
        "overlap_ratio": ("overlap_breakdown", {}),
        "memory_anomalies": ("memory_bandwidth", {"limit": 5}),
        "h2d_spikes": ("h2d_distribution", {}),
    }

    def build(self, only: list[str] | None = None) -> EvidenceReport:
        """Run analyzers (via skill pipeline) and return a combined EvidenceReport.

        Args:
            only: If provided, run only the named analyzers.
                  Valid names: slow_iterations, idle_gaps, nccl_stalls,
                  kernel_hotspots, overlap_ratio, memory_anomalies, h2d_spikes.
                  If None, run all analyzers.
        """
        from .skills.registry import get_skill

        findings: list[Finding] = []
        for analyzer_name, (skill_name, params) in self._SKILL_PIPELINE.items():
            if only is not None and analyzer_name not in only:
                continue
                
            try:
                skill = get_skill(skill_name)
                # Map runtime parameters into skill args
                kwargs = {**params, "device": self.device}
                if self.trim:
                    kwargs["trim_start_ns"] = self.trim[0]
                    kwargs["trim_end_ns"] = self.trim[1]
                
                # Use DuckDB if available, fallback to SQLite
                conn = self.prof.db if self.prof.db is not None else self.prof.conn
                rows = skill.execute(conn, **kwargs)
                if skill.to_findings_fn:
                    findings.extend(skill.to_findings_fn(rows))
            except Exception as e:
                import traceback
                print(f"FAILED TO RUN {analyzer_name}: {e}")
                traceback.print_exc()
                _log.debug("Analyzer %s (skill %s) failed", analyzer_name, skill_name, exc_info=True)
                
        return EvidenceReport(
            title="Auto-Analysis",
            profile_path=getattr(self.prof, "path", ""),
            findings=findings,
        )


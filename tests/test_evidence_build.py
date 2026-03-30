"""Tests for evidence build CLI and EvidenceBuilder.only parameter."""

import pytest

from nsys_ai.evidence_builder import EvidenceBuilder


class TestEvidenceBuilderOnly:
    """Test the `only` parameter on EvidenceBuilder.build()."""

    def test_build_all_analyzers(self, minimal_nsys_db_path):
        """build() with no args runs all analyzers without error."""
        from nsys_ai.profile import Profile

        with Profile(minimal_nsys_db_path) as prof:
            builder = EvidenceBuilder(prof, device=0)
            report = builder.build()
            assert report is not None
            assert isinstance(report.findings, list)

    def test_build_selective_only(self, minimal_nsys_db_path):
        """build(only=...) runs only specified analyzers."""
        from nsys_ai.profile import Profile

        with Profile(minimal_nsys_db_path) as prof:
            builder = EvidenceBuilder(prof, device=0)
            report_all = builder.build()
            report_one = builder.build(only=["kernel_hotspots"])
            # Selective should have fewer or equal findings
            assert len(report_one.findings) <= len(report_all.findings)

    def test_build_unknown_analyzer_ignored(self, minimal_nsys_db_path):
        """Unknown analyzer names in `only` are silently ignored."""
        from nsys_ai.profile import Profile

        with Profile(minimal_nsys_db_path) as prof:
            builder = EvidenceBuilder(prof, device=0)
            report = builder.build(only=["nonexistent_analyzer"])
            assert report is not None
            assert len(report.findings) == 0

    def test_backward_compatible(self, minimal_nsys_db_path):
        """build() with no args is backward compatible."""
        from nsys_ai.profile import Profile

        with Profile(minimal_nsys_db_path) as prof:
            builder = EvidenceBuilder(prof, device=0)
            report = builder.build()
            assert report.title == "Auto-Analysis"

    def test_findings_have_required_keys(self, minimal_nsys_db_path):
        """Each Finding has type, label, start_ns."""
        from nsys_ai.profile import Profile

        with Profile(minimal_nsys_db_path) as prof:
            builder = EvidenceBuilder(prof, device=0)
            report = builder.build()
            for f in report.findings:
                assert hasattr(f, "type")
                assert hasattr(f, "label")
                assert hasattr(f, "start_ns")

    def test_pipeline_has_registered_skills(self):
        """All skills in _SKILL_PIPELINE are registered in the registry."""
        from nsys_ai.skills.registry import get_skill
        
        for name, (skill_name, params) in EvidenceBuilder._SKILL_PIPELINE.items():
            skill = get_skill(skill_name)
            assert skill is not None, (
                f"Missing skill {skill_name} for analyzer '{name}'"
            )
